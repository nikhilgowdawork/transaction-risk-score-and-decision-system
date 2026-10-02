import os
import sys
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv

# Force Python path to recognize project root and backend modules
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Explicitly load .env from backend directory
env_path = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".env"))
load_dotenv(dotenv_path=env_path)

from backend.app.schemas import TransactionRequest, AssessmentResponse
from backend.src.explainability import generate_llm_explanation

app = FastAPI(title="AML & Fraud Risk Engine API")

# Enable CORS for local file origins and development servers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve ML model path
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "xgb_kmeans_smote.pkl")
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"✅ ML Model successfully loaded from {MODEL_PATH}")
    else:
        print(f"⚠️ Model not found at {MODEL_PATH}. Operating in rule-based fallback mode.")

@app.post("/api/v1/assess", response_model=AssessmentResponse)
async def assess_transaction(tx: TransactionRequest):
    type_TRANSFER = 1 if tx.is_transfer == "TRANSFER" else 0
    error_orig = (tx.newbalanceOrig + tx.amount) - tx.oldbalanceOrg
    error_dest = (tx.oldbalanceDest + tx.amount) - tx.newbalanceDest
    zero_orig_after = 1 if (tx.newbalanceOrig == 0 and tx.oldbalanceOrg > 0) else 0
    zero_dest_before = 1 if (tx.oldbalanceDest == 0 and tx.amount > 0) else 0

    feature_dict = {
        'step': tx.step,
        'amount': tx.amount,
        'oldbalanceOrg': tx.oldbalanceOrg,
        'newbalanceOrig': tx.newbalanceOrig,
        'oldbalanceDest': tx.oldbalanceDest,
        'newbalanceDest': tx.newbalanceDest,
        'type_TRANSFER': type_TRANSFER,
        'errorBalanceOrig': error_orig,
        'errorBalanceDest': error_dest,
        'isMerchantDest': 0,
        'zeroBalOrigAfter': zero_orig_after,
        'zeroBalDestBefore': zero_dest_before
    }

    df = pd.DataFrame([feature_dict])

    if model:
        risk_score = float(model.predict_proba(df)[0][1])
    else:
        risk_score = 0.95 if zero_orig_after == 1 else 0.05

    if risk_score > 0.75:
        decision = "BLOCK TRANSACTION"
        driver = "Severe sender account wipeout & balance discrepancy"
    elif risk_score > 0.35:
        decision = "FLAG FOR MANUAL REVIEW"
        driver = "Moderate balance movement anomaly"
    else:
        decision = "ALLOW TRANSACTION"
        driver = "Standard settlement indicators"

    # Call Gemini XAI Explainability Engine
    try:
        audit_note = generate_llm_explanation(risk_score, decision, df)
    except Exception as e:
        audit_note = f"Audit explanation generated with default engine specs. Error: {e}"

    return AssessmentResponse(
        risk_score=risk_score,
        decision=decision,
        primary_driver=driver,
        audit_summary=audit_note,
        feature_vector=feature_dict
    )

# Locate and serve static HTML/JS frontend
possible_frontend_paths = [
    os.path.join(PROJECT_ROOT, "frontend"),
    os.path.join(PROJECT_ROOT, "backend", "frontend"),
    os.path.abspath(os.path.join(CURRENT_DIR, "..", "frontend"))
]

FRONTEND_PATH = None
for path in possible_frontend_paths:
    if os.path.exists(os.path.join(path, "index.html")):
        FRONTEND_PATH = path
        break

if FRONTEND_PATH:
    app.mount("/static", StaticFiles(directory=FRONTEND_PATH), name="static")

    @app.get("/")
    async def read_index():
        return FileResponse(os.path.join(FRONTEND_PATH, "index.html"))
    print(f"✅ Serving HTML Frontend directly from: {FRONTEND_PATH}")
else:
    print("⚠️ Warning: frontend/index.html not found in expected paths.")