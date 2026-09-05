import os
import sys
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Force Python path to recognize project root and backend modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "backend"))

from backend.app.schemas import TransactionRequest, AssessmentResponse
from backend.src.explainability import generate_llm_explanation

load_dotenv()

app = FastAPI(title="AML & Fraud Risk Engine API")

# Enable CORS for Vite React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "xgb_kmeans_smote.pkl")
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        print(f"⚠️ Model not found at {MODEL_PATH}")

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
        driver = "Severe sender account wipeout & discrepancy"
    elif risk_score > 0.35:
        decision = "FLAG FOR MANUAL REVIEW"
        driver = "Moderate balance movement anomaly"
    else:
        decision = "ALLOW TRANSACTION"
        driver = "Standard settlement indicators"

    # Call explainability engine
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