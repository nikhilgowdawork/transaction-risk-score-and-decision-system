import os
import sys
import json
import asyncio
import uuid
import random
from typing import Dict, Any, List, Optional
from pathlib import Path
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import pandas as pd
from river import metrics, forest

# ------------------------------------------------------------------------------
# 1. PATH RESOLUTION & MODULE IMPORTS
# ------------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from src.data_pipeline import PaySimDataPipeline
from src.explainability import FraudXAIExplainer
from src.database import DatabaseManager

FRONTEND_DIR = PROJECT_ROOT / "frontend"
MODELS_DIR = BACKEND_DIR / "models"
DATA_DIR = BACKEND_DIR / "data"

# Persistent SQLite database
DATABASE_PATH = DATA_DIR / "fraud_system.db"
database = DatabaseManager(DATABASE_PATH)

env_path = BACKEND_DIR / ".env"
if not env_path.exists():
    env_path = PROJECT_ROOT / ".env"

if env_path.exists():
    load_dotenv(dotenv_path=env_path)

# ------------------------------------------------------------------------------
# 2. GLOBAL STATE & WEBSOCKET MANAGER
# ------------------------------------------------------------------------------
class SystemState:
    def __init__(self):
        self.xai_engine: Optional[FraudXAIExplainer] = None
        self.pipeline: Optional[PaySimDataPipeline] = None
        self.test_df: pd.DataFrame = pd.DataFrame()
        self.train_df: pd.DataFrame = pd.DataFrame()
        self.metrics_data: Dict[str, Any] = {}

        # Runtime cache only. Permanent history is stored in SQLite.
        self.simulated_history: List[Dict[str, Any]] = []

        self.river_model: Optional[forest.ARFClassifier] = None
        self.online_accuracy = metrics.Accuracy()
        self.online_f1 = metrics.F1()
        self.online_rocauc = metrics.ROCAUC()
        self.river_history: List[Dict[str, Any]] = []

        self.is_simulating: bool = False
        self.simulation_task: Optional[asyncio.Task] = None


state = SystemState()


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect(connection)


manager = ConnectionManager()

# ------------------------------------------------------------------------------
# 3. FASTAPI APP & SCHEMAS
# ------------------------------------------------------------------------------
app = FastAPI(
    title="Transaction Risk Scoring & Decision System",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class UnifiedTransactionRequest(BaseModel):
    step: int = Field(default=1, ge=1)
    is_transfer: int = Field(default=1, ge=0, le=1)
    amount: float = Field(..., ge=0.0)
    oldbalanceOrg: float = Field(..., ge=0.0)
    newbalanceOrig: float = Field(..., ge=0.0)
    oldbalanceDest: float = Field(..., ge=0.0)
    newbalanceDest: float = Field(..., ge=0.0)


# ------------------------------------------------------------------------------
# 4. LIFECYCLE HANDLERS
# ------------------------------------------------------------------------------
@app.on_event("startup")
def startup_event():
    model_path = MODELS_DIR / "xgb_kmeans_smote.pkl"
    if model_path.exists():
        state.xai_engine = FraudXAIExplainer(model_filename="xgb_kmeans_smote.pkl")
        print("Initialized XGBoost FraudXAIExplainer Engine.")

    state.river_model = forest.ARFClassifier(n_models=10, seed=42)
    print("Initialized River Adaptive Random Forest Classifier.")

    test_csv = DATA_DIR / "test.csv"
    train_csv = DATA_DIR / "train.csv"

    if test_csv.exists():
        state.test_df = pd.read_csv(test_csv)
        print(f"Loaded test.csv ({len(state.test_df)} records)")

    if train_csv.exists():
        state.train_df = pd.read_csv(train_csv)

    metrics_path = MODELS_DIR / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            state.metrics_data = json.load(f)

    state.pipeline = PaySimDataPipeline(raw_filepath="", sample_size=1000)

    # Load only recent records into RAM.
    # The complete history remains safely stored in SQLite.
    state.simulated_history = database.get_recent_transactions(limit=1000)

    print(f"Persistent transaction count: {database.get_transaction_count()}")
    print(f"River simulation resume index: {database.get_last_stream_index()}")


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/", response_class=FileResponse)
def serve_index():
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html missing.")
    return FileResponse(index_file)


# ------------------------------------------------------------------------------
# 5. STATIC MODEL ENDPOINTS
# ------------------------------------------------------------------------------

@app.get("/api/v1/dashboard/summary")
def get_dashboard_summary() -> Dict[str, Any]:
    """Returns persistent dashboard statistics from SQLite."""

    total_evaluated = database.get_transaction_count()
    decision_counts = database.get_decision_counts()

    # Full graph is loaded from the database, not from a temporary Python list.
    risk_score_graph = database.get_risk_score_graph()

    return {
        "system_status": "ONLINE",
        "active_model": "XGBoost + KMeans-SMOTE",
        "total_evaluated_transactions": total_evaluated,
        "decision_counts": decision_counts,
        "risk_score_graph": risk_score_graph,
        "recent_transactions": database.get_recent_transactions(limit=10),
    }


@app.get("/api/v1/data/summary")
def get_data_summary() -> Dict[str, Any]:
    return {
        "dataset_metadata": {
            "train_samples": len(state.train_df),
            "test_samples": len(state.test_df),
            "raw_fields": [
                "type",
                "amount",
                "oldbalanceOrg",
                "newbalanceOrig",
                "oldbalanceDest",
                "newbalanceDest",
            ],
            "engineered_features": [
                "errorBalanceOrig",
                "errorBalanceDest",
                "isMerchantDest",
                "zeroBalOrigAfter",
                "zeroBalDestBefore",
                "type_TRANSFER",
            ],
        }
    }


@app.get("/api/v1/models/performance")
def get_model_performance() -> Dict[str, Any]:
    if state.metrics_data:
        return state.metrics_data
    return {"status": "warning", "message": "metrics.json not found."}


@app.post("/api/v1/transaction/analyze")
def analyze_transaction(payload: UnifiedTransactionRequest) -> Dict[str, Any]:
    if not state.xai_engine:
        raise HTTPException(status_code=500, detail="XAI Engine checkpoint is missing.")

    raw_input_df = pd.DataFrame([{
        "step": payload.step,
        "type": "TRANSFER" if payload.is_transfer == 1 else "CASH_OUT",
        "amount": payload.amount,
        "oldbalanceOrg": payload.oldbalanceOrg,
        "newbalanceOrig": payload.newbalanceOrig,
        "oldbalanceDest": payload.oldbalanceDest,
        "newbalanceDest": payload.newbalanceDest,
        "nameDest": "C123456789",
    }])

    input_data = state.pipeline.engineer_features(raw_input_df)

    if hasattr(state.xai_engine.model, "feature_names_in_"):
        input_data = input_data[state.xai_engine.model.feature_names_in_]

    prob = float(state.xai_engine.model.predict_proba(input_data)[0][1])

    decision = (
        "BLOCK TRANSACTION"
        if prob > 0.75
        else ("FLAG FOR REVIEW" if prob > 0.35 else "APPROVE")
    )

    contributions = state.xai_engine.get_feature_contributions(input_data)

    llm_audit = state.xai_engine.generate_llm_explanation(
        risk_score=prob,
        decision=decision,
        input_df=input_data,
        shap_df=contributions,
    )

    record = {
        "transaction_id": f"TX-{uuid.uuid4().hex[:8].upper()}",
        "input_raw": payload.dict(),
        "input_transformed": input_data.to_dict(orient="records")[0],
        "risk_score": prob,
        "decision": decision,
        "shap_contributions": contributions.to_dict(orient="records"),
        "llm_audit": llm_audit,
    }

    # Runtime cache
    state.simulated_history.append(record)

    # Permanent storage
    database.save_transaction(record, source="analyze")

    return record


@app.post("/api/v1/transaction/simulate-single")
def simulate_single_transaction() -> Dict[str, Any]:
    if state.test_df.empty:
        raise HTTPException(status_code=400, detail="Test dataset is empty or missing.")

    if not state.xai_engine:
        raise HTTPException(status_code=500, detail="XAI Engine model checkpoint is missing.")

    is_fraud_sample = random.randint(1, 10) == 1

    if (
        is_fraud_sample
        and "isFraud" in state.test_df.columns
        and (state.test_df["isFraud"] == 1).any()
    ):
        sampled_row = state.test_df[state.test_df["isFraud"] == 1].sample(n=1).iloc[0]
    else:
        if "isFraud" in state.test_df.columns and (state.test_df["isFraud"] == 0).any():
            sampled_row = state.test_df[state.test_df["isFraud"] == 0].sample(n=1).iloc[0]
        else:
            sampled_row = state.test_df.sample(n=1).iloc[0]

    raw_df = pd.DataFrame([sampled_row.to_dict()])

    actual_label = (
        int(raw_df["isFraud"].values[0])
        if "isFraud" in raw_df.columns
        else None
    )

    if "type" in raw_df.columns:
        input_data = state.pipeline.engineer_features(raw_df)
    else:
        input_data = raw_df.drop(columns=["isFraud"], errors="ignore")

    if hasattr(state.xai_engine.model, "feature_names_in_"):
        for col in state.xai_engine.model.feature_names_in_:
            if col not in input_data.columns:
                input_data[col] = 0

        input_data = input_data[state.xai_engine.model.feature_names_in_]

    prob = float(state.xai_engine.model.predict_proba(input_data)[0][1])

    decision = (
        "BLOCK TRANSACTION"
        if prob > 0.75
        else ("FLAG FOR REVIEW" if prob > 0.35 else "APPROVE")
    )

    contributions = state.xai_engine.get_feature_contributions(input_data)

    llm_audit = state.xai_engine.generate_llm_explanation(
        risk_score=prob,
        decision=decision,
        input_df=input_data,
        shap_df=contributions,
    )

    record = {
        "transaction_id": f"SIM-{uuid.uuid4().hex[:8].upper()}",
        "ground_truth_label": actual_label,
        "input_raw": {
            "step": int(sampled_row.get("step", 1)),
            "amount": float(sampled_row.get("amount", 0.0)),
            "oldbalanceOrg": float(sampled_row.get("oldbalanceOrg", 0.0)),
            "newbalanceOrig": float(sampled_row.get("newbalanceOrig", 0.0)),
            "oldbalanceDest": float(sampled_row.get("oldbalanceDest", 0.0)),
            "newbalanceDest": float(sampled_row.get("newbalanceDest", 0.0)),
            "type": str(sampled_row.get("type", "TRANSFER")),
        },
        "input_transformed": input_data.to_dict(orient="records")[0],
        "risk_score": prob,
        "decision": decision,
        "shap_contributions": contributions.to_dict(orient="records"),
        "llm_audit": llm_audit,
    }

    state.simulated_history.append(record)

    # NEW: persist simulated transaction permanently
    database.save_transaction(record, source="simulate-single")

    return record


# ------------------------------------------------------------------------------
# 6. STREAMING ONLINE MODEL ENDPOINTS (RIVER ADAPTIVE RANDOM FOREST)
# ------------------------------------------------------------------------------

async def run_river_simulation_loop():
    """
    Executes predict-then-learn online streaming.

    The last processed dataset index is stored in SQLite so the simulation
    can continue from the previous position after the backend restarts.
    """

    if state.test_df.empty:
        print("River simulation canceled: test.csv is empty.")
        return

    records = state.test_df.to_dict(orient="records")
    total_records = len(records)

    # NEW: resume from the last persisted position
    idx = database.get_last_stream_index()

    if idx >= total_records:
        print("River simulation already reached the end of test.csv.")
        return

    while state.is_simulating and idx < total_records:
        x_row = records[idx].copy()
        y_row = int(x_row.pop("isFraud", 0))

        # 1. Predict before learning
        y_pred_prob = float(
            state.river_model.predict_proba_one(x_row).get(1, 0.0)
        )

        y_pred = state.river_model.predict_one(x_row)

        if y_pred is None:
            y_pred = 0

        # 2. Update running metrics
        state.online_accuracy.update(y_row, y_pred)
        state.online_f1.update(y_row, y_pred)
        state.online_rocauc.update(y_row, y_pred_prob)

        # 3. Learn from current transaction
        state.river_model.learn_one(x_row, y_row)

        decision = (
            "BLOCK TRANSACTION"
            if y_pred_prob > 0.75
            else ("FLAG FOR REVIEW" if y_pred_prob > 0.35 else "APPROVE")
        )

        tx_id = f"RIVER-{uuid.uuid4().hex[:6].upper()}"

        record = {
            "index": idx + 1,
            "transaction_id": tx_id,
            "step": int(x_row.get("step", 1)),
            "amount": float(x_row.get("amount", 0.0)),
            "risk_score": y_pred_prob,
            "actual_label": y_row,
            "predicted_label": y_pred,
            "decision": decision,
        }

        state.river_history.append(record)

        # NEW: persist River transaction
        database.save_transaction(record, source="river")

        # NEW: persist next dataset position
        database.save_last_stream_index(idx + 1)

        risk_graph_data = [
            {
                "index": tx["index"],
                "transaction_id": tx["transaction_id"],
                "risk_score": round(tx["risk_score"], 4),
                "decision": tx["decision"],
            }
            for tx in state.river_history[-100:]
        ]

        broadcast_payload = {
            "current_transaction": record,

            # Persistent global count
            "total_processed": database.get_transaction_count(),

            "total_dataset_size": total_records,

            "online_metrics": {
                "accuracy": round(state.online_accuracy.get(), 4),
                "f1_score": round(state.online_f1.get(), 4),
                "roc_auc": round(state.online_rocauc.get(), 4),
            },

            "risk_graph_data": risk_graph_data,
        }

        await manager.broadcast(broadcast_payload)

        idx += 1
        await asyncio.sleep(0.05)


@app.post("/api/v1/stream/start")
def start_simulation():
    if state.is_simulating:
        return {"status": "already_running"}

    if database.get_last_stream_index() >= len(state.test_df):
        return {
            "status": "completed",
            "message": "All test transactions have already been processed.",
        }

    state.is_simulating = True
    state.simulation_task = asyncio.create_task(run_river_simulation_loop())

    return {
        "status": "river_simulation_started",
        "resume_from_transaction": database.get_last_stream_index() + 1,
        "total_records": len(state.test_df),
        "persistent_total_transactions": database.get_transaction_count(),
    }


@app.post("/api/v1/stream/stop")
def stop_simulation():
    state.is_simulating = False

    if state.simulation_task:
        state.simulation_task.cancel()
        state.simulation_task = None

    return {
        "status": "simulation_stopped",
        "persistent_total_transactions": database.get_transaction_count(),
        "last_stream_index": database.get_last_stream_index(),
    }


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await manager.connect(websocket)

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
