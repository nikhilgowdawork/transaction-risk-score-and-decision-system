import sys
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

import joblib
import pandas as pd
from river import metrics, forest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

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
        self.model_comparison: Optional[List[Dict[str, Any]]] = None

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
        state.xai_engine = FraudXAIExplainer(model_path= model_path)
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

    state.pipeline = PaySimDataPipeline(raw_filepath="", sample_size=1000)

    # Load only recent records into RAM.
    # The complete history remains safely stored in SQLite.
    state.simulated_history = database.get_recent_transactions(limit=1000)

    print(f"Persistent transaction count: {database.get_transaction_count()}")
    


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
        "average_risk_score": database.get_average_risk_score(),
        "risk_score_graph": risk_score_graph,
        "recent_transactions": database.get_recent_transactions(limit=10),
    }


@app.get("/api/v1/data/summary")
def get_data_summary() -> Dict[str, Any]:
    raw_fields = [
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]
    engineered_features = [
        {
            "name": "type_TRANSFER",
            "definition": "1 for TRANSFER, 0 for CASH_OUT",
        },
        {
            "name": "errorBalanceOrig",
            "definition": "newbalanceOrig + amount - oldbalanceOrg",
        },
        {
            "name": "errorBalanceDest",
            "definition": "oldbalanceDest + amount - newbalanceDest",
        },
        {
            "name": "isMerchantDest",
            "definition": "1 when the destination account starts with M",
        },
        {
            "name": "zeroBalOrigAfter",
            "definition": "1 when newbalanceOrig is zero",
        },
        {
            "name": "zeroBalDestBefore",
            "definition": "1 when oldbalanceDest is zero",
        },
    ]

    def label_distribution(frame: pd.DataFrame) -> Dict[str, Any]:
        if frame.empty or "isFraud" not in frame:
            return {"total": len(frame), "legitimate": 0, "fraud": 0}

        counts = frame["isFraud"].value_counts()
        legitimate = int(counts.get(0, 0))
        fraud = int(counts.get(1, 0))
        total = legitimate + fraud
        return {
            "total": total,
            "legitimate": legitimate,
            "fraud": fraud,
            "legitimate_percent": round(legitimate * 100 / total, 4) if total else 0,
            "fraud_percent": round(fraud * 100 / total, 4) if total else 0,
            "imbalance_ratio": round(legitimate / fraud, 2) if fraud else None,
        }

    raw_example: Dict[str, Any] = {}
    engineered_example: Dict[str, Any] = {}
    raw_data_path = DATA_DIR / "transactiondata.csv"
    if raw_data_path.exists():
        raw_sample = pd.read_csv(raw_data_path, nrows=1000)
        if "type" in raw_sample:
            eligible_rows = raw_sample[raw_sample["type"].isin(["TRANSFER", "CASH_OUT"])]
            if not eligible_rows.empty:
                raw_row = eligible_rows.iloc[[0]]
                raw_example = {
                    key: value.item() if hasattr(value, "item") else value
                    for key, value in raw_row.iloc[0][raw_fields].items()
                }
                transformed = state.pipeline.engineer_features(raw_row)
                engineered_example = {
                    key: value.item() if hasattr(value, "item") else value
                    for key, value in transformed.iloc[0].items()
                    if key in {feature["name"] for feature in engineered_features}
                }

    return {
        "dataset_metadata": {
            "train_samples": len(state.train_df),
            "test_samples": len(state.test_df),
            "raw_fields": raw_fields,
            "engineered_features": engineered_features,
            "raw_example": raw_example,
            "engineered_example": engineered_example,
            "train_class_distribution": label_distribution(state.train_df),
            "test_class_distribution": label_distribution(state.test_df),
        },
    }


@app.get("/api/v1/models/performance")
def get_model_performance() -> Dict[str, Any]:
    if state.model_comparison is not None:
        return {
            "available": True,
            "source": "All three saved XGBoost models evaluated on the same untouched test set.",
            "classification_threshold": 0.5,
            "test_transactions": len(state.test_df),
            "models": state.model_comparison,
        }

    if state.test_df.empty or "isFraud" not in state.test_df:
        raise HTTPException(
            status_code=503,
            detail="Cannot compare models: the labeled test dataset is unavailable.",
        )

    model_specs = [
        ("Baseline XGBoost", "xgb_baseline.pkl", "Scale-positive-weighted original training data"),
        ("Standard SMOTE + XGBoost", "xgb_smote.pkl", "Standard SMOTE-resampled training data"),
        ("KMeans-SMOTE + XGBoost", "xgb_kmeans_smote.pkl", "KMeans-SMOTE-resampled training data"),
    ]
    actual = state.test_df["isFraud"].astype(int)
    test_features = state.test_df.drop(columns=["isFraud"])
    comparison = []

    for model_name, artifact_name, training_strategy in model_specs:
        model_path = MODELS_DIR / artifact_name
        if not model_path.is_file():
            raise HTTPException(
                status_code=503,
                detail=f"Cannot compare all models: missing artifact {artifact_name}.",
            )

        model = joblib.load(model_path)
        features = test_features.copy()
        feature_names = getattr(model, "feature_names_in_", None)
        if feature_names is not None:
            for feature in feature_names:
                if feature not in features.columns:
                    features[feature] = 0
            features = features[list(feature_names)]

        probabilities = model.predict_proba(features)[:, 1]
        predicted = (probabilities >= 0.5).astype(int)
        true_negative, false_positive, false_negative, true_positive = (
            int(value)
            for value in confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
        )
        negative_predictions = true_negative + false_positive
        auc = (
            float(roc_auc_score(actual, probabilities))
            if actual.nunique() > 1
            else None
        )

        comparison.append({
            "model_name": model_name,
            "artifact": artifact_name,
            "training_strategy": training_strategy,
            "accuracy": round(float(accuracy_score(actual, predicted)), 4),
            "precision": round(float(precision_score(actual, predicted, zero_division=0)), 4),
            "recall": round(float(recall_score(actual, predicted, zero_division=0)), 4),
            "f1_score": round(float(f1_score(actual, predicted, zero_division=0)), 4),
            "roc_auc": round(auc, 4) if auc is not None else None,
            "specificity": (
                round(true_negative / negative_predictions, 4)
                if negative_predictions
                else None
            ),
            "false_positives": false_positive,
            "missed_fraud": false_negative,
            "confusion_matrix": {
                "true_negative": true_negative,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "true_positive": true_positive,
            },
        })

    state.model_comparison = comparison
    return {
        "available": True,
        "source": "All three saved XGBoost models evaluated on the same untouched test set.",
        "classification_threshold": 0.5,
        "test_transactions": len(actual),
        "models": comparison,
    }


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

    idx = database.get_last_stream_index()

    if idx >= total_records:
        print("River simulation already reached the end of test.csv.")
        return

    try:
        while state.is_simulating and idx < total_records:
            x_row = records[idx].copy()
            y_row = int(x_row.pop("isFraud", 0))

            y_pred_prob = float(
                state.river_model.predict_proba_one(x_row).get(1, 0.0)
            )
            y_pred = state.river_model.predict_one(x_row)
            if y_pred is None:
                y_pred = 0

            state.online_accuracy.update(y_row, y_pred)
            state.online_f1.update(y_row, y_pred)
            state.online_rocauc.update(y_row, y_pred_prob)
            state.river_model.learn_one(x_row, y_row)

            decision = (
                "BLOCK TRANSACTION"
                if y_pred_prob > 0.75
                else ("FLAG FOR REVIEW" if y_pred_prob > 0.35 else "APPROVE")
            )
            record = {
                "index": idx + 1,
                "transaction_id": f"RIVER-{uuid.uuid4().hex[:6].upper()}",
                "step": int(x_row.get("step", 1)),
                "amount": float(x_row.get("amount", 0.0)),
                "risk_score": y_pred_prob,
                "actual_label": y_row,
                "predicted_label": y_pred,
                "decision": decision,
            }

            state.river_history.append(record)
            database.save_transaction(record, source="river")
            database.save_last_stream_index(idx + 1)

            await manager.broadcast({
                "current_transaction": record,
                "total_processed": database.get_transaction_count(),
                "stream_processed": idx + 1,
                "total_dataset_size": total_records,
                "checkpoint": (
                    f"{idx + 1} transaction predictions completed"
                    if (idx + 1) % 1000 == 0 or idx + 1 == total_records
                    else None
                ),
                "online_metrics": {
                    "accuracy": round(state.online_accuracy.get(), 4),
                    "f1_score": round(state.online_f1.get(), 4),
                    "roc_auc": round(state.online_rocauc.get(), 4),
                },
            })

            idx += 1
            await asyncio.sleep(0.02)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        print(f"River stream failed at transaction {idx + 1}: {exc}")
        await manager.broadcast({"stream_error": str(exc)})
    finally:
        state.is_simulating = False

    if idx >= total_records:
        await manager.broadcast({
            "stream_complete": True,
            "stream_processed": idx,
            "total_dataset_size": total_records,
        })


@app.post("/api/v1/stream/start")
async def start_simulation():
    if state.is_simulating:
        return {"status": "already_running"}

    if state.test_df.empty:
        raise HTTPException(status_code=400, detail="Test dataset is empty or missing.")

    if database.get_last_stream_index() >= len(state.test_df):
        return {
            "status": "completed",
            "total_records": len(state.test_df),
        }

    state.is_simulating = True
    state.simulation_task = asyncio.create_task(run_river_simulation_loop())

    return {
        "status": "river_simulation_started",
        "total_records": len(state.test_df),
    }


@app.post("/api/v1/stream/stop")
async def stop_simulation():
    state.is_simulating = False

    if state.simulation_task:
        state.simulation_task.cancel()
        try:
            await state.simulation_task
        except asyncio.CancelledError:
            pass
        state.simulation_task = None

    return {
        "status": "simulation_stopped",
    }


@app.delete("/api/v1/database")
async def reset_database() -> Dict[str, Any]:
    state.is_simulating = False
    if state.simulation_task:
        state.simulation_task.cancel()
        try:
            await state.simulation_task
        except asyncio.CancelledError:
            pass
        state.simulation_task = None

    database.clear_transactions()
    state.simulated_history.clear()
    state.river_history.clear()
    state.online_accuracy = metrics.Accuracy()
    state.online_f1 = metrics.F1()
    state.online_rocauc = metrics.ROCAUC()
    state.river_model = forest.ARFClassifier(n_models=10, seed=42)

    await manager.broadcast({"stream_reset": True})
    return {
        "status": "database_reset",
        "total_evaluated_transactions": database.get_transaction_count(),
    }

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await manager.connect(websocket)

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
