# Transaction Risk Score & Decision System

This repository is a fraud-detection prototype built around PaySim-style transaction data. The project focuses on preprocessing, model benchmarking, SHAP-based explainability, and streaming evaluation for transaction fraud detection.

## Project overview

The active implementation currently includes:

- A preprocessing pipeline that filters PaySim transactions and engineers fraud-focused features.
- A benchmark training workflow for baseline XGBoost, SMOTE + XGBoost, and KMeans-SMOTE + XGBoost.
- Native XGBoost TreeSHAP explainability with optional Gemini audit text.
- A River-based online learning simulation for stream evaluation and concept-drift monitoring.
- A FastAPI dashboard for static-model transaction simulation, persistent history, model metrics, and live River predictions.
- A preprocessing view showing raw-to-engineered features and train/test fraud-class imbalance.
- Supporting configuration and dependency setup for the ML workflow.

## Repository structure

```text
transaction-risk-score-and-decision-system/
├── .env
├── .gitignore
├── dummy.py
├── readme.md
├── requirements.txt
├── backend/
│   ├── __init__.py
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py          # FastAPI routes and streaming websocket
│   │   └── schemas.py       # Pydantic request/response models
│   ├── data/
│   │   ├── transactiondata.csv
│   │   ├── train.csv
│   │   └── test.csv
│   ├── models/
│   │   ├── xgb_baseline.pkl
│   │   ├── xgb_smote.pkl
│   │   └── xgb_kmeans_smote.pkl
│   └── src/
│       ├── __init__.py
│       ├── data_pipeline.py
│       ├── database.py
│       ├── explainability.py
│       ├── model_trainer.py
│       └── streaming_simulation.py
├── frontend/
│   └── index.html           # empty placeholder
└── venv/
```

## Current project state

The repository includes both the model-development workflow and a local FastAPI dashboard.

The dashboard is served from `frontend/index.html` and backed by `backend/app/main.py`. It provides:

- Persistent totals, decision counts, average risk, transaction history, and a risk-score chart.
- Single-sample simulation and manual transaction analysis with SHAP and optional Gemini audit text.
- A memory-only River predict-then-learn stream with live websocket updates and in-session checkpoints.
- Side-by-side holdout metrics for all three saved XGBoost models, including confusion matrices, plus preprocessing/class-imbalance views.
- A simple static-model risk-score chart; transaction deep-analysis from the recent-transactions context menu; and a line-chart comparison of all three static models across evaluation metrics.
- River predictions, progress, metrics, and learner state exist only in RAM and are discarded when the backend stops. They never enter SQLite or the static dashboard.
- The obsolete `backend/data/river_stream.db` from the earlier implementation has been removed.
- Analyze TX opens transaction details and explainability for a dashboard selection; manual-entry fields are hidden until requested.
- A confirmed reset action that clears static dashboard transactions only; it does not change the current in-memory River session.

Run the dashboard from the repository root:

```powershell
uvicorn backend.app.main:app --reload
```

Open `http://127.0.0.1:8000`. The dashboard requires the trained model and processed `backend/data/train.csv` and `backend/data/test.csv` files. Transaction scoring and SHAP details are served without waiting for Gemini; audit generation is an explicit optional request with a 15-second provider timeout. Gemini audit text additionally requires `GEMINI_API_KEY`. River session state resets whenever the backend process stops or restarts.

## Data pipeline

The data preprocessing logic is implemented in `backend/src/data_pipeline.py` and centers on the `PaySimDataPipeline` class.

### What the pipeline does

The class handles the following steps:

- validates that the raw dataset exists
- reads the source CSV with `pandas.read_csv()`
- filters rows to `TRANSFER` and `CASH_OUT` transaction types
- optionally samples the dataset to a configured size
- uses fixed-seed stratified class quotas when sampling, preserving the fraud share as closely as integer class counts allow
- sorts records by `step` to preserve chronology
- engineers fraud-detection features
- saves processed `train.csv` and `test.csv` files to the data directory
- loads previously processed splits when they already exist

### Engineered features

The `engineer_features()` method creates the following features:

- `type_TRANSFER` = 1 when the transaction type is `TRANSFER`, otherwise 0
- `errorBalanceOrig` = `newbalanceOrig + amount - oldbalanceOrg`
- `errorBalanceDest` = `oldbalanceDest + amount - newbalanceDest`
- `isMerchantDest` = 1 when `nameDest` starts with `M`, otherwise 0
- `zeroBalOrigAfter` = 1 when `newbalanceOrig == 0`, otherwise 0
- `zeroBalDestBefore` = 1 when `oldbalanceDest == 0`, otherwise 0

It also drops:

- `nameOrig`
- `nameDest`
- `isFlaggedFraud`

### Entry point

```python
from backend.src.data_pipeline import PaySimDataPipeline

pipeline = PaySimDataPipeline(
    raw_filepath="backend/data/transactiondata.csv",
    sample_size=200000,
    random_state=42,
)

pipeline.prepare_pipeline(test_size=0.2)
train_df, test_df = pipeline.load_processed_data()
```

`prepare_pipeline()` is the main orchestration method and produces the processed training and testing splits.

### Dataset size and sampling

For the checked-in PaySim dataset, the raw file is 493,534,783 bytes and contains 6,362,620 rows. Filtering to `TRANSFER` and `CASH_OUT` retains 2,770,409 rows with 8,213 fraud cases (0.2965%). The current preprocessed files contain 200,000 rows and 561 fraud cases (0.2805%), a measured 0.0160 percentage-point difference from the eligible population. Those existing files predate the stratified sampler. When regenerated, the pipeline now assigns proportional fixed-seed `isFraud` quotas so the 200,000-row sample preserves the eligible fraud share up to whole-row rounding.

## Model training

The benchmark model-training code lives in `backend/src/model_trainer.py` and uses the `FraudModelTrainer` class.

### Included scenarios

The script evaluates three training paths:

1. Baseline XGBoost
2. XGBoost with standard SMOTE resampling
3. XGBoost with KMeans-SMOTE resampling

Each scenario trains a model and saves the artifact to `backend/models/`:

- `xgb_baseline.pkl`
- `xgb_smote.pkl`
- `xgb_kmeans_smote.pkl`

### Run the benchmark

```powershell
python -m backend.src.model_trainer
```

The script prints precision, recall, F1-score, and ROC-AUC for each model.

## Explainability

The transaction explainability logic is implemented in `backend/src/explainability.py`.

### Features of the module

The file includes:

- `FraudXAIExplainer` for loading a trained XGBoost model and calculating native TreeSHAP contributions
- `get_feature_contributions()` for ranking feature contribution values
- `generate_llm_explanation()` to convert SHAP findings into a human-readable narrative using Gemini when `GEMINI_API_KEY` is configured

The default model path is:

```text
backend/models/xgb_kmeans_smote.pkl
```

### Run the explainability script

```powershell
python -m backend.src.explainability
```

The dashboard calculates TreeSHAP values through XGBoost's native contribution predictor, avoiding the deprecated `ntree_limit` compatibility path. Deep analysis is requested separately from the fast scoring operation. Gemini audit generation is optional and may take longer than scoring.

## Streaming simulation

The River-based online learning simulation is implemented in `backend/src/streaming_simulation.py`.

### What it does

- creates an `ARFClassifier` from River
- loads the processed `backend/data/test.csv`
- iterates through rows one at a time
- predicts before learning each sample
- tracks online accuracy, F1-score, and ROC-AUC

This is intended as a concept-drift and online-learning stress test rather than a production inference service.

### Run the streaming simulation

```powershell
python -m backend.src.streaming_simulation
```

## API schema layer

The Pydantic schema definitions live in `backend/app/schemas.py`.

### Current models

```python
class TransactionRequest(BaseModel):
    step: int
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    is_transfer: str
```

```python
class AssessmentResponse(BaseModel):
    risk_score: float
    decision: str
    primary_driver: str
    audit_summary: str
    feature_vector: dict
```

These models are defined, but the API route layer is not implemented in `backend/app/main.py` yet.

## Data files

The repository includes the following data assets:

- `backend/data/transactiondata.csv` — source PaySim-style dataset
- `backend/data/train.csv` — preprocessed training split
- `backend/data/test.csv` — preprocessed holdout split

The root-level `dummy.py` script reads the raw dataset and reports basic counts for the full and filtered transaction sets.

## Environment setup

### Prerequisites

- Python 3.10+
- A project root `.env` file for environment variables

### Install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Optional Gemini configuration

The explainability script checks for a `GEMINI_API_KEY` value in the project root environment file:

```env
GEMINI_API_KEY=your_api_key_here
```

## Dependency summary

The project uses the packages listed in `requirements.txt`, including:

- pandas
- numpy
- scikit-learn
- imbalanced-learn
- xgboost
- shap
- river
- streamlit
- joblib
- google-genai
- python-dotenv

## Notes

- This is primarily a research and model-development repository.
- The core ML workflow is implemented, but the deployment layer is still incomplete.
- The trained model artifacts under `backend/models/` are the main reusable output of the project.
- The repository is not yet a complete web app or dashboard.
