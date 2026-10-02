# Transaction Risk Score & Decision System

This repository contains a fraud-detection and decision-support system for financial transactions. The project trains an XGBoost fraud model on PaySim-style transaction data, exposes a FastAPI assessment endpoint, and includes explainability and streaming evaluation modules for AML monitoring.

## Project overview

The implementation uses:

- A feature-engineering pipeline that filters transaction types and creates balance anomaly features.
- XGBoost benchmarks trained with baseline, SMOTE, and KMeans-SMOTE strategies.
- A FastAPI backend that scores a transaction and returns a risk decision with a narrative explanation.
- SHAP-based interpretability for explaining the contribution of each feature.
- A River-based online validation simulation to evaluate streaming concept drift behavior.

## Repository structure

```text
transaction-risk-score-and-decision-system/
├── .env
├── .gitignore
├── dummy.py
├── readme.md
├── requirements.txt
├── models/
│   ├── xgb_baseline.pkl
│   ├── xgb_smote.pkl
│   └── xgb_kmeans_smote.pkl
├── backend/
│   ├── __init__.py
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── schemas.py
│   ├── data/
│   │   ├── transactiondata.csv
│   │   └── paysim.csv   # optional fallback dataset
│   └── src/
│       ├── __init__.py
│       ├── data_pipeline.py
│       ├── explainability.py
│       ├── model_trainer.py
│       └── streaming_simulation.py
└── venv/
```

## Data pipeline

The data pipeline is implemented in `backend/src/data_pipeline.py` and does the following:

- Loads the PaySim dataset from `backend/data/transactiondata.csv`.
- Filters to transaction types `TRANSFER` and `CASH_OUT`.
- Sorts records by `step` to preserve chronology.
- Engineers features such as:
  - `type_TRANSFER`
  - `errorBalanceOrig`
  - `errorBalanceDest`
  - `isMerchantDest`
  - `zeroBalOrigAfter`
  - `zeroBalDestBefore`
- Drops non-model identifiers like `nameOrig`, `nameDest`, and `isFlaggedFraud`.
- Creates a time-based train/test split.

The main logic is in `PaySimDataPipeline.prepare_pipeline()`.

## Model training

The training logic lives in `backend/src/model_trainer.py`.

It trains and compares:

1. Baseline XGBoost
2. XGBoost + SMOTE
3. XGBoost + KMeans-SMOTE

The primary trained model is saved as:

- `models/xgb_kmeans_smote.pkl`

The script also saves:

- `models/xgb_baseline.pkl`
- `models/xgb_smote.pkl`

### Command to train models

```powershell
python -m backend.src.model_trainer
```

## FastAPI backend

The API server is in `backend/app/main.py`.

### Startup

From the project root:

```powershell
uvicorn backend.app.main:app --reload
```

The application loads the model from `models/xgb_kmeans_smote.pkl` at startup if present.

### Endpoint

```http
POST /api/v1/assess
```

Request body model: `TransactionRequest` in `backend/app/schemas.py`

Example JSON:

```json
{
  "step": 180,
  "amount": 500000.0,
  "oldbalanceOrg": 500000.0,
  "newbalanceOrig": 0.0,
  "oldbalanceDest": 0.0,
  "newbalanceDest": 250000.0,
  "is_transfer": "TRANSFER"
}
```

### Decision logic

The backend computes a probability score and maps it to a decision:

| Risk score | Decision |
| --- | --- |
| > 0.75 | `BLOCK TRANSACTION` |
| > 0.35 | `FLAG FOR MANUAL REVIEW` |
| <= 0.35 | `ALLOW TRANSACTION` |

It also returns:

- `risk_score`
- `decision`
- `primary_driver`
- `audit_summary`
- `feature_vector`

## Explainability

The XAI module is in `backend/src/explainability.py`.

It uses SHAP tree explanations to analyze a transaction prediction and rank which features contributed most to the fraud score. It also supports Gemini-based narrative summaries when `GEMINI_API_KEY` is configured in the environment.

### Run the explainability utility

```powershell
python -m backend.src.explainability
```

## Streaming simulation

The streaming behavior is implemented in `backend/src/streaming_simulation.py`.

It uses River's `ARFClassifier` to simulate online learning and metrics tracking:

- Accuracy
- F1-score
- ROC-AUC

The model predicts before learning each record, which is useful for concept-drift evaluation.

### Run the streaming simulation

```powershell
python -m backend.src.streaming_simulation
```

## Environment setup

### Prerequisites

- Python 3.10+
- A PaySim-style dataset placed under `backend/data/transactiondata.csv`

### Install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Optional API configuration

The audit summary generator expects a Gemini API key. Add this to a local `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
```

## Verified model performance

The benchmark values below were measured by running the current project scripts against the repo’s dataset on 2026-10-02.

### Batch XGBoost benchmark results

| Model | Precision | Recall | F1-score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Baseline XGBoost | 0.9965 | 0.9930 | 0.9947 | 0.9997 |
| XGBoost + Standard SMOTE | 1.0000 | 0.9930 | 0.9965 | 0.9992 |
| XGBoost + KMeans-SMOTE | 0.9965 | 1.0000 | 0.9983 | 1.0000 |

### Streaming evaluation results

The River online simulation was run with the same chronological holdout split and produced:

| Metric | Value |
| --- | ---: |
| Final Online Accuracy | 0.9966 |
| Final Online F1-Score | 0.6895 |
| Final Online ROC-AUC | 0.8611 |

### Notes on interpretation

- The best batch performance on this dataset is from the KMeans-SMOTE model with an F1-score of 0.9983 and ROC-AUC of 1.0000.
- The streaming River model is useful for online drift monitoring, but its current F1-score is lower than the offline XGBoost models in this dataset.
- These values are the exact results from the currently checked-in project configuration and data.

## Dependencies

The project uses the packages listed in `requirements.txt`, including:

- pandas
- numpy
- scikit-learn
- imbalanced-learn
- xgboost
- shap
- river
- fastapi
- python-dotenv
- google-genai
- uvicorn
- joblib

## Notes

- The project is organized around a backend-first architecture, not a Streamlit dashboard.
- The primary model is expected in `models/xgb_kmeans_smote.pkl`.
- If the default dataset path is missing, the scripts fall back to `backend/data/paysim.csv`.
- The trained fraud logic is designed for AML monitoring, anomaly review, and transaction risk triage rather than direct auto-approval.

