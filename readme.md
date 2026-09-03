 # Transaction Risk Scoring & Decision Engine

 A hybrid fraud-detection system for highly imbalanced financial transaction data subject to concept drift. It combines offline XGBoost training with KMeans-SMOTE, adaptive River learning, and local SHAP explanations in an interactive Streamlit dashboard.
 ## System Architecture

 ```text
 PaySim CSV Dataset
	 |
	 v
 Data Loading, Filtering & Chronological Sampling
	 |
	 v
 Feature Engineering (12 model features)
	 |
	 +------------------------------+
	 |                              |
	 v                              v
 Chronological Train/Test Split    Chronological Test Stream
	 |                              |
	 v                              v
 KMeans-SMOTE Resampling           River Adaptive Random Forest
	 |                              |
	 v                              v
 XGBoost Fraud Classifier           Online Metrics & Checkpoints
	 |
	 +------------------------------+
	 |
	 v
 Serialized Model (models/*.pkl)
	 |
	 v
 Streamlit Dashboard
   |              |                 |
   v              v                 v
 Risk Score   SHAP Waterfall   Streaming Charts
   |
   v
 Decision Engine: Allow / Flag / Block
 ```

 ## Feature Engineering Matrix

 The pipeline filters the PaySim data to `TRANSFER` and `CASH_OUT` records, sorts records by `step`, removes high-cardinality identifiers, and produces the following model inputs.
 | Feature | Type | Description |
 | --- | --- | --- |
 | `step` | Numeric | Simulation hour of the transaction. |
 | `amount` | Numeric | Transaction amount. |
 | `oldbalanceOrg` | Numeric | Sender balance before the transaction. |
 | `newbalanceOrig` | Numeric | Sender balance after the transaction. |
 | `oldbalanceDest` | Numeric | Receiver balance before the transaction. |
 | `newbalanceDest` | Numeric | Receiver balance after the transaction. |
 | `type_TRANSFER` | Binary | `1` when the transaction type is `TRANSFER`; otherwise `0`. |
 | `errorBalanceOrig` | Numeric | Sender balance discrepancy: `(newbalanceOrig + amount) - oldbalanceOrg`. |
 | `errorBalanceDest` | Numeric | Receiver balance discrepancy: `(oldbalanceDest + amount) - newbalanceDest`. |
 | `isMerchantDest` | Binary | `1` when the destination identifier starts with `M`; otherwise `0`. |
 | `zeroBalOrigAfter` | Binary | `1` when the sender's post-transaction balance is zero. |
 | `zeroBalDestBefore` | Binary | `1` when the receiver's pre-transaction balance is zero. |

 The raw identifiers `nameOrig` and `nameDest`, along with `isFlaggedFraud`, are excluded to reduce overfitting.
 ## Project Directory Structure

 ```text
 transaction-risk-score-and-decision-system/
 |
 +-- app.py                         # Streamlit dashboard and decision engine
 +-- dummy.py                       # Auxiliary script
 +-- readme.md                      # Project documentation
 +-- requirements.txt               # Dependencies
 |
 +-- data/
 |   +-- transactiondata.csv        # Input dataset
 |
 +-- models/
     +-- xgb_baseline.pkl           # Baseline XGBoost artifact
     +-- xgb_smote.pkl              # Standard SMOTE XGBoost artifact
     +-- xgb_kmeans_smote.pkl       # Primary KMeans-SMOTE artifact
 |
 +-- src/
     +-- __init__.py
	+-- data_pipeline.py           # Loading, features, and splits
	+-- model_trainer.py            # Training and benchmarks
	+-- explainability.py           # SHAP utilities
	+-- streaming_simulation.py     # River simulation
 ```

 ## Setup & Installation
**Prerequisites:** Python 3.10+ and a dataset( you can get it from kaggle, dataset "PS_201704392729_1491223214571_PS.csv") which is placed inside "data" folder
 Create and activate a virtual environment:

 ```powershell
 python -m venv .venv
 .\.venv\Scripts\Activate.ps1
 ```
 Install the project dependencies:
 ```powershell
 pip install -r requirements.txt
 ```
 Dependencies include Pandas, NumPy, scikit-learn, XGBoost, imbalanced-learn, SHAP, River, Streamlit, Joblib, and Matplotlib. `lightgbm` is listed but not used by the current trainer.

 ## Running the Project
 ### Launch the Streamlit dashboard
 From the repository root:
 ```powershell
 streamlit run app.py
 ```
 The dashboard loads `models/xgb_kmeans_smote.pkl` first, then falls back to `models/xgb_smote.pkl` and `models/xgb_baseline.pkl`.
 ### Train or regenerate model artifacts
 This runs the pipeline, creates a chronological split, trains three XGBoost variants, evaluates them, and saves artifacts:
 ```powershell
 python -m src.model_trainer
 ```
 Defaults: up to 200,000 records, an 80/20 chronological split, and `data/transactiondata.csv`. It also checks for `data/paysim.csv` when the default filename is absent.

 ### Run the data pipeline directly
 ```powershell
 python -m src.data_pipeline
 ```
 ### Run the streaming simulation
 ```powershell
 python -m src.streaming_simulation
 ```
 The stream predicts before learning from each chronological test record and reports checkpoints every 10,000 records.
 ### Run the SHAP audit utility
 ```powershell
 python -m src.explainability
 ```
 This loads the primary model, selects a fraudulent test transaction, and prints feature contributions.
 ## Streamlit Dashboard
 The dashboard uses Streamlit resource caching, forms, tabs, and session state.
 ### Transaction scenarios
 The sidebar includes pre-filled examples for full account drain, mismatched receiver balance, everyday retail payment, and manual custom input.
 A transaction is scored only after **Evaluate Transaction** is submitted. Inputs include transaction hour, amount, sender and receiver balances, and transaction type.
 ### Decision engine
 The primary XGBoost model returns a fraud probability used as the risk score:
 | Risk score | Decision |
 | --- | --- |
 | Greater than `0.75` | **BLOCK TRANSACTION** |
 | `0.35` through `0.75` | **FLAG FOR REVIEW** |
 | Less than `0.35` | **ALLOW TRANSACTION** |

 ### Dashboard tabs
 1. **Single Transaction Risk Evaluator**: Displays the risk percentage, decision, rationale, and the engineered feature vector sent to the model.
 2. **SHAP Explainable AI**: Generates a local SHAP explanation and renders a waterfall plot showing which features increased or decreased the transaction's fraud risk.
 3. **Streaming Performance Metrics**: Displays River checkpoint data and line charts for online F1-score and ROC-AUC.

 ## Module Breakdown
 ### `src/data_pipeline.py`
 `PaySimDataPipeline` loads and samples data, filters transaction types, sorts by simulation time, engineers balance features, removes identifiers, and returns chronological partitions.
 ### `src/model_trainer.py`
 `FraudModelTrainer` evaluates a class-weighted baseline, standard SMOTE, and KMeans-SMOTE XGBoost. KMeans-SMOTE uses MiniBatchKMeans and falls back to standard SMOTE when cluster sparsity prevents resampling. Models are persisted with Joblib in `models/`.
 ### `src/explainability.py`
 `FraudXAIExplainer` loads the primary model with SHAP's tree explainer and provides local explanations and sorted contribution tables.
 ### `src/streaming_simulation.py`
 `StreamingFraudDetector` uses River's Adaptive Random Forest Classifier with prequential evaluation: predict, update accuracy/F1/ROC-AUC, then learn from the labeled record.
 ### `app.py`
 The entry point loads the model, collects inputs, reproduces feature engineering, applies thresholds, renders SHAP output, and presents streaming checkpoints.

 ## Model Evaluation Benchmarks
 Results below are from the chronological holdout set. F1 balances precision and recall; ROC-AUC measures ranking quality across thresholds.

 | Model | Resampling / Learning mode | F1-score | ROC-AUC |
 | --- | --- | ---: | ---: |
 | XGBoost Baseline | Raw imbalanced data | 0.8120 | 0.9240 |
 | XGBoost + Standard SMOTE | Synthetic minority oversampling | 0.7430 | 0.9410 |
 | XGBoost + KMeans-SMOTE | Cluster-aware oversampling; primary model | **0.8415** | **0.9680** |
 | River Adaptive Random Forest | Online streaming learning | 0.6895 | 0.8611 |

 XGBoost measures batch scoring performance; River measures sequential prediction and adaptation.
 ## Data Notes

 The pipeline expects PaySim-style columns including `step`, `type`, `amount`, sender and receiver balances, `nameOrig`, `nameDest`, and `isFraud`. The included data is for demonstration. Production use would require calibrated thresholds, temporal validation, monitoring, feature-quality checks, governance, and false-positive cost analysis.

