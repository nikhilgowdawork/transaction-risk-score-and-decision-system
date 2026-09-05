import os
import sys
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from imblearn.over_sampling import SMOTE, KMeansSMOTE
from sklearn.cluster import MiniBatchKMeans

# Fix module import path resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_pipeline import PaySimDataPipeline


class FraudModelTrainer:
    """
    Model training and resampling benchmark suite.
    Evaluates baseline, standard SMOTE, and KMeans-SMOTE configurations using XGBoost.
    """
    def __init__(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                 y_train: pd.Series, y_test: pd.Series, models_dir: str = "models"):
        self.X_train = X_train.astype(np.float32)
        self.X_test = X_test.astype(np.float32)
        self.y_train = y_train.astype(int)
        self.y_test = y_test.astype(int)
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)

    def evaluate_performance(self, model, model_name: str) -> dict:
        """Calculates and prints publication-grade classification metrics."""
        y_pred = model.predict(self.X_test)
        y_prob = model.predict_proba(self.X_test)[:, 1]

        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_test, y_pred, average='binary', zero_division=0
        )
        auc_roc = roc_auc_score(self.y_test, y_prob)

        print(f"\n==================== {model_name} Results ====================")
        print(f" Precision: {precision:.4f} (Low False Positive Rate)")
        print(f" Recall:    {recall:.4f}  (High Fraud Detection Rate)")
        print(f" F1-Score:  {f1:.4f}  (Harmonic Balance)")
        print(f" AUC-ROC:   {auc_roc:.4f}")
        print("==============================================================")

        return {
            "model_name": model_name,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc_roc": auc_roc
        }

    def train_baseline(self):
        """Trains baseline XGBoost without resampling."""
        print("\n[Scenario 1A] Training Baseline XGBoost (Imbalanced Data)...")
        fraud_count = self.y_train.sum()
        total_count = len(self.y_train)
        scale_pos_weight = (total_count - fraud_count) / max(fraud_count, 1)
        
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss"
        )
        model.fit(self.X_train, self.y_train)
        joblib.dump(model, os.path.join(self.models_dir, "xgb_baseline.pkl"))
        return model, self.evaluate_performance(model, "Baseline XGBoost")

    def train_standard_smote(self):
        """Trains XGBoost using standard SMOTE oversampling."""
        print("\n[Scenario 1B] Training XGBoost with Standard SMOTE...")
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(self.X_train, self.y_train)

        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss"
        )
        model.fit(X_res, y_res)
        joblib.dump(model, os.path.join(self.models_dir, "xgb_smote.pkl"))
        return model, self.evaluate_performance(model, "Standard SMOTE + XGBoost")

    def train_kmeans_smote(self):
        """Trains XGBoost using KMeans-SMOTE to clean boundary overlap."""
        print("\n[Scenario 1C] Training XGBoost with KMeans-SMOTE (Target Architecture)...")
        
        kmeans_smote = KMeansSMOTE(
            kmeans_estimator=MiniBatchKMeans(n_clusters=50, random_state=42, batch_size=256, n_init=3),
            cluster_balance_threshold=0.01,
            random_state=42
        )
        
        try:
            X_res, y_res = kmeans_smote.fit_resample(self.X_train, self.y_train)
        except Exception as e:
            print(f"KMeans-SMOTE fallback triggered due to cluster sparsity: {e}")
            kmeans_smote = SMOTE(random_state=42)
            X_res, y_res = kmeans_smote.fit_resample(self.X_train, self.y_train)

        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss"
        )
        model.fit(X_res, y_res)
        joblib.dump(model, os.path.join(self.models_dir, "xgb_kmeans_smote.pkl"))
        return model, self.evaluate_performance(model, "KMeans-SMOTE + XGBoost")


if __name__ == "__main__":
    data_path = os.path.join("data", "transactiondata.csv")
    if not os.path.exists(data_path):
        data_path = os.path.join("data", "paysim.csv")

    pipeline = PaySimDataPipeline(raw_filepath=data_path, sample_size=200000)
    X_train, X_test, y_train, y_test = pipeline.prepare_pipeline()

    trainer = FraudModelTrainer(X_train, X_test, y_train, y_test)

    results = []
    _, res_base = trainer.train_baseline()
    _, res_smote = trainer.train_standard_smote()
    _, res_kmeans = trainer.train_kmeans_smote()

    results.extend([res_base, res_smote, res_kmeans])

    df_results = pd.DataFrame(results)
    print("\n==================== RESAMPLING BENCHMARK SUMMARY ====================")
    print(df_results.to_string(index=False))