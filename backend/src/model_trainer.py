import os
import joblib
import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from imblearn.over_sampling import SMOTE, KMeansSMOTE
from sklearn.cluster import MiniBatchKMeans


class FraudModelTrainer:
    """
    Model training and resampling benchmark suite.

    Evaluates:
    1. Baseline XGBoost
    2. Standard SMOTE + XGBoost
    3. KMeans-SMOTE + XGBoost

    Training and testing data are provided by the
    preprocessed PaySimDataPipeline.
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        models_dir: str
    ):
        # Convert feature data to float32 to reduce memory usage
        self.X_train = X_train.astype(np.float32)
        self.X_test = X_test.astype(np.float32)

        # Convert labels to integers
        self.y_train = y_train.astype(int)
        self.y_test = y_test.astype(int)

        # Model storage directory
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)

    def evaluate_performance(self, model, model_name: str) -> dict:
        """
        Evaluate the trained model on the untouched test set.
        """

        # Predicted class: 0 = legitimate, 1 = fraud
        y_pred = model.predict(self.X_test)

        # Fraud probability
        y_prob = model.predict_proba(self.X_test)[:, 1]

        # Calculate classification metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_test,
            y_pred,
            average="binary",
            zero_division=0
        )

        # ROC-AUC uses probability scores
        auc_roc = roc_auc_score(self.y_test, y_prob)

        print(
            f"\n==================== {model_name} Results ===================="
        )
        print(f" Precision: {precision:.4f}")
        print(f" Recall:    {recall:.4f}")
        print(f" F1-Score:  {f1:.4f}")
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
        """
        Train baseline XGBoost on the original imbalanced
        training data using scale_pos_weight.
        """

        print(
            "\n[Scenario 1A] "
            "Training Baseline XGBoost (Imbalanced Data)..."
        )

        fraud_count = self.y_train.sum()
        total_count = len(self.y_train)

        # Give more importance to the minority fraud class
        scale_pos_weight = (
            (total_count - fraud_count)
            / max(fraud_count, 1)
        )

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

        model_path = os.path.join(
            self.models_dir,
            "xgb_baseline.pkl"
        )

        joblib.dump(model, model_path)

        print(f"Model saved to: {model_path}")

        results = self.evaluate_performance(
            model,
            "Baseline XGBoost"
        )

        return model, results

    def train_standard_smote(self):
        """
        Train XGBoost after applying standard SMOTE
        to the training data.
        """

        print(
            "\n[Scenario 1B] "
            "Training XGBoost with Standard SMOTE..."
        )

        smote = SMOTE(random_state=42)

        X_res, y_res = smote.fit_resample(
            self.X_train,
            self.y_train
        )

        print(
            f"Original training shape: {self.X_train.shape}"
        )
        print(
            f"SMOTE training shape:    {X_res.shape}"
        )

        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss"
        )

        model.fit(X_res, y_res)

        model_path = os.path.join(
            self.models_dir,
            "xgb_smote.pkl"
        )

        joblib.dump(model, model_path)

        print(f"Model saved to: {model_path}")

        results = self.evaluate_performance(
            model,
            "Standard SMOTE + XGBoost"
        )

        return model, results

    def train_kmeans_smote(self):
        """
        Train XGBoost using KMeans-SMOTE.

        KMeans-SMOTE attempts to generate synthetic minority
        samples in useful regions of the feature space instead
        of treating the entire minority class uniformly.
        """

        print(
            "\n[Scenario 1C] "
            "Training XGBoost with KMeans-SMOTE "
            "(Target Architecture)..."
        )

        kmeans_smote = KMeansSMOTE(
            kmeans_estimator=MiniBatchKMeans(
                n_clusters=50,
                random_state=42,
                batch_size=256,
                n_init=3
            ),
            cluster_balance_threshold=0.01,
            random_state=42
        )

        try:
            X_res, y_res = kmeans_smote.fit_resample(
                self.X_train,
                self.y_train
            )

        except Exception as e:
            print(
                "KMeans-SMOTE fallback triggered "
                f"due to cluster sparsity: {e}"
            )

            # Fallback to standard SMOTE
            kmeans_smote = SMOTE(random_state=42)

            X_res, y_res = kmeans_smote.fit_resample(
                self.X_train,
                self.y_train
            )

        print(
            f"Original training shape: {self.X_train.shape}"
        )
        print(
            f"Resampled training shape: {X_res.shape}"
        )

        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss"
        )

        model.fit(X_res, y_res)

        model_path = os.path.join(
            self.models_dir,
            "xgb_kmeans_smote.pkl"
        )

        joblib.dump(model, model_path)

        print(f"Model saved to: {model_path}")

        results = self.evaluate_performance(
            model,
            "KMeans-SMOTE + XGBoost"
        )

        return model, results


if __name__ == "__main__":

    CURRENT_DIR = os.path.dirname(
        os.path.abspath(__file__)
    )

    BACKEND_DIR = os.path.abspath(
        os.path.join(CURRENT_DIR, "..")
    )

    DATA_DIR = os.path.join(
        BACKEND_DIR,
        "data"
    )

    MODELS_DIR = os.path.join(
        BACKEND_DIR,
        "models"
    )

    # ---------------------------------------------------------
    # LOAD ALREADY PROCESSED DATA
    # ---------------------------------------------------------

    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Processed training data not found: {train_path}"
        )

    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Processed test data not found: {test_path}"
        )

    print("Loading processed training data...")
    train_df = pd.read_csv(train_path)

    print("Loading processed test data...")
    test_df = pd.read_csv(test_path)

    # ---------------------------------------------------------
    # SEPARATE FEATURES AND TARGET
    # ---------------------------------------------------------

    X_train = train_df.drop(columns=["isFraud"])
    y_train = train_df["isFraud"]

    X_test = test_df.drop(columns=["isFraud"])
    y_test = test_df["isFraud"]

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape:  {X_test.shape}")

    # ---------------------------------------------------------
    # TRAIN MODELS
    # ---------------------------------------------------------

    trainer = FraudModelTrainer(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        models_dir=MODELS_DIR
    )

    results = []

    _, res_base = trainer.train_baseline()
    _, res_smote = trainer.train_standard_smote()
    _, res_kmeans = trainer.train_kmeans_smote()

    results.extend([
        res_base,
        res_smote,
        res_kmeans
    ])

    # ---------------------------------------------------------
    # RESULTS
    # ---------------------------------------------------------

    df_results = pd.DataFrame(results)

    print(
        "\n==================== "
        "RESAMPLING BENCHMARK SUMMARY "
        "===================="
    )

    print(df_results.to_string(index=False))