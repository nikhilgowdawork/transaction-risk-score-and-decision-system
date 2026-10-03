import os
import pandas as pd
from river import metrics, forest


class StreamingFraudDetector:
    """
    Real-Time Streaming & Concept Drift Simulation Engine using River.

    Uses the already processed test.csv and simulates
    row-by-row streaming ingestion.
    """

    def __init__(self):
        # Adaptive Random Forest Classifier
        self.model = forest.ARFClassifier(
            n_models=10,
            seed=42
        )

        # Online evaluation metrics
        self.metric_accuracy = metrics.Accuracy()
        self.metric_f1 = metrics.F1()
        self.metric_rocauc = metrics.ROCAUC()

    def run_stream_simulation(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        log_interval: int = 5000
    ):
        """
        Feeds test samples sequentially into the River
        streaming model.

        Prediction happens BEFORE learning from the current
        transaction, simulating real-time fraud detection.
        """

        print(
            "\n[Scenario 3: Real-Time Streaming & "
            "Concept Drift Stress-Testing]"
        )

        print(
            "Ingesting processed test data row-by-row "
            "using River adaptive learning...\n"
        )

        # Convert DataFrame rows into dictionaries.
        # River expects features in dictionary format.
        records = X_test.to_dict(orient="records")

        labels = y_test.astype(int).tolist()

        total_records = len(records)

        for i in range(total_records):

            x_row = records[i]
            y_row = labels[i]

            # -------------------------------------------------
            # 1. PREDICT BEFORE LEARNING
            # -------------------------------------------------

            # Fraud probability
            y_pred_prob = (
                self.model
                .predict_proba_one(x_row)
                .get(1, 0.0)
            )

            # Predicted class
            y_pred = self.model.predict_one(x_row)

            if y_pred is None:
                y_pred = 0

            # -------------------------------------------------
            # 2. UPDATE ONLINE METRICS
            # -------------------------------------------------

            self.metric_accuracy.update(
                y_row,
                y_pred
            )

            self.metric_f1.update(
                y_row,
                y_pred
            )

            self.metric_rocauc.update(
                y_row,
                y_pred_prob
            )

            # -------------------------------------------------
            # 3. LEARN FROM CURRENT TRANSACTION
            # -------------------------------------------------

            self.model.learn_one(
                x_row,
                y_row
            )

            # -------------------------------------------------
            # 4. LOG CHECKPOINT
            # -------------------------------------------------

            if (i + 1) % log_interval == 0:

                print(
                    f" Stream Checkpoint "
                    f"[{i + 1}/{total_records} Records Processed]"
                )

                print(
                    f"   ├─ Online Accuracy: "
                    f"{self.metric_accuracy.get():.4f}"
                )

                print(
                    f"   ├─ Online F1-Score: "
                    f"{self.metric_f1.get():.4f}"
                )

                print(
                    f"   └─ Online ROC-AUC:  "
                    f"{self.metric_rocauc.get():.4f}"
                )

        # -----------------------------------------------------
        # FINAL RESULTS
        # -----------------------------------------------------

        print(
            "\n==================== "
            "STREAMING SIMULATION COMPLETE "
            "===================="
        )

        print(
            f" Final Online Accuracy: "
            f"{self.metric_accuracy.get():.4f}"
        )

        print(
            f" Final Online F1-Score: "
            f"{self.metric_f1.get():.4f}"
        )

        print(
            f" Final Online ROC-AUC:  "
            f"{self.metric_rocauc.get():.4f}"
        )

        print(
            "=============================================================="
        )


if __name__ == "__main__":

    # ---------------------------------------------------------
    # PATHS
    # ---------------------------------------------------------

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

    TEST_PATH = os.path.join(
        DATA_DIR,
        "test.csv"
    )

    # ---------------------------------------------------------
    # LOAD PROCESSED TEST DATA
    # ---------------------------------------------------------

    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(
            f"Processed test data not found at: {TEST_PATH}\n"
            "Run data_pipeline.py once to create train.csv "
            "and test.csv."
        )

    print(f"Loading processed test data from: {TEST_PATH}")

    test_df = pd.read_csv(TEST_PATH)

    print(f"Loaded test data: {test_df.shape}")

    # ---------------------------------------------------------
    # SEPARATE FEATURES AND TARGET
    # ---------------------------------------------------------

    X_test = test_df.drop(
        columns=["isFraud"]
    )

    y_test = test_df["isFraud"]

    print(f"Features: {X_test.shape}")
    print(f"Labels:   {y_test.shape}")

    # ---------------------------------------------------------
    # RUN STREAMING SIMULATION
    # ---------------------------------------------------------

    stream_engine = StreamingFraudDetector()

    stream_engine.run_stream_simulation(
        X_test=X_test,
        y_test=y_test,
        log_interval=10000
    )