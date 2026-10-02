import os
import sys
import pandas as pd
from river import metrics, forest

# Fix module import path resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_pipeline import PaySimDataPipeline


class StreamingFraudDetector:
    """
    Real-Time Streaming & Concept Drift Simulation Engine using 'River'.
    Simulates row-by-row streaming ingestion along the time-series step sequence.
    """
    def __init__(self):
        # Adaptive Random Forest Classifier optimized for online stream learning
        self.model = forest.ARFClassifier(n_models=10, seed=42)
        # Online tracking metrics
        self.metric_accuracy = metrics.Accuracy()
        self.metric_f1 = metrics.F1()
        self.metric_rocauc = metrics.ROCAUC()

    def run_stream_simulation(self, X_test: pd.DataFrame, y_test: pd.Series, log_interval: int = 5000):
        """
        Feeds test samples sequentially into the streaming model using vectorized dict conversion.
        Updates model weights and metrics dynamically to evaluate concept drift.
        """
        print("\n[Scenario 3: Real-Time Streaming & Concept Drift Stress-Testing]")
        print("Ingesting stream row-by-row using River dynamic adaptive learning...\n")

        # Fast vectorization conversion to avoid pandas .iloc indexing bottleneck
        records = X_test.to_dict(orient='records')
        labels = y_test.astype(int).tolist()
        total_records = len(records)

        for i in range(total_records):
            x_row = records[i]
            y_row = labels[i]

            # 1. Predict risk probability BEFORE learning (Simulates live scoring)
            y_pred_prob = self.model.predict_proba_one(x_row).get(1, 0.0)
            y_pred = self.model.predict_one(x_row)
            if y_pred is None:
                y_pred = 0

            # 2. Update online metrics
            self.metric_accuracy.update(y_row, y_pred)
            self.metric_f1.update(y_row, y_pred)
            self.metric_rocauc.update(y_row, y_pred_prob)

            # 3. Dynamic Weight Update (Online adaptive learning step)
            self.model.learn_one(x_row, y_row)

            # Log dynamic adaptation checkpoints
            if (i + 1) % log_interval == 0:
                print(f" Stream Checkpoint [{i + 1}/{total_records} Records Processed]")
                print(f"   ├─ Online Accuracy: {self.metric_accuracy.get():.4f}")
                print(f"   ├─ Online F1-Score: {self.metric_f1.get():.4f}")
                print(f"   └─ Online ROC-AUC:  {self.metric_rocauc.get():.4f}")

        print("\n==================== STREAMING SIMULATION COMPLETE ====================")
        print(f" Final Online Accuracy: {self.metric_accuracy.get():.4f}")
        print(f" Final Online F1-Score: {self.metric_f1.get():.4f}")
        print(f" Final Online ROC-AUC:  {self.metric_rocauc.get():.4f}")
        print("=========================================================================")


if __name__ == "__main__":
    data_path = os.path.join("backend/data", "transactiondata.csv")
    if not os.path.exists(data_path):
        data_path = os.path.join("backend/data", "paysim.csv")

    pipeline = PaySimDataPipeline(raw_filepath=data_path, sample_size=200000)
    _, X_test, _, y_test = pipeline.prepare_pipeline()

    stream_engine = StreamingFraudDetector()
    stream_engine.run_stream_simulation(X_test, y_test, log_interval=10000)