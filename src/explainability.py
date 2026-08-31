import os
import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class FraudXAIExplainer:
    """
    Explainable AI (XAI) engine using SHAP (SHapley Additive exPlanations)
    to generate global feature importance and local transaction decision waterfalls.
    """
    def __init__(self, model_path: str = "models/xgb_kmeans_smote.pkl"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Train models first.")
        
        print(f"Loading trained model for XAI from {model_path}...")
        self.model = joblib.load(model_path)
        # Initialize SHAP Tree Explainer for XGBoost
        self.explainer = shap.TreeExplainer(self.model)

    def explain_instance(self, instance_df: pd.DataFrame):
        """
        Generates local SHAP values for a single transaction instance.
        Returns base value, SHAP values, and feature values for UI rendering.
        """
        shap_values = self.explainer(instance_df)
        return shap_values

    def get_feature_contributions(self, instance_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms local SHAP values into a readable pandas DataFrame for API/UI display.
        """
        shap_values = self.explainer(instance_df)
        vals = shap_values.values[0]
        feature_names = instance_df.columns

        contributions = pd.DataFrame({
            'Feature': feature_names,
            'Feature_Value': instance_df.iloc[0].values,
            'SHAP_Contribution': vals
        }).sort_values(by='SHAP_Contribution', key=abs, ascending=False).reset_index(drop=True)

        return contributions

if __name__ == "__main__":
    from src.data_pipeline import PaySimDataPipeline

    # Load test data sample
    data_path = os.path.join("data", "transactiondata.csv")
    if not os.path.exists(data_path):
        data_path = os.path.join("data", "paysim.csv")

    pipeline = PaySimDataPipeline(raw_filepath=data_path, sample_size=200000)
    _, X_test, _, y_test = pipeline.prepare_pipeline()

    # Initialize Explainer
    xai = FraudXAIExplainer()

    # Find a true fraudulent transaction from test set to audit
    fraud_indices = y_test[y_test == 1].index
    sample_fraud = X_test.loc[[fraud_indices[0]]]

    print("\n[Scenario 2: XAI Interpretability Audit]")
    print(f"Auditing Transaction Index: {fraud_indices[0]}")
    contributions = xai.get_feature_contributions(sample_fraud)
    print("\nLocal Feature Contributions (Impact on Fraud Risk Score):")
    print(contributions.to_string(index=False))