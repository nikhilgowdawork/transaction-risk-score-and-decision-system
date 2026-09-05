import os
import textwrap
import joblib
import pandas as pd
import numpy as np
import shap
from google import genai
from google.genai import types

class FraudXAIExplainer:
    """
    Explainable AI (XAI) engine using SHAP (SHapley Additive exPlanations)
    to generate feature contributions for XGBoost model predictions.
    """
    def __init__(self, model_path: str = "models/xgb_kmeans_smote.pkl"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}.")
        
        print(f"Loading model for XAI from {model_path}...")
        self.model = joblib.load(model_path)
        self.explainer = shap.TreeExplainer(self.model)

    def explain_instance(self, instance_df: pd.DataFrame):
        """Generates raw SHAP Explanation object for visualization."""
        return self.explainer(instance_df)

    def get_feature_contributions(self, instance_df: pd.DataFrame) -> pd.DataFrame:
        """Transforms local SHAP values into a readable pandas DataFrame for API response."""
        shap_values = self.explainer(instance_df)
        vals = shap_values.values[0]
        feature_names = instance_df.columns

        contributions = pd.DataFrame({
            'Feature': feature_names,
            'Feature_Value': instance_df.iloc[0].values,
            'SHAP_Contribution': vals
        }).sort_values(by='SHAP_Contribution', key=abs, ascending=False).reset_index(drop=True)

        return contributions


def generate_llm_explanation(risk_score: float, decision: str, input_df: pd.DataFrame) -> str:
    """Invokes Google Gemini API to produce executive AML compliance summaries."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "⚠️ GEMINI_API_KEY environment variable is not configured."

    amount = float(input_df['amount'].iloc[0]) if 'amount' in input_df.columns else 0.0
    old_orig = float(input_df['oldbalanceOrg'].iloc[0]) if 'oldbalanceOrg' in input_df.columns else 0.0
    new_orig = float(input_df['newbalanceOrig'].iloc[0]) if 'newbalanceOrig' in input_df.columns else 0.0
    old_dest = float(input_df['oldbalanceDest'].iloc[0]) if 'oldbalanceDest' in input_df.columns else 0.0
    new_dest = float(input_df['newbalanceDest'].iloc[0]) if 'newbalanceDest' in input_df.columns else 0.0

    prompt = textwrap.dedent(f"""\
    You are a Senior AML & Financial Crime Analyst reviewing an automated fraud alert.
    Explain the operational reason why this transaction triggered a risk flag.

    TRANSACTION AUDIT TELEMETRY:
    - Amount Transferred: USD {amount:,.2f}
    - Sender Balance (Old -> New): USD {old_orig:,.2f} -> USD {new_orig:,.2f}
    - Receiver Balance (Old -> New): USD {old_dest:,.2f} -> USD {new_dest:,.2f}
    - Risk Score: {risk_score * 100:.2f}%
    - Engine Decision: {decision}

    RULES:
    1. Do not use generic phrases like "high model threshold".
    2. Format output in concise Markdown.
    3. Do NOT use LaTeX math code ($...$).

    FORMAT:
    ### Executive Summary
    (2 direct sentences explaining the money movement anomaly)

    ### Key Findings
    - **Sender Balance Movement:** (Describe account drain)
    - **Destination Impact:** (Describe receiving balance dynamic)
    """)

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=500
            )
        )
        return response.text.strip()
    except Exception as e:
        return f"⚠️ LLM Audit Generation Error: {str(e)}"


if __name__ == "__main__":
    from src.data_pipeline import PaySimDataPipeline

    data_path = os.path.join("data", "transactiondata.csv")
    if not os.path.exists(data_path):
        data_path = os.path.join("data", "paysim.csv")

    pipeline = PaySimDataPipeline(raw_filepath=data_path, sample_size=200000)
    _, X_test, _, y_test = pipeline.prepare_pipeline()

    xai = FraudXAIExplainer()
    fraud_indices = y_test[y_test == 1].index
    
    if len(fraud_indices) > 0:
        sample_fraud = X_test.loc[[fraud_indices[0]]]
        print(f"\n[XAI Interpretability Audit - Transaction Index: {fraud_indices[0]}]")
        contributions = xai.get_feature_contributions(sample_fraud)
        print(contributions.to_string(index=False))