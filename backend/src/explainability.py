import os
import textwrap
import joblib
import pandas as pd
import shap
from google import genai
from google.genai import types
from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.abspath(os.path.join(CURRENT_DIR, "..","..", ".env"))
load_dotenv(dotenv_path=env_path)

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

        shap_values = self.explainer(instance_df)     #SHAP analyses the transaction
        vals = shap_values.values[0]          #get SHAP vaklues for every feature inthe data

        feature_names = instance_df.columns     #get feature names

        contributions = pd.DataFrame({
            'Feature': feature_names,
            'Feature_Value': instance_df.iloc[0].values,
            'SHAP_Contribution': vals

            # create a clean table of feature names , feature values and shap values

        }).sort_values(by='SHAP_Contribution', key=abs, ascending=False).reset_index(drop=True)           #sort by importance and reset index
        
        return contributions


def generate_llm_explanation(
    risk_score: float, 
    decision: str, 
    input_df: pd.DataFrame, 
    shap_df: pd.DataFrame = None
) -> str:
    """Invokes Google Gemini API to produce executive AML compliance summaries using SHAP attributions."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "⚠️ GEMINI_API_KEY environment variable is not configured."

#---------------------------------------------------------------------
# Extract transaction values

    amount = float(input_df['amount'].iloc[0]) if 'amount' in input_df.columns else 0.0
    old_orig = float(input_df['oldbalanceOrg'].iloc[0]) if 'oldbalanceOrg' in input_df.columns else 0.0
    new_orig = float(input_df['newbalanceOrig'].iloc[0]) if 'newbalanceOrig' in input_df.columns else 0.0
    old_dest = float(input_df['oldbalanceDest'].iloc[0]) if 'oldbalanceDest' in input_df.columns else 0.0
    new_dest = float(input_df['newbalanceDest'].iloc[0]) if 'newbalanceDest' in input_df.columns else 0.0

    # Format SHAP explanations into string context for Gemini
    shap_text = ""
    if shap_df is not None and not shap_df.empty:
        top_factors = []
        for _, row in shap_df.head(5).iterrows():
            direction = "INCREASED risk (+fraud)" if row['SHAP_Contribution'] > 0 else "DECREASED risk (-safe)"
            top_factors.append(
                f"- Feature '{row['Feature']}' (Value: {row['Feature_Value']}) {direction} with a SHAP impact of {row['SHAP_Contribution']:+.4f}"
            )
        shap_text = "\n".join(top_factors)
    else:
        shap_text = "No SHAP feature values supplied."

    prompt = textwrap.dedent(f"""\
    You are a Senior AML & Financial Crime Analyst reviewing an automated fraud alert.
    Explain the operational reason why this transaction triggered a risk flag using the provided SHAP model explanations.

    TRANSACTION AUDIT TELEMETRY:
    - Amount Transferred: USD {amount:,.2f}
    - Sender Balance (Old -> New): USD {old_orig:,.2f} -> USD {new_orig:,.2f}
    - Receiver Balance (Old -> New): USD {old_dest:,.2f} -> USD {new_dest:,.2f}
    - Model Risk Score: {risk_score * 100:.2f}%
    - Engine Decision: {decision}

    MATHEMATICAL SHAP FEATURE ATTRIBUTIONS:
    {shap_text}

    RULES:
    1. Base your operational explanation explicitly on the top SHAP feature drivers listed above.
    2. Do not use generic phrases like "high model threshold".
    3. Format output in concise Markdown.
    4. Do NOT use LaTeX math code ($...$).

    FORMAT:
    ### Executive Summary
    (2 direct sentences explaining the money movement anomaly and top SHAP drivers)

    ### Key Findings
    - **Primary Model Driver:** (Describe the highest SHAP contribution feature and its business logic)
    - **Sender/Receiver Dynamics:** (Describe account movement context supported by SHAP inputs)
    """)

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-3.6-flash",
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
    from data_pipeline import PaySimDataPipeline

    data_path = os.path.join("backend/data", "transactiondata.csv")
    if not os.path.exists(data_path):
        data_path = os.path.join("backend/data", "paysim.csv")

    pipeline = PaySimDataPipeline(raw_filepath=data_path, sample_size=200000)
    _, X_test, _, y_test = pipeline.prepare_pipeline()

    xai = FraudXAIExplainer()
    fraud_indices = y_test[y_test == 1].index
    
    if len(fraud_indices) > 0:
        sample_fraud = X_test.loc[[fraud_indices[0]]]
        
        # 1. Compute SHAP Values
        contributions = xai.get_feature_contributions(sample_fraud)
        print(f"\n[XAI Interpretability Audit - Transaction Index: {fraud_indices[0]}]")
        print(contributions.to_string(index=False))

        # 2. Get Model Prediction Score & Decision
        risk_score = float(xai.model.predict_proba(sample_fraud)[0][1])
        decision = "BLOCK TRANSACTION" if risk_score > 0.75 else "FLAG FOR REVIEW"

        # 3. Pass SHAP Data into Gemini to Generate Natural Language Audit
        audit_report = generate_llm_explanation(
            risk_score=risk_score,
            decision=decision,
            input_df=sample_fraud,
            shap_df=contributions
        )
        print("\n--- GEMINI AUDIT REPORT ---")
        print(audit_report)