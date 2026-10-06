import os
import textwrap
import time
import joblib
import pandas as pd
import shap

from google import genai
from google.genai import types
from dotenv import load_dotenv


# ================================================================
# PATH CONFIGURATION
# ================================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# backend/src
BACKEND_DIR = os.path.abspath(
    os.path.join(CURRENT_DIR, "..")
)

# project root
PROJECT_DIR = os.path.abspath(
    os.path.join(BACKEND_DIR, "..")
)

# .env is in project root
env_path = os.path.join(
    PROJECT_DIR,
    ".env"
)

load_dotenv(dotenv_path=env_path)


# ================================================================
# FRAUD XAI EXPLAINER
# ================================================================

class FraudXAIExplainer:
    """
    Explainable AI (XAI) engine using SHAP
    to generate feature contributions for
    XGBoost model predictions.
    """

    def __init__(
        self,
        model_path: str = None
    ):
        # --------------------------------------------------------
        # If model path is not provided, use backend/models
        # --------------------------------------------------------

        if model_path is None:
            model_path = os.path.join(
                BACKEND_DIR,
                "models",
                "xgb_kmeans_smote.pkl"
            )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model checkpoint not found at {model_path}."
            )

        print(
            f"Loading model for XAI from {model_path}..."
        )

        self.model = joblib.load(model_path)

        # SHAP TreeExplainer works with tree-based models
        
        self.explainer = shap.TreeExplainer(
            self.model
        )

    def explain_instance(
        self,
        instance_df: pd.DataFrame
    ):
        """
        Generates the raw SHAP Explanation object
        for a single transaction.
        """

        return self.explainer(instance_df)

    def get_feature_contributions(
        self,
        instance_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Converts SHAP values into a readable DataFrame.
        """

        # SHAP analyzes the transaction
        shap_values = self.explainer(
            instance_df
        )

        # Get SHAP values for the first transaction
        vals = shap_values.values[0]

        # Get feature names
        feature_names = instance_df.columns

        # Create readable table
        contributions = pd.DataFrame({
            "Feature": feature_names,
            "Feature_Value": instance_df.iloc[0].values,
            "SHAP_Contribution": vals
        })

        # Sort by absolute SHAP impact
        contributions = (
            contributions
            .sort_values(
                by="SHAP_Contribution",
                key=abs,
                ascending=False
            )
            .reset_index(drop=True)
        )

        return contributions


# ================================================================
# GEMINI EXPLANATION
# ================================================================

    def generate_llm_explanation(self,
        risk_score: float,
        decision: str,
        input_df: pd.DataFrame,
        shap_df: pd.DataFrame = None
    ) -> str:

        """
        Uses Gemini to convert mathematical SHAP
        explanations into a human-readable fraud audit.
        """

        api_key = os.getenv(
            "GEMINI_API_KEY"
        )

        if not api_key:
            return (
                "⚠️ GEMINI_API_KEY environment variable "
                "is not configured."
            )

        # ------------------------------------------------------------
        # Extract transaction values
        # ------------------------------------------------------------

        amount = (
            float(input_df["amount"].iloc[0])
            if "amount" in input_df.columns
            else 0.0
        )

        old_orig = (
            float(input_df["oldbalanceOrg"].iloc[0])
            if "oldbalanceOrg" in input_df.columns
            else 0.0
        )

        new_orig = (
            float(input_df["newbalanceOrig"].iloc[0])
            if "newbalanceOrig" in input_df.columns
            else 0.0
        )

        old_dest = (
            float(input_df["oldbalanceDest"].iloc[0])
            if "oldbalanceDest" in input_df.columns
            else 0.0
        )

        new_dest = (
            float(input_df["newbalanceDest"].iloc[0])
            if "newbalanceDest" in input_df.columns
            else 0.0
        )

        # ------------------------------------------------------------
        # Convert SHAP contributions into text
        # ------------------------------------------------------------

        if shap_df is not None and not shap_df.empty:

            top_factors = []

            for _, row in shap_df.head(5).iterrows():

                if row["SHAP_Contribution"] > 0:
                    direction = (
                        "INCREASED risk (+fraud)"
                    )
                else:
                    direction = (
                        "DECREASED risk (-safe)"
                    )

                top_factors.append(
                    f"- Feature '{row['Feature']}' "
                    f"(Value: {row['Feature_Value']}) "
                    f"{direction} "
                    f"with a SHAP impact of "
                    f"{row['SHAP_Contribution']:+.4f}"
                )

            shap_text = "\n".join(
                top_factors
            )

        else:

            shap_text = (
                "No SHAP feature values supplied."
            )

        # ------------------------------------------------------------
        # Gemini prompt
        # ------------------------------------------------------------

        prompt = textwrap.dedent(f"""
            You are a Senior AML & Financial Crime Analyst
            reviewing an automated fraud alert.

            Explain the operational reason why this transaction
            triggered a risk flag using the provided SHAP
            model explanations.

            TRANSACTION AUDIT TELEMETRY:

            - Amount Transferred: USD {amount:,.2f}
            - Sender Balance (Old -> New):
            USD {old_orig:,.2f} -> USD {new_orig:,.2f}

            - Receiver Balance (Old -> New):
            USD {old_dest:,.2f} -> USD {new_dest:,.2f}

            - Model Risk Score:
            {risk_score * 100:.2f}%

            - Engine Decision:
            {decision}

            MATHEMATICAL SHAP FEATURE ATTRIBUTIONS:

            {shap_text}

            RULES:

            1. Base the explanation explicitly on the
            top SHAP feature drivers.

            2. Do not use generic phrases such as
            "high model threshold".

            3. Explain why the transaction appears risky
            based on the transaction values and SHAP results.

            4. Format the output in concise Markdown.

            5. Do NOT use LaTeX math code.

            FORMAT:

            ### Executive Summary

            (2 direct sentences explaining the transaction
            anomaly and top SHAP drivers)

            ### Key Findings

            - **Primary Model Driver:**
            Describe the highest SHAP contribution feature
            and its business meaning.

            - **Sender/Receiver Dynamics:**
            Describe the account movement context.

        """)

        # ------------------------------------------------------------
        # Gemini API call
        # ------------------------------------------------------------

        try:

            client = genai.Client(
                api_key=api_key
            )

            for attempt in range(3):
                try:
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.1,
                            max_output_tokens=500
                        )
                    )
                    break
                except Exception as error:
                    is_service_unavailable = (
                        getattr(error, "code", None) == 503
                        or getattr(error, "status", None) == 503
                        or "503 UNAVAILABLE" in str(error)
                    )
                    if not is_service_unavailable or attempt == 2:
                        raise
                    time.sleep(2 ** attempt)

            audit_text = response.text
            if not audit_text or not audit_text.strip():
                finish_reasons = [
                    str(getattr(candidate, "finish_reason", "unknown"))
                    for candidate in (getattr(response, "candidates", None) or [])
                ]
                reason = (
                    f" (finish reason: {', '.join(finish_reasons)})"
                    if finish_reasons
                    else ""
                )
                return (
                    "⚠️ LLM Audit Generation Error: Gemini returned no audit text"
                    f"{reason}."
                )

            return audit_text.strip()

        except Exception as e:

            return (
                f"⚠️ LLM Audit Generation Error: {str(e)}"
            )


# ================================================================
# MAIN TEST
# ================================================================

if __name__ == "__main__":

    # ------------------------------------------------------------
    # Import pipeline
    # ------------------------------------------------------------

    from data_pipeline import PaySimDataPipeline

    # ------------------------------------------------------------
    # Original dataset path
    # ------------------------------------------------------------

    data_path = os.path.join(
        BACKEND_DIR,
        "data",
        "transactiondata.csv"
    )

    if not os.path.exists(data_path):

        data_path = os.path.join(
            BACKEND_DIR,
            "data",
            "paysim.csv"
        )

    # ------------------------------------------------------------
    # Create pipeline object
    # ------------------------------------------------------------

    pipeline = PaySimDataPipeline(
        raw_filepath=data_path,
        sample_size=200000
    )

    # ------------------------------------------------------------
    # IMPORTANT:
    #
    # Load already processed train/test files.
    #
    # This does NOT load the 6.3M-row original dataset.
    # ------------------------------------------------------------

    train_df, test_df = (
        pipeline.load_processed_data()
    )

    # ------------------------------------------------------------
    # Separate X and y
    # ------------------------------------------------------------

    X_test = test_df.drop(
        columns=["isFraud"]
    )

    y_test = test_df["isFraud"]

    # ------------------------------------------------------------
    # Load XAI model
    # ------------------------------------------------------------

    xai = FraudXAIExplainer()

    # ------------------------------------------------------------
    # Find fraud transactions
    # ------------------------------------------------------------

    fraud_indices = y_test[
        y_test == 1
    ].index

    if len(fraud_indices) == 0:

        print(
            "No fraud transactions found in test dataset."
        )

    else:

        # --------------------------------------------------------
        # Select first fraud transaction
        # --------------------------------------------------------

        fraud_index = fraud_indices[0]

        sample_fraud = X_test.loc[
            [fraud_index]
        ]

        # --------------------------------------------------------
        # 1. Compute SHAP values
        # --------------------------------------------------------

        contributions = (
            xai.get_feature_contributions(
                sample_fraud
            )
        )

        print(
            f"\n[XAI Interpretability Audit "
            f"- Transaction Index: {fraud_index}]"
        )

        print(
            contributions.to_string(
                index=False
            )
        )

        # --------------------------------------------------------
        # 2. Model prediction
        # --------------------------------------------------------

        risk_score = float(
            xai.model.predict_proba(
                sample_fraud
            )[0][1]
        )

        # --------------------------------------------------------
        # 3. Decision
        # --------------------------------------------------------

        if risk_score > 0.75:

            decision = (
                "BLOCK TRANSACTION"
            )

        else:

            decision = (
                "FLAG FOR REVIEW"
            )

        print(
            f"\nRisk Score: "
            f"{risk_score * 100:.2f}%"
        )

        print(
            f"Decision: {decision}"
        )

        # --------------------------------------------------------
        # 4. Gemini explanation
        # --------------------------------------------------------

        audit_report = (xai.generate_llm_explanation(
                risk_score=risk_score,
                decision=decision,
                input_df=sample_fraud,
                shap_df=contributions
            )
        )

        print(
            "\n--- GEMINI AUDIT REPORT ---"
        )

        print(audit_report)