import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
import textwrap
from google import genai
from google.genai import types

from dotenv import load_dotenv
load_dotenv()  # Loads variables from .env into os.environ


def generate_llm_explanation(risk_score, decision, input_df, shap_values, feature_names):
    """
    Generates a clear, professional fraud compliance audit explanation.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            api_key = None

    if not api_key:
        return "⚠️ Google Gemini API Key not found. Please set `GEMINI_API_KEY` in `.env`."

    FEATURE_MAP = {
        'errorBalanceOrig': 'Originating account balance discrepancy',
        'errorBalanceDest': 'Destination account balance discrepancy',
        'newbalanceOrig': 'Sender final account balance',
        'oldbalanceOrg': 'Sender initial account balance',
        'newbalanceDest': 'Receiver final account balance',
        'oldbalanceDest': 'Receiver initial account balance',
        'amount': 'Transaction transfer amount',
        'type_TRANSFER': 'Wire Transfer type indicator',
        'type_CASH_OUT': 'Cash Out withdrawal indicator',
        'type_PAYMENT': 'Payment transaction indicator',
        'type_DEBIT': 'Debit card transaction indicator',
        'step': 'Simulation step hour'
    }

    row = input_df.iloc[0]
    amount = float(row.get('amount', 0))
    old_orig = float(row.get('oldbalanceOrg', 0))
    new_orig = float(row.get('newbalanceOrig', 0))
    old_dest = float(row.get('oldbalanceDest', 0))
    new_dest = float(row.get('newbalanceDest', 0))

    row_shap = shap_values[0].values if hasattr(shap_values[0], 'values') else shap_values[0]

    shap_pairs = []
    for feat, shap_val in zip(feature_names, row_shap):
        val_inp = row[feat] if feat in row else "N/A"
        clean_feat = FEATURE_MAP.get(feat, feat)
        shap_pairs.append((clean_feat, float(shap_val), str(val_inp)))

    risk_drivers = sorted([p for p in shap_pairs if p[1] > 0], key=lambda x: x[1], reverse=True)[:3]
    safe_drivers = sorted([p for p in shap_pairs if p[1] < 0], key=lambda x: x[1])[:3]

    risk_text = "\n".join([f"- {f}: {v}" for f, _, v in risk_drivers]) if risk_drivers else "None"
    safe_text = "\n".join([f"- {f}: {v}" for f, _, v in safe_drivers]) if safe_drivers else "None"

    prompt = textwrap.dedent(f"""\
    System: You are an expert AI Fraud Auditor. Provide a clear compliance audit report for the financial transaction below.

    TRANSACTION DATA:
    - Transfer Amount: ${amount:,.2f}
    - Sender Balance: ${old_orig:,.2f} -> ${new_orig:,.2f}
    - Receiver Balance: ${old_dest:,.2f} -> ${new_dest:,.2f}
    - Fraud Risk Score: {risk_score * 100:.2f}%
    - Action Taken: {decision}

    PRIMARY ANOMALIES:
    {risk_text}

    MITIGATING FACTORS:
    {safe_text}

    REQUIRED FORMAT:
    Produce your response exactly using this structure:

    **Executive Summary:**
    State clearly that the transaction was assigned the outcome "{decision}" because of the specific balance changes observed (e.g. sender balance wiped out to $0.00 while transferring ${amount:,.2f}).

    **Key Audit Findings:**
    - Bullet point 1: Detail the sender account drain/discrepancy.
    - Bullet point 2: Detail the receiver balance status or transfer type anomaly.
    """)

    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=400,
            )
        )
        return response.text.strip()

    except Exception as e:
        return f"⚠️ Could not generate Gemini explanation: {str(e)}"
    
# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Transaction Risk Scoring & Decision Engine",
    page_icon="🛡️",
    layout="wide"
)

# Initialize evaluation flag in session state
if "evaluated" not in st.session_state:
    st.session_state.evaluated = False

# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS & MODEL LOADING
# -----------------------------------------------------------------------------
@st.cache_resource
def load_xgboost_model():
    """Loads the primary trained XGBoost model or falls back to alternatives."""
    model_path = os.path.join("models", "xgb_kmeans_smote.pkl")
    fallback_paths = [
        os.path.join("models", "xgb_smote.pkl"),
        os.path.join("models", "xgb_baseline.pkl")
    ]

    if os.path.exists(model_path):
        return joblib.load(model_path)

    for path in fallback_paths:
        if os.path.exists(path):
            st.sidebar.warning(f"⚠️ Primary model missing. Loaded fallback: {path}")
            return joblib.load(path)

    return None

model = load_xgboost_model()

# -----------------------------------------------------------------------------
# 3. PRE-SET SAMPLES & SIDEBAR INPUT FORM
# -----------------------------------------------------------------------------
PRESET_TRANSACTIONS = {
    "-- Manual Custom Input --": {
        "step": 1, "amount": 0.0, "oldbalanceOrg": 0.0, "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0, "newbalanceDest": 0.0, "is_transfer": "TRANSFER"
    },
    "🚨 Fraud Sample 1: Full Account Drain": {
        "step": 180, "amount": 500000.0, "oldbalanceOrg": 500000.0, "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0, "newbalanceDest": 0.0, "is_transfer": "TRANSFER"
    },
    "🚨 Fraud Sample 2: Mismatched Receiver Balance": {
        "step": 320, "amount": 250000.0, "oldbalanceOrg": 250000.0, "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0, "newbalanceDest": 0.0, "is_transfer": "TRANSFER"
    },
    "✅ Legitimate Sample: Everyday Retail Payment": {
        "step": 45, "amount": 75.50, "oldbalanceOrg": 1200.0, "newbalanceOrig": 1124.50,
        "oldbalanceDest": 5000.0, "newbalanceDest": 5075.50, "is_transfer": "PAYMENT"
    }
}

st.sidebar.header("🕹️ Quick Pre-fill Sample Selector")

def on_preset_change():
    st.session_state.evaluated = False

selected_preset = st.sidebar.selectbox(
    "Choose a pre-set transaction scenario:",
    options=list(PRESET_TRANSACTIONS.keys()),
    index=0,
    on_change=on_preset_change
)

preset_data = PRESET_TRANSACTIONS[selected_preset]

st.sidebar.markdown("---")
st.sidebar.header("Input Transaction Parameters")

# Sidebar Form keeps inputs contained until button is clicked
with st.sidebar.form(key="transaction_form"):
    type_options = ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT"]
    type_index = type_options.index(preset_data["is_transfer"]) if preset_data["is_transfer"] in type_options else 0

    step = st.number_input("Step (Hour)", min_value=1, max_value=744, value=int(preset_data["step"]))
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=float(preset_data["amount"]))
    oldbalanceOrg = st.number_input("Sender Old Balance ($)", min_value=0.0, value=float(preset_data["oldbalanceOrg"]))
    newbalanceOrig = st.number_input("Sender New Balance ($)", min_value=0.0, value=float(preset_data["newbalanceOrig"]))
    oldbalanceDest = st.number_input("Receiver Old Balance ($)", min_value=0.0, value=float(preset_data["oldbalanceDest"]))
    newbalanceDest = st.number_input("Receiver New Balance ($)", min_value=0.0, value=float(preset_data["newbalanceDest"]))
    is_transfer = st.selectbox("Transaction Type", type_options, index=type_index)

    submit_button = st.form_submit_button(label="⚡ Evaluate Transaction", use_container_width=True)

if submit_button:
    st.session_state.evaluated = True

# -----------------------------------------------------------------------------
# 4. FEATURE ENGINEERING LOGIC
# -----------------------------------------------------------------------------
if st.session_state.evaluated:
    type_TRANSFER = 1 if is_transfer == "TRANSFER" else 0
    errorBalanceOrig = (newbalanceOrig + amount) - oldbalanceOrg
    errorBalanceDest = (oldbalanceDest + amount) - newbalanceDest
    isMerchantDest = 0
    zeroBalOrigAfter = 1 if (newbalanceOrig == 0 and oldbalanceOrg > 0) else 0
    zeroBalDestBefore = 1 if (oldbalanceDest == 0 and amount > 0) else 0

    input_data = pd.DataFrame([{
        'step': step,
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest,
        'type_TRANSFER': type_TRANSFER,
        'errorBalanceOrig': errorBalanceOrig,
        'errorBalanceDest': errorBalanceDest,
        'isMerchantDest': isMerchantDest,
        'zeroBalOrigAfter': zeroBalOrigAfter,
        'zeroBalDestBefore': zeroBalDestBefore
    }])
else:
    input_data = None

# -----------------------------------------------------------------------------
# 5. DASHBOARD MAIN BODY & TABS
# -----------------------------------------------------------------------------
st.title("🛡️ Real-Time Transaction Risk Scoring & Explainable AI Engine")
st.markdown("Automated fraud detection using XGBoost, SHAP Explanations, and River streaming metrics.")

tab1, tab2, tab3 = st.tabs([
    "⚡ Single Transaction Risk Evaluator", 
    "🔍 SHAP Explainable AI", 
    "📉 Streaming Performance Metrics"
])

# TAB 1: Single Transaction Risk Evaluator
with tab1:
    st.subheader("Transaction Risk Assessment")
    
    if not st.session_state.evaluated or input_data is None:
        st.info("👈 Please select a preset transaction or fill parameters in the sidebar, then click **'⚡ Evaluate Transaction'**.")
    elif model is None:
        st.error("❌ Model not found! Ensure `xgb_kmeans_smote.pkl` is inside the `models/` directory.")
    else:
        risk_score = float(model.predict_proba(input_data)[0][1])
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Risk Score", f"{risk_score * 100:.2f}%")
        
        if risk_score > 0.75:
            col2.error("Decision: BLOCK TRANSACTION")
            col3.warning("Reason: Critical anomaly detected (high risk of fraud).")
        elif risk_score > 0.35:
            col2.warning("Decision: FLAG FOR REVIEW")
            col3.info("Reason: Moderate balance discrepancy detected.")
        else:
            col2.success("Decision: ALLOW TRANSACTION")
            col3.success("Reason: Standard transaction profile.")

        st.markdown("---")
        st.write("#### Evaluated Feature Vector")
        st.dataframe(input_data, use_container_width=True)

# TAB 2: SHAP Explanations
# TAB 2: SHAP & LLM Natural Language Explanation
with tab2:
    st.subheader("🔍 Explainable AI & Audit Narrative")
    
    if not st.session_state.evaluated or input_data is None:
        st.info("👈 Please evaluate a transaction first to view feature attributions and AI commentary.")
    elif model is None:
        st.error("❌ Model not loaded. Cannot calculate SHAP values.")
    else:
        # Calculate SHAP Values
        explainer = shap.Explainer(model)
        shap_values = explainer(input_data)
        
        # Calculate Risk Score & Decision
        risk_score = float(model.predict_proba(input_data)[0][1])
        if risk_score > 0.75:
            decision = "BLOCK TRANSACTION"
        elif risk_score > 0.35:
            decision = "FLAG FOR REVIEW"
        else:
            decision = "ALLOW TRANSACTION"

        # Divide layout into two columns: LLM Text Narrative on Left, SHAP Plot on Right
        col_text, col_plot = st.columns([1, 1])

        with col_text:
            st.markdown("### 🤖 Compliance AI Audit Note")
            with st.spinner("Generating LLM investigation summary..."):
                llm_summary = generate_llm_explanation(
                    risk_score=risk_score,
                    decision=decision,
                    input_df=input_data,
                    shap_values=shap_values,
                    feature_names=input_data.columns
                )
            st.info(llm_summary)

        with col_plot:
            st.markdown("### 📊 SHAP Feature Attribution")
            try:
                fig, ax = plt.subplots(figsize=(7, 4))
                shap.plots.waterfall(shap_values[0], show=False)
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.error(f"Error rendering SHAP plot: {e}")

# TAB 3: River Streaming Simulation Logs
with tab3:
    st.subheader("River Streaming Checkpoint Performance")
    
    stream_results = pd.DataFrame({
        "Checkpoint Records": [10000, 20000, 30000, 40000],
        "Online Accuracy": [0.9986, 0.9985, 0.9978, 0.9966],
        "Online F1-Score": [0.0000, 0.0000, 0.4037, 0.6895],
        "Online ROC-AUC": [0.5000, 0.5000, 0.6889, 0.8611]
    })
    
    st.dataframe(stream_results, use_container_width=True)
    st.line_chart(stream_results.set_index("Checkpoint Records")[["Online F1-Score", "Online ROC-AUC"]])