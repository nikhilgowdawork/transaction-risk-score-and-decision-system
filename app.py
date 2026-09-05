import os
import joblib
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import xgboost as xgb
import shap
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & INJECTED CUSTOM CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Transaction Risk Scoring Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global Design System & Fixes for Overflow / Font Inconsistency
st.markdown("""
    <style>
    /* Global Typography Fixes */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }

    /* Prevent text overflow in markdown & audit containers */
    div[data-testid="stMarkdownContainer"] p, 
    div[data-testid="stMarkdownContainer"] li,
    .audit-card {
        word-break: break-word !important;
        overflow-wrap: anywhere !important;
        white-space: normal !important;
        font-size: 0.95rem !important;
        line-height: 1.6 !important;
    }

    /* Disable Streamlit KaTeX Math Styling Overrides */
    .katex, .katex-display {
        font-family: inherit !important;
        font-size: 100% !important;
    }

    /* Custom Risk Metric Badge Styles */
    .badge-block {
        background-color: #EF4444;
        color: white;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-review {
        background-color: #F59E0B;
        color: white;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-allow {
        background-color: #10B981;
        color: white;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    
    /* Container Box Styling */
    .audit-box {
        background-color: #0F172A;
        border: 1px solid #334155;
        border-left: 5px solid #3B82F6;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 15px;
        color: #F8FAFC;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize evaluation session state
if "evaluated" not in st.session_state:
    st.session_state.evaluated = False

# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS & MODEL LOADING
# -----------------------------------------------------------------------------
@st.cache_resource
def load_xgboost_model():
    """Loads trained XGBoost model from disk with fallback handling."""
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

def generate_llm_explanation(risk_score, decision, input_df, shap_values, feature_names):
    """Generates a clear, professional fraud compliance audit explanation."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            api_key = None

    if not api_key:
        return "⚠️ Google Gemini API Key not found. Please verify GEMINI_API_KEY configuration."

    row = input_df.iloc[0]
    amount = float(row.get('amount', 0))
    old_orig = float(row.get('oldbalanceOrg', 0))
    new_orig = float(row.get('newbalanceOrig', 0))
    old_dest = float(row.get('oldbalanceDest', 0))
    new_dest = float(row.get('newbalanceDest', 0))

    prompt = textwrap.dedent(f"""\
    You are a Senior AML & Financial Crime Analyst reviewing an automated fraud alert.
    Explain the EXACT operational reason why this specific transaction triggered a high-risk flag.

    TRANSACTION AUDIT EVIDENCE:
    - Transfer Amount: USD {amount:,.2f}
    - Sender Balance Before: USD {old_orig:,.2f}
    - Sender Balance After: USD {new_orig:,.2f}
    - Receiver Balance Before: USD {old_dest:,.2f}
    - Receiver Balance After: USD {new_dest:,.2f}
    - Fraud Engine Action: {decision}

    FORMAT AND STYLE RULES:
    1. DO NOT state generic phrases like "it was blocked due to a high risk score" or "due to model threshold".
    2. NEVER output LaTeX notation (do NOT use dollar signs around math formulas or brackets like $...$).
    3. Detail the balance mechanics clearly:
       - **Sender Account Wipeout:** Was the sender's balance drained from USD {old_orig:,.2f} to USD {new_orig:,.2f}?
       - **Uncredited Destination Anomaly:** Did the receiver's balance fail to reflect the transfer (USD {old_dest:,.2f} remaining USD {new_dest:,.2f} after a transfer of USD {amount:,.2f})?

    FORMAT:
    Executive Summary:
    (2 short direct sentences explaining the physical movement of money and why the transfer pattern is fraudulent)

    ### Key Audit Findings
    - **Sender Account Drain:** ( Description the dollar wipeout)
    - **Destination Discrepancy:** ( Describption the uncredited destination balance anomaly)
    """)

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=800,
            )
        )
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Could not generate Gemini explanation: {str(e)}"

# -----------------------------------------------------------------------------
# 3. PRESETS & SIDEBAR INPUTS
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
    "🚨 Fraud Sample 3: High-Value Wire Takeover": {
        "step": 410, "amount": 850000.0, "oldbalanceOrg": 900000.0, "newbalanceOrig": 50000.0,
        "oldbalanceDest": 0.0, "newbalanceDest": 0.0, "is_transfer": "TRANSFER"
    },
    "🚨 Fraud Sample 4: Immediate Cash Out Liquidation": {
        "step": 512, "amount": 175000.0, "oldbalanceOrg": 175000.0, "newbalanceOrig": 0.0,
        "oldbalanceDest": 10000.0, "newbalanceDest": 10000.0, "is_transfer": "CASH_OUT"
    },
    "🚨 Fraud Sample 5: Empty Destination Uncredited Drain": {
        "step": 605, "amount": 320000.0, "oldbalanceOrg": 320000.0, "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0, "newbalanceDest": 0.0, "is_transfer": "TRANSFER"
    },
    "🚨 Fraud Sample 6: Large Scale Mule Withdrawal": {
        "step": 710, "amount": 600000.0, "oldbalanceOrg": 600000.0, "newbalanceOrig": 0.0,
        "oldbalanceDest": 50000.0, "newbalanceDest": 50000.0, "is_transfer": "CASH_OUT"
    },
    "✅ Legitimate Sample 1: Everyday Retail Payment": {
        "step": 45, "amount": 75.50, "oldbalanceOrg": 1200.0, "newbalanceOrig": 1124.50,
        "oldbalanceDest": 5000.0, "newbalanceDest": 5075.50, "is_transfer": "PAYMENT"
    },
    "✅ Legitimate Sample 2: Reconciled Wire Transfer": {
        "step": 112, "amount": 5000.0, "oldbalanceOrg": 18500.0, "newbalanceOrig": 13500.0,
        "oldbalanceDest": 1200.0, "newbalanceDest": 6200.0, "is_transfer": "TRANSFER"
    },
    "✅ Legitimate Sample 3: Merchant Debit Transaction": {
        "step": 215, "amount": 420.0, "oldbalanceOrg": 2500.0, "newbalanceOrig": 2080.0,
        "oldbalanceDest": 15000.0, "newbalanceDest": 15420.0, "is_transfer": "DEBIT"
    },
    "✅ Legitimate Sample 4: Small Account Transfer": {
        "step": 300, "amount": 1500.0, "oldbalanceOrg": 8200.0, "newbalanceOrig": 6700.0,
        "oldbalanceDest": 3400.0, "newbalanceDest": 4900.0, "is_transfer": "TRANSFER"
    },
    "✅ Legitimate Sample 5: Routine Bill Payment": {
        "step": 410, "amount": 180.25, "oldbalanceOrg": 3100.0, "newbalanceOrig": 2919.75,
        "oldbalanceDest": 0.0, "newbalanceDest": 0.0, "is_transfer": "PAYMENT"
    },
    "✅ Legitimate Sample 6: Verified Cash Out Settlement": {
        "step": 520, "amount": 800.0, "oldbalanceOrg": 4500.0, "newbalanceOrig": 3700.0,
        "oldbalanceDest": 12000.0, "newbalanceDest": 12800.0, "is_transfer": "CASH_OUT"
    }
}

st.sidebar.title("🛡️ Control Center")
st.sidebar.subheader("🕹️ Quick Preset Selector")

def on_preset_change():
    st.session_state.evaluated = False

selected_preset = st.sidebar.selectbox(
    "Scenario Presets:",
    options=list(PRESET_TRANSACTIONS.keys()),
    index=0,
    on_change=on_preset_change
)

preset_data = PRESET_TRANSACTIONS[selected_preset]

st.sidebar.markdown("---")
st.sidebar.subheader("Transaction Features")

with st.sidebar.form(key="transaction_form"):
    type_options = ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT"]
    type_index = type_options.index(preset_data["is_transfer"]) if preset_data["is_transfer"] in type_options else 0

    step = st.number_input("Simulation Hour (Step)", min_value=1, max_value=744, value=int(preset_data["step"]))
    amount = st.number_input("Amount ($)", min_value=0.0, value=float(preset_data["amount"]))
    oldbalanceOrg = st.number_input("Sender Initial Balance ($)", min_value=0.0, value=float(preset_data["oldbalanceOrg"]))
    newbalanceOrig = st.number_input("Sender Final Balance ($)", min_value=0.0, value=float(preset_data["newbalanceOrig"]))
    oldbalanceDest = st.number_input("Receiver Initial Balance ($)", min_value=0.0, value=float(preset_data["oldbalanceDest"]))
    newbalanceDest = st.number_input("Receiver Final Balance ($)", min_value=0.0, value=float(preset_data["newbalanceDest"]))
    is_transfer = st.selectbox("Transaction Type", type_options, index=type_index)

    submit_button = st.form_submit_button(label="⚡ Run Risk Assessment", use_container_width=True)

if submit_button:
    st.session_state.evaluated = True

# -----------------------------------------------------------------------------
# 4. FEATURE ENGINEERING
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
# 5. MAIN DASHBOARD BODY
# -----------------------------------------------------------------------------
st.title("🛡️ Enterprise Financial Fraud & Risk Decision Platform")
st.markdown("Real-time transaction risk evaluation powered by XGBoost, SHAP Explanations, and Gemini Compliance Intelligence.")

tab1, tab2, tab3 = st.tabs([
    "⚡ Risk Assessment Engine", 
    "🔍 SHAP & Compliance Audit Note", 
    "📈 Streaming Analytics"
])

# -----------------------------------------------------------------------------
# TAB 1: RISK ASSESSMENT
# -----------------------------------------------------------------------------
with tab1:
    if not st.session_state.evaluated or input_data is None:
        st.info("👈 Select a pre-set scenario or adjust transaction parameters in the sidebar, then click **'⚡ Run Risk Assessment'**.")
    elif model is None:
        st.error("❌ Fraud Detection Model not found! Ensure `models/xgb_kmeans_smote.pkl` is in place.")
    else:
        risk_score = float(model.predict_proba(input_data)[0][1])
        
        # Header Metrics Cards
        m1, m2, m3 = st.columns(3)
        
        with m1:
            st.metric("Probability of Fraud", f"{risk_score * 100:.2f}%")
            st.progress(risk_score)
            
        with m2:
            st.write("**Automated Decision**")
            if risk_score > 0.75:
                st.markdown('<div class="badge-block">⛔ BLOCK TRANSACTION</div>', unsafe_allow_html=True)
            elif risk_score > 0.35:
                st.markdown('<div class="badge-review">⚠️ FLAG FOR MANUAL REVIEW</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="badge-allow">✅ ALLOW TRANSACTION</div>', unsafe_allow_html=True)
                
        with m3:
            st.write("**Primary Risk Driver**")
            if risk_score > 0.75:
                st.error("Severe balance discrepancy & sender depletion")
            elif risk_score > 0.35:
                st.warning("Moderate origin/destination balance anomaly")
            else:
                st.success("Balanced account settlement metrics")

        st.markdown("---")
        st.subheader("Evaluated Transaction Feature Vector")
        st.dataframe(input_data.style.format("{:,.2f}"), use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 2: SHAP & LLM AUDIT SUMMARY
# -----------------------------------------------------------------------------
with tab2:
    if not st.session_state.evaluated or input_data is None:
        st.info("👈 Please evaluate a transaction first to view feature attributions and AI commentary.")
    elif model is None:
        st.error("❌ Model not loaded.")
    else:
        explainer = shap.Explainer(model)
        shap_values = explainer(input_data)
        
        risk_score = float(model.predict_proba(input_data)[0][1])
        if risk_score > 0.75:
            decision = "BLOCK TRANSACTION"
        elif risk_score > 0.35:
            decision = "FLAG FOR REVIEW"
        else:
            decision = "ALLOW TRANSACTION"

        # Split into two equal-width columns
        col_text, col_plot = st.columns([1, 1], gap="medium")

        with col_text:
            st.markdown("### 🤖 Compliance AI Audit Narrative")
            with st.spinner("Analyzing transaction telemetry..."):
                llm_summary = generate_llm_explanation(
                    risk_score=risk_score,
                    decision=decision,
                    input_df=input_data,
                    shap_values=shap_values,
                    feature_names=input_data.columns
                )
            
            # Custom container with explicit CSS boundaries to prevent overflow
            st.markdown(
                f"""
                <div class="audit-box">
                    {llm_summary}
                </div>
                """, 
                unsafe_allow_html=True
            )

        with col_plot:
            st.markdown("### 📊 SHAP Feature Attribution Waterfall")
            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                shap.plots.waterfall(shap_values[0], show=False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.error(f"Error rendering SHAP plot: {e}")

# -----------------------------------------------------------------------------
# TAB 3: STREAMING METRICS
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("River Online Streaming Checkpoint Metrics")
    
    stream_results = pd.DataFrame({
        "Checkpoint Records": [10000, 20000, 30000, 40000],
        "Online Accuracy": [0.9986, 0.9985, 0.9978, 0.9966],
        "Online F1-Score": [0.0000, 0.0000, 0.4037, 0.6895],
        "Online ROC-AUC": [0.5000, 0.5000, 0.6889, 0.8611]
    })
    
    st.dataframe(stream_results, use_container_width=True)
    st.line_chart(stream_results.set_index("Checkpoint Records")[["Online F1-Score", "Online ROC-AUC"]])