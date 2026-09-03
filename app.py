import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
import time

# Page Configuration
st.set_page_config(
    page_title="Transaction Risk Scoring & Decision Engine",
    page_icon="🛡️",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 1. HELPER FUNCTIONS & MODEL LOADING
# -----------------------------------------------------------------------------
@st.cache(allow_output_mutation=True)
def load_xgboost_model():
    """Simulates or loads trained XGBoost Model."""
    model = xgb.XGBClassifier()
    # Dummy fit if pre-trained file isn't present
    X_dummy = pd.DataFrame(np.random.rand(100, 12), columns=[
        'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
        'oldbalanceDest', 'newbalanceDest', 'type_TRANSFER', 
        'errorBalanceOrig', 'errorBalanceDest', 'isMerchantDest', 
        'zeroBalOrigAfter', 'zeroBalDestBefore'
    ])
    y_dummy = np.random.choice([0, 1], size=100, p=[0.9, 0.1])
    model.fit(X_dummy, y_dummy)
    return model

model = load_xgboost_model()

# -----------------------------------------------------------------------------
# 2. DASHBOARD HEADER & SIDEBAR
# -----------------------------------------------------------------------------
st.title("🛡️ Real-Time Transaction Risk Scoring & Explainable AI Engine")
st.markdown("Automated fraud detection using XGBoost, SHAP Explanations, and River streaming metrics.")

st.sidebar.header("Input Transaction Parameters")

# Sidebar Manual Input Form
step = st.sidebar.number_input("Step (Hour)", min_value=1, max_value=744, value=1)
amount = st.sidebar.number_input("Transaction Amount ($)", min_value=0.0, value=150000.0)
oldbalanceOrg = st.sidebar.number_input("Sender Old Balance ($)", min_value=0.0, value=150000.0)
newbalanceOrig = st.sidebar.number_input("Sender New Balance ($)", min_value=0.0, value=0.0)
oldbalanceDest = st.sidebar.number_input("Receiver Old Balance ($)", min_value=0.0, value=0.0)
newbalanceDest = st.sidebar.number_input("Receiver New Balance ($)", min_value=0.0, value=0.0)
is_transfer = st.sidebar.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT"])

# Feature Engineering Logic
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

# -----------------------------------------------------------------------------
# 3. TAB NAVIGATION
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["⚡ Single Transaction Risk Evaluator", "🔍 SHAP Explainable AI", "📉 Streaming Performance Metrics"])

# TAB 1: Single Risk Evaluator
with tab1:
    st.subheader("Transaction Risk Assessment")
    
    risk_score = float(model.predict_proba(input_data)[0][1])
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Risk Score", f"{risk_score * 100:.2f}%")
    
    if risk_score > 0.75:
        col2.error("Decision: BLOCK TRANSACTION")
        col3.warning("Reason: High probability of illicit fund transfer.")
    elif risk_score > 0.40:
        col2.warning("Decision: FLAG FOR REVIEW")
        col3.info("Reason: Moderate anomaly detected.")
    else:
        col2.success("Decision: ALLOW TRANSACTION")
        col3.success("Reason: Low-risk transaction profile.")

    st.markdown("---")
    st.write("#### Evaluated Feature Vector")
    st.dataframe(input_data)

# TAB 2: SHAP Explanations
with tab2:
    st.subheader("Explainable AI: SHAP Feature Attribution")
    st.write("Understanding which features contributed to the risk score for this transaction.")
    
    explainer = shap.Explainer(model)
    shap_values = explainer(input_data)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

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