import streamlit as st
import numpy as np 
import pandas as pd
import joblib
import os

# initializing the tokens 
if "page" not in st.session_state :
 st.session_state.page = 1

if "transaction" not in st.session_state:
 st.session_state["transaction"] = None

if "transaction_scaled" not in st.session_state:
 st.session_state["transaction_scaled"] = None

if "prob" not in st.session_state:
 st.session_state["prob"]= 0.0

if "prediction" not in st.session_state:
 st.session_state["prediction"]=""

if "error" not in st.session_state:
 st.session_state["error"] = 0.0

if "risk_score" not in st.session_state:
 st.session_state["risk_score"] = 0.0



#LOAD THE FUNCTIONS TO GENERATE OUTPUT
from services.xgb_service import XGBoostcome
from services.autoencoder_service import AutoencoderCome
from services.decision_service import get_final_decision

#page configuration
st.set_page_config(
 page_title ="CCFD",
 page_icon="💳",
 layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data_path=os.path.join(BASE_DIR,"data","transactiondata.csv")

#load dataset
df = pd.read_csv(data_path)
#remove label
X = df.drop("Class",axis=1) 

# UI for page one - takes input
if st.session_state.page == 1:
 st.title("TRANSACTION ANALYSIS AND DECISION 💳") 
 st.info("Model: XGBoost | Imbalance handled using SMOTE | "
    "Custom decision threshold enabled | Real-time risk simulation")

 #takes the transaction index as input 
 if "index" not in st.session_state:
  st.session_state.index =0

 #select transaction
 st.session_state.index = st.number_input(
    "Select Transaction Index",
    min_value=0,
    max_value =len(df)-1,
    step=1,
    value=st.session_state.index
 )
 

 if st.button("Analyse",type="primary"):
  
   transaction= X.iloc[st.session_state.index:st.session_state.index+1]
   # XGBOOST FUNCTION CALL
   prob, xgb_decision = XGBoostcome(transaction)  
   # AUTO ENCODER FUNCTION CALL
   error, AE_decision = AutoencoderCome(transaction)

   # COMBINE SCORES
   risk_score = (0.7 * prob) + (0.3 * error)

   final_prediction = get_final_decision(risk_score)

   st.session_state["transaction"] =transaction
   st.session_state["prob"]= prob
   st.session_state["final_prediction"]=final_prediction
   st.session_state["risk_score"] = risk_score
   st.session_state["error"] =  error
  
   st.session_state.page =2
   st.rerun()
 
 
 #OUTPUT
elif st.session_state.page == 2:  
  
 st.title("Prediction Result")

 col1,col2,col3 = st.columns(3)
 with col1:
  st.metric(
    label="Fraud Probability (XGboost)",
    value= f"{float(st.session_state['prob'])*100:.2f}%"
 )
 with col2:
  st.metric(
   label="Anomaly Score (Auto encoder)",
   value= f"{float(st.session_state['error']):.4f}"
   )
 with col3:
  st.metric(
   label="Final Risk Score",
   value=f"{float(st.session_state['risk_score']):.2f}"
  )

 st.subheader(f"Final Decision: {st.session_state['final_prediction']}")

 st.write(f"**Risk Meter-**")
 st.progress(float(st.session_state["risk_score"]))

 #show raw data
 st.title("Transaction Details")  
 st.dataframe(st.session_state["transaction"]) 

 if st.button("back"):
  st.session_state.page = 1
  st.rerun()
  
  



