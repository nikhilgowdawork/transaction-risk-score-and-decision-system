import streamlit as st
import numpy as np 
import pandas as pd
import joblib

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



#load trained data or scaler 
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

#page configuration
st.set_page_config(
 page_title ="CCFD",
 page_icon="💳",
 layout="wide"
)
#load dataset
df = pd.read_csv("transactiondata.csv")
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
  #scale
   transaction_scaled = scaler.transform(transaction)

  #predict 
   prob= model.predict_proba(transaction_scaled)[0][1]

   threshold = 0.6
   prediction = "FRAUD ❌ " if prob >= threshold else "SAFE ✅" 
  
   st.session_state["transaction"] =transaction
   st.session_state["transaction_scaled"] = transaction_scaled
   st.session_state["prob"]= prob
   st.session_state["prediction"]=prediction
  
   st.session_state.page =2
   st.rerun()
 
 
 #OUTPUT
elif st.session_state.page == 2:  
  
 st.title("Prediction Result")

 col1,col2 = st.columns(2)
 with col1:
  st.metric(
    label="Fraud Probability",
    value= f"{st.session_state["prob"]*100:.2f}%",
 )
 with col2:
  st.metric(
   label="Decision",
   value= f"{st.session_state["prediction"]}"
   )
 st.write(f"**Risk Meter-**")
 st.progress(float(st.session_state["prob"]))

 #show raw data
 st.title("Transaction Details")  
 st.dataframe(st.session_state["transaction"]) 

 if st.button("back"):
  st.session_state.page = 1
  st.rerun()
  
  



