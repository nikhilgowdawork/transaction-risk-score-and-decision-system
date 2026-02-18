# Transaction Risk Scoring & Decision System

## Overview
This project is a machine learning-based transaction risk scoring system.  
It predicts the fraud probability of a credit card transaction and assigns a decision (Allow / Review / Block) based on a configurable risk threshold.

The system simulates real-time risk evaluation using historical transaction data.


## Problem Statement
Financial institutions face significant losses due to fraudulent transactions.  
This system predicts fraud risk using historical data and applies decision logic to support automated transaction control.


## Approach
- Data preprocessing and scaling using StandardScaler
- Imbalance handling using SMOTE
- Model training using XGBoost
- Probability-based risk scoring
- Custom decision threshold logic
- Interactive Streamlit interface


## Features
- Fraud probability score (%)
- Configurable risk threshold
- Decision output (Safe / Fraud)
- Risk meter visualization
- Raw transaction display
- Real-time simulation interface

## model performance
- ROC-AUC: 0.88
- Precision: 0.688
- Recall: 0.98
- F1-Score: 0.77

this performance is under 0.45 threshold 



## Tech Stack
- Python
- XGBoost
- Scikit-learn
- SMOTE (Imbalanced-learn)
- Streamlit
- Joblib
- Pandas / NumPy

---

## How to Run

1. Install dependencies: pip install -r requirements.txt
2. Run the app: streamlit run app.py

## Project Structureapp.py

scaler.pkl
Transaction risk scoring and decision system.pkl
transactiondata.csv
README.md
requirements.txt

## Author
Nikhil Gowda S