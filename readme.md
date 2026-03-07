# Transaction Risk Scoring & Decision System

## Overview
This project is a machine learning and deep learning based transaction risk scoring system.
It predicts the fraud probability of a credit card transaction and assigns a final decision (Safe / Fraud) using configurable risk thresholds.

The system simulates real-time transaction risk evaluation using historical transaction data and an interactive Streamlit interface.

## Problem Statement
Financial institutions lose billions annually due to fraudulent transactions.
Detecting fraud quickly and accurately is critical for preventing financial loss and protecting customers.

This system analyzes transaction patterns using machine learning and deep learning models to estimate fraud probability and assist automated transaction decision systems.

## Approach
The pipeline follows these steps
### 1.data Preprocessing: ###
- Data cleaning
- Feature Scaling Using StandardScaler
### 2.Imbalance Handling: ###
- Fraud datasets are highly imbalanced
-Handled using SMOTE
 ### 3.Model Training ###
 - XGBoost Classifier for fraud prediction
 - Autoencoder (Deep Learning) for anomaly detection
### 4.Risk Scoring: ###
- Fraud probability generated from the model
- Converted into a percentage risk score
### 5.Decision Logic ###
- Threshold-based classification
- Final output: Safe ✅ or Fraud ❌

### 6.User Interface ###
- Built using Streamlit
- Simulates real-time fraud analysis

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
- tensorflow
- data set ( transaction.csv) - link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data?select=creditcard.csv 

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
Nikhil gowda S





