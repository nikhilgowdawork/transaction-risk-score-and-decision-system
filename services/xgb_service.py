import os
from sklearn.preprocessing import StandardScaler
import joblib
import shap
import pandas as pd
import numpy as np
from xgboost import XGBClassifier


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #finding the path in the whole folder

#paths for specific files (like model and scaler)
model_path =os.path.join(BASE_DIR,"models","xgb_model.json")
scaler_path = os.path.join(BASE_DIR,"models","scaler.pkl")

#Loading the model and scaler

model = XGBClassifier()
model.load_model(model_path)
scaler = joblib.load(scaler_path)

#Implementing SHAP to explain that "WHY"
explainer = shap.TreeExplainer(model)


def XGBoostcome(transaction):
    transaction_scaled = scaler.transform(transaction)

    prob = model.predict_proba(transaction_scaled)[:,1]
    
    threshold = 0.6
    
    decision = (prob >= threshold).astype(int)

    #SHAP Explaination
    shap_values = explainer.shap_values(transaction_scaled)

    feature_impact = pd.DataFrame({
        "feature": transaction.columns,
        "impact" : shap_values[0]
    })

    top_features = feature_impact.iloc[
        np.argsort(np.abs(feature_impact["impact"]))[::-1]

    ].head(3)


    return prob ,decision,top_features


