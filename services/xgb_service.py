import os
from sklearn.preprocessing import StandardScaler
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #finding the path in the whole folder

#paths for specific files (like model and scaler)
model_path =os.path.join(BASE_DIR,"models","xgb_model.pkl")
scaler_path = os.path.join(BASE_DIR,"models","scaler.pkl")

#Loading the model and scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)


def XGBoostcome(transaction):
    transaction_scaled = scaler.transform(transaction)

    prob = model.predict_proba(transaction_scaled)[:,1]

    threshold = 0.6
    decision = (prob >= threshold).astype(int)

    return prob ,decision


