
import os
import joblib
from tensorflow.keras.models import load_model # type: ignore
import numpy as np


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# for accesing the model from 'models' folder
model_path = os.path.join(BASE_DIR,"models","autoencoder_model.keras")

# for accessing the sclaer from modesls folder
scaler_path =os.path.join(BASE_DIR,"models","AEscaler.pkl")

#for accessing the threshold from models folder
threshold_path = os.path.join(BASE_DIR,"models","AE_threshold.pkl")

# Load the model and scaler 
autoencoder = load_model(model_path)
scaler = joblib.load(scaler_path)
threshold = joblib.load(threshold_path)


def AutoencoderCome(transaction):
    transaction_scaled = scaler.transform(transaction)

    reconstruction = autoencoder.predict(transaction_scaled)

    anomaly_score = np.mean(np.square(transaction_scaled - reconstruction),axis=1)

    decision = (anomaly_score > threshold).astype(int)

    error = np.minimum(anomaly_score/ threshold ,1)

    return error , decision

    

