
import os 
import pandas as pd 
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
import joblib
from sklearn.metrics import confusion_matrix,roc_curve,classification_report,precision_recall_curve

# Load the autoencoder model and sacler


#load the data using pandas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# For accessing the dataset from 'data' folder 
data_path=os.path.join(BASE_DIR,"data","transactiondata.csv")
# for accesing the model from 'models' folder
model_path = os.path.join(BASE_DIR,"models","autoencoder_model.keras")
# for accessing the sclaer from modesls folder
scaler_path =os.path.join(BASE_DIR,"models","AEscaler.pkl")

# Load the model and scaler 
autoencoder = load_model(model_path)
scaler = joblib.load(scaler_path)

df = pd.read_csv(data_path)
x = df.drop("Class", axis=1) # contain all 30 columns excluding [Class],
y = df["Class"] # Contains the CLass column Only

# Use scaler that is imported and Use it just transfrom the X 
x_scaled = scaler.transform(x)


x_pred = autoencoder.predict(x_scaled)
reconstruction_error = np.mean(np.square(x_scaled - x_pred),axis=1)
df["anomaly_score"] = reconstruction_error



print("Normal mean errror:",
      df[df["Class"] == 0]["anomaly_score"].mean())

print("fraud mean error:",
      df[df["Class"] == 1]["anomaly_score"].mean())

# creating the threshold by precision_recall_curve , Anything beyond the threshold is anomaly
precision,recall,thresholds = precision_recall_curve(y,reconstruction_error)
f1_score = 2*(precision * recall) / (precision + recall + 1e-8)
threshold = thresholds[np.argmax(f1_score)]
print("threshold:",threshold)


# convert the score to prediction
df["ae_prediction"] = (df["anomaly_score"] > threshold).astype(int)
y_pred = df["ae_prediction"]



print("confusion_matirx:",confusion_matrix(y,y_pred))
print("classification_report:",classification_report(y,y_pred))
