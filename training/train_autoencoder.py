import pandas as pd 
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers,models


BASE_DIR =os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path =os.path.join(BASE_DIR,"data","transactiondata.csv")


df = pd.read_csv(data_path)
x = df.drop("Class", axis=1)
y = df["Class"]

# AUTOENCODER is a neural network that learn " what does normal data look like ?" 
# it just learns the pattern of normal transactions.

#Scale 
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

#train only on normal
x_normal = x_scaled[y ==0]
x_fraud =  x_scaled[y == 1]

x_train,x_val = train_test_split(x_normal, test_size = 0.2 , random_state = 42)

input_dim = x_train.shape[1]

#Build AutoEncoder
model = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(16,activation="relu"),
    layers.Dense(8,activation="relu"),
    layers.Dense(16,activation="relu"),
    layers.Dense(input_dim,activation="linear")
])

model.compile(
    optimizer= "adam",
    loss = "mse"
)

#train
history = model.fit(
    x_train,x_train,
    epochs = 20,
    batch_size = 256,
    validation_data=(x_val, x_val),
    shuffle = True

)
x_pred = model.predict(x_scaled)
reconstruction_error = np.mean(np.square(x_scaled - x_pred),axis=1)

df["anomaly_score"] = reconstruction_error

print("Normal mean errror:",
      df[df["Class"] == 0]["anomaly_score"].mean())

print("fraud mean error:",
      df[df["Class"] == 1]["anomaly_score"].mean())