import pandas as pd 
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers,models # type: ignore
from sklearn.metrics import classification_report,confusion_matrix,roc_curve
import joblib
import random


np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)


BASE_DIR =os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path =os.path.join(BASE_DIR,"data","transactiondata.csv")


df = pd.read_csv(data_path)
# Seperate the data 
x = df.drop("Class", axis=1) # contain all 30 columns excluding [Class],
y = df["Class"] # Contains the CLass column Only

# AUTOENCODER is a neural network that learn " what does normal data look like ?" 
# it just learns the pattern of normal transactions.

# Scale 
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x) # All data (numberings) are converted to nearest of 0. eg: -0.1234 and 0.1234 [ All columns excluding Class]

# Train only on normal
x_normal = x_scaled[y ==0] # Contains all 30 scaled features,where rows's Class == 0 , all normal transactions
x_fraud =  x_scaled[y == 1] # COntains all 30 scaled scaled features ,where rows's  Class == 0 , all fraud transactions 

x_train,x_val = train_test_split(x_normal, test_size = 0.2 , random_state = 42)

input_dim = x_train.shape[1] # Shape[1] means we are givein gall the 30 columns ( time, amount and v1-v28)...

# Build AutoEncoder
autoencoder= models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(16,activation="relu"),
    layers.Dense(8,activation="relu"),
    layers.Dense(16,activation="relu"),
    layers.Dense(input_dim,activation="linear")
])

autoencoder.compile(
    optimizer= "adam",
    loss = "mse"
)

#train
history = autoencoder.fit(
    x_train,x_train,
    epochs = 20,
    batch_size = 256,
    validation_data=(x_val, x_val),
    shuffle = True

)

autoencoder.save("models/autoencoder_model.keras")
joblib.dump(scaler,"models/AEscaler.pkl")