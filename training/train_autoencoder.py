import pandas as pd 
import os


BASE_DIR =os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path =os.path.join(BASE_DIR,"data","transactiondata.csv")


df = pd.read_csv(data_path)

df_normal = df[df["Class"] == 0]
df_fraud = df[df["Class"] == 1]

# AUTOENCODER is a neural network that learn " what does normal data look like ?" 
# it just learns the pattern of normal transactions.
