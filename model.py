import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score,recall_score, f1_score, roc_auc_score ,roc_curve
import joblib 
from xgboost import XGBClassifier


#load the data using pandas.
df = pd.read_csv("transactiondata.csv")

#seperate the input and out put to teach the mpdel 
#x = other than class,y = class

x=df.drop("Class",axis=1)
y=df["Class"]

x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.2,random_state=42,stratify=y

    )

#to scale AMOUNT and TIME values acc to other values 
scaler= StandardScaler()
x_train_scaled= scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


#introducing SMOTE to handle imabalanced data 
smote = SMOTE(random_state = 42)
x_train_smote,y_train_smote = smote.fit_resample(x_train_scaled,y_train)

#FRAUD DETECT THROGH XGBOOST
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1,   # SMOTE already handled imbalance
    eval_metric="logloss",
    random_state=42
  
)
model.fit(x_train_smote,y_train_smote)
y_pred = model.predict(x_test_scaled)

y_prob = model.predict_proba(x_test_scaled)[:,1]
threshold = 0.45
y_pred_custom = (y_prob >= threshold).astype(int)

print("XGBOOST + SMOTE")
print("CONFUSION MATRIX:")
print(confusion_matrix(y_test, y_pred_custom))
print("RECALL:", round(recall_score(y_test, y_pred_custom),2))
print("PRECISION:", precision_score(y_test, y_pred_custom))
print("ROC_AUC:", round(roc_auc_score(y_test, y_prob),2))
print("F1_score:", round(f1_score(y_test, y_pred_custom),2))



# To Save The Trained Model
joblib.dump(model,"model.pkl") 
joblib.dump(scaler,"scaler.pkl")























