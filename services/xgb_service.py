
import os 
import pandas as pd 
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score,f1_score,recall_score,confusion_matrix,roc_auc_score


#load the data using pandas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path=os.path.join(BASE_DIR,"data","transactiondata.csv")
model_path =os.path.join(BASE_DIR,"models","xgb_model.pkl")
scaler_path = os.path.join(BASE_DIR,"models","scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

print(model_path, os.path.exists(model_path))
print(scaler_path, os.path.exists(scaler_path))


df = pd.read_csv(data_path)
#Seperate the data
x=df.drop("Class",axis=1) #Contains all columns exculding Class -- time,amount,v1-v28

y=df["Class"] #contains Class column only - this will be predictd by the model

#split the data -- in train_xbg.py we used x_train and y_train to train the model
# Now we are going to use x_test and y_test to test the model performance 
x_train,x_test,y_train,y_test=train_test_split(
        x,y,test_size=0.2,random_state=42,stratify=y

        )

    #to scale AMOUNT and TIME values acc to other values 
scaler= StandardScaler()
x_test_scaled= scaler.fit_transform(x_test)

y_pred = model.predict(x_test_scaled)

y_prob = model.predict_proba(x_test_scaled)[:,1]
threshold = 0.6
y_pred_custom = (y_prob >= threshold).astype(int)

print("XGBOOST + SMOTE")
print("CONFUSION MATRIX:")
print(confusion_matrix(y_test, y_pred_custom))
print("RECALL:", round(recall_score(y_test, y_pred_custom),2))
print("PRECISION:", precision_score(y_test, y_pred_custom))
print("ROC_AUC:", round(roc_auc_score(y_test, y_prob),2))
print("F1_score:", round(f1_score(y_test, y_pred_custom),2))