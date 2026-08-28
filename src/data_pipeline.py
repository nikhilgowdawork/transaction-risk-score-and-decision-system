import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class PaySimDataPipeline:
    """
    Data loading, feature engineering, sampling, and time-series train/test splitting pipeline
    tailored for the PaySim transaction fraud dataset.
    """
    def __init__(self, raw_filepath: str, sample_size: int = 200000, random_state: int = 42):
        self.raw_filepath = raw_filepath
        self.sample_size = sample_size
        self.random_state = random_state

    def load_and_sample_data(self) -> pd.DataFrame:
        """Loads dataset and performs stratified, time-consistent sampling."""
        if not os.path.exists(self.raw_filepath):
            raise FileNotFoundError(f"Raw dataset not found at {self.raw_filepath}")

        print(f"Loading raw dataset from {self.raw_filepath}...")
        
        # Read columns needed for modeling
        df = pd.read_csv(self.raw_filepath)
        
        # PaySim fraud primarily occurs in 'TRANSFER' and 'CASH_OUT' transactions
        # Filtering irrelevant types improves learning speed and boundary density
        df = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])].reset_index(drop=True)
        
        # Perform time-preserving sampling if the dataset exceeds target size
        if len(df) > self.sample_size:
            print(f"Sampling {self.sample_size} records while maintaining class imbalance ratio...")
            df = df.sample(n=self.sample_size, random_state=self.random_state)
            
        # Crucial for stream simulation & concept drift: Sort chronologically by simulation hour
        df = df.sort_values(by='step').reset_index(drop=True)
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineers structural domain features from transaction balances."""
        print("Engineering domain features (balance dynamics & discrepancy indicators)...")
        df = df.copy()

        # One-hot encode categorical transaction types
        df = pd.get_dummies(df, columns=['type'], drop_first=True, dtype=int)

        # Feature 1: Balance Discrepancies (Origin Account)
        # Identifies unrecorded balance drops or phantom fund injections
        df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']

        # Feature 2: Balance Discrepancies (Destination Account)
        df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']

        # Feature 3: Zero Balance Flags (Frequent fraud indicator in PaySim)
        df['isMerchantDest'] = df['nameDest'].str.startswith('M').astype(int)
        df['zeroBalOrigAfter'] = (df['newbalanceOrig'] == 0).astype(int)
        df['zeroBalDestBefore'] = (df['oldbalanceDest'] == 0).astype(int)

        # Drop identifiers that cause overfitting (high-cardinality string IDs)
        df = df.drop(columns=['nameOrig', 'nameDest', 'isFlaggedFraud'], errors='ignore')
        
        return df

    def prepare_pipeline(self, test_size: float = 0.2):
        """Runs loading, feature engineering, and performs time-series train/test split."""
        raw_df = self.load_and_sample_data()
        processed_df = self.engineer_features(raw_df)

        X = processed_df.drop(columns=['isFraud'])
        y = processed_df['isFraud']

        # Time-Series / Chronological Split (No shuffle to avoid temporal data leakage)
        split_idx = int(len(processed_df) * (1 - test_size))
        
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"\nPipeline Ready:")
        print(f" - Total Samples Processed: {len(processed_df)}")
        print(f" - Features: {list(X.columns)}")
        print(f" - Train Shape: {X_train.shape} | Fraud Count: {y_train.sum()}")
        print(f" - Test Shape:  {X_test.shape}  | Fraud Count: {y_test.sum()}")

        return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Test path execution
    DATA_PATH = os.path.join("data", "transactiondata.csv")
    
    # Adjust filename if your downloaded Kaggle file is named differently
    if not os.path.exists(DATA_PATH):
        # Fallback check for alternative kaggle filename
        DATA_PATH = os.path.join("data", "paysim.csv")

    pipeline = PaySimDataPipeline(raw_filepath=DATA_PATH, sample_size=200000)
    X_train, X_test, y_train, y_test = pipeline.prepare_pipeline()