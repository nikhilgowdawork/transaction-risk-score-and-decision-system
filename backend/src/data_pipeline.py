import os
import pandas as pd


class PaySimDataPipeline:
    """
    Data loading, feature engineering, sampling,
    and time-series train/test splitting pipeline
    tailored for the PaySim transaction fraud dataset.
    """

    def __init__(
        self,
        raw_filepath: str,
        sample_size: int = 200000,
        random_state: int = 42
    ):
        self.raw_filepath = raw_filepath
        self.sample_size = sample_size
        self.random_state = random_state

        # Folder where processed datasets will be saved
        self.data_dir = os.path.dirname(self.raw_filepath)

        self.train_filepath = os.path.join(
            self.data_dir, "train.csv"
        )

        self.test_filepath = os.path.join(
            self.data_dir, "test.csv"
        )

    def load_and_sample_data(self) -> pd.DataFrame:
        """
        Loads the original dataset, filters relevant transaction types,
        samples the required number of records, and sorts by time.
        """

        if not os.path.exists(self.raw_filepath):
            raise FileNotFoundError(
                f"Raw dataset not found at {self.raw_filepath}"
            )

        print(f"Loading raw dataset from {self.raw_filepath}...")

        df = pd.read_csv(self.raw_filepath)

        # PaySim fraud occurs almost exclusively
        # in TRANSFER and CASH_OUT transactions
        df = df[
            df["type"].isin(["TRANSFER", "CASH_OUT"])
        ].reset_index(drop=True)

        # Sample if dataset is larger than required
        if len(df) > self.sample_size:
            print(
                f"Sampling {self.sample_size} records..."
            )

            df = df.sample(
                n=self.sample_size,
                random_state=self.random_state
            )

        # Sort chronologically
        df = df.sort_values(
            by="step"
        ).reset_index(drop=True)

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates additional features useful for fraud detection.
        """

        df = df.copy()

        # TRANSFER = 1
        # CASH_OUT = 0
        if "type" in df.columns:
            df["type_TRANSFER"] = (
                df["type"] == "TRANSFER"
            ).astype(int)

            df = df.drop(columns=["type"])

        # Origin balance discrepancy
        df["errorBalanceOrig"] = (
            df["newbalanceOrig"]
            + df["amount"]
            - df["oldbalanceOrg"]
        )

        # Destination balance discrepancy
        df["errorBalanceDest"] = (
            df["oldbalanceDest"]
            + df["amount"]
            - df["newbalanceDest"]
        )

        # Merchant destination indicator
        if "nameDest" in df.columns:
            df["isMerchantDest"] = (
                df["nameDest"]
                .astype(str)
                .str.startswith("M")
                .astype(int)
            )

        elif "isMerchantDest" not in df.columns:
            df["isMerchantDest"] = 0

        # Zero balance indicators
        df["zeroBalOrigAfter"] = (
            df["newbalanceOrig"] == 0
        ).astype(int)

        df["zeroBalDestBefore"] = (
            df["oldbalanceDest"] == 0
        ).astype(int)

        # Remove identifiers and unnecessary column
        df = df.drop(
            columns=[
                "nameOrig",
                "nameDest",
                "isFlaggedFraud"
            ],
            errors="ignore"
        )

        return df

    def save_processed_data(
        self,
        processed_df: pd.DataFrame,
        test_size: float = 0.2
    ):
        """
        Performs chronological train/test split
        and saves the processed datasets.
        """

        split_idx = int(
            len(processed_df) * (1 - test_size)
        )

        train_df = processed_df.iloc[:split_idx].copy()
        test_df = processed_df.iloc[split_idx:].copy()

        train_df.to_csv(
            self.train_filepath,
            index=False
        )

        test_df.to_csv(
            self.test_filepath,
            index=False
        )

        print("\nProcessed datasets saved:")
        print(f" - Train: {self.train_filepath}")
        print(f" - Test:  {self.test_filepath}")

        return train_df, test_df

    def load_processed_data(self):
        """
        Loads previously processed train/test datasets.
        This avoids loading the original 6.3M-row dataset.
        """

        if not os.path.exists(self.train_filepath):
            raise FileNotFoundError(
                f"Train dataset not found: {self.train_filepath}"
            )

        if not os.path.exists(self.test_filepath):
            raise FileNotFoundError(
                f"Test dataset not found: {self.test_filepath}"
            )

        print("Loading previously processed datasets...")

        train_df = pd.read_csv(self.train_filepath)
        test_df = pd.read_csv(self.test_filepath)

        return train_df, test_df

    def prepare_pipeline(self, test_size: float = 0.2):
        """
        Creates processed datasets only once.

        If train.csv and test.csv already exist,
        they are loaded directly.
        """

        # ---------------------------------------------------------
        # STEP 1: Check whether processed datasets already exist
        # ---------------------------------------------------------

        if (
            os.path.exists(self.train_filepath)
            and os.path.exists(self.test_filepath)
        ):
            print("\nProcessed datasets already exist.")
            print("Skipping original 6.3M-row dataset.")

            train_df, test_df = self.load_processed_data()

        # ---------------------------------------------------------
        # STEP 2: Process original dataset only once
        # ---------------------------------------------------------

        else:
            print("\nProcessed datasets not found.")
            print("Running full preprocessing for the first time...")

            raw_df = self.load_and_sample_data()

            processed_df = self.engineer_features(raw_df)

            train_df, test_df = self.save_processed_data(
                processed_df,
                test_size=test_size
            )

        # ---------------------------------------------------------
        # STEP 3: Separate X and y
        # ---------------------------------------------------------

        X_train = train_df.drop(
            columns=["isFraud"]
        )

        y_train = train_df["isFraud"]

        X_test = test_df.drop(
            columns=["isFraud"]
        )

        y_test = test_df["isFraud"]

        # ---------------------------------------------------------
        # STEP 4: Display information
        # ---------------------------------------------------------

        print("\nPipeline Ready:")

        print(
            f" - Train Shape: {X_train.shape}"
            f" | Fraud Count: {y_train.sum()}"
        )

        print(
            f" - Test Shape: {X_test.shape}"
            f" | Fraud Count: {y_test.sum()}"
        )

        print(
            f" - Features ({len(X_train.columns)}): "
            f"{list(X_train.columns)}"
        )

        return X_train, X_test, y_train, y_test


# ================================================================
# RUN PIPELINE
# ================================================================

if __name__ == "__main__":

    DATA_PATH = os.path.join(
        "backend",
        "data",
        "transactiondata.csv"
    )

    if not os.path.exists(DATA_PATH):
        DATA_PATH = os.path.join(
            "backend",
            "data",
            "paysim.csv"
        )

    pipeline = PaySimDataPipeline(
        raw_filepath=DATA_PATH,
        sample_size=200000
    )

    X_train, X_test, y_train, y_test = (
        pipeline.prepare_pipeline()
    )