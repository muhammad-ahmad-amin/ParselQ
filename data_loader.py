import pandas as pd
from sklearn.model_selection import train_test_split
import os


class DataLoader:
    def __init__(self, csv_path, save_name=None, test_size=0.2, random_state=42):
        self.csv_path = csv_path
        self.test_size = test_size
        self.random_state = random_state
        self.save_name = save_name

        os.makedirs(os.path.join("main", "csv"), exist_ok=True)

    def load_dataset(self):
        try:
            df = pd.read_csv(self.csv_path)

            # Strip any leading/trailing spaces from column names
            df.columns = df.columns.str.strip()

            print(f"[INFO] Loaded dataset: {df.shape[0]} rows")
            print(f"[INFO] Columns detected: {df.columns.tolist()}")

            # Determine save path
            save_path = (
                os.path.join("main", "csv", self.save_name)
                if self.save_name
                else os.path.join("main", "csv", os.path.basename(self.csv_path))
            )

            # Save CSV copy
            df.to_csv(save_path, index=False)
            print(f"[INFO] Saved dataset copy to {save_path}")

            return df
        except FileNotFoundError:
            raise FileNotFoundError("Dataset file not found. Check the CSV path.")

    def split_dataset(self, df, label_column="label"):
        df.columns = [col.lower().strip() for col in df.columns]

        # Check if the label column exists
        if label_column.lower() not in df.columns:
            print(f"[WARNING] Column '{label_column}' not found. Splitting without stratification.")
            train_df, test_df = train_test_split(
                df,
                test_size=self.test_size,
                random_state=self.random_state
            )
        else:
            train_df, test_df = train_test_split(
                df,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=df[label_column.lower()]
            )

        print(f"[INFO] Training samples: {train_df.shape[0]}")
        print(f"[INFO] Testing samples: {test_df.shape[0]}")
        return train_df, test_df


if __name__ == "__main__":
    loader = DataLoader(
        "main/csv/full_dataset_cleaned.csv", 
        save_name="dataset_copy.csv"
    )

    df = loader.load_dataset()
    train_df, test_df = loader.split_dataset(df)