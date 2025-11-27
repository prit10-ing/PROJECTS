import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

class DataIngestion:

    def __init__(self):
        self.raw_path = os.getenv("RAW_DATA_PATH")
        self.df = None

    # Load Raw Dataset
    def load_raw_data(self):
        self.df = pd.read_csv(self.raw_path)
        print("✔ Raw Data Loaded Successfully")
        return self.df

    # Optional: save backup raw data
    def save_raw_copy(self, output_path):
        self.df.to_csv(output_path, index=False)
        print("✔ Raw Data Backup Saved Successfully")


# Run manually (VS Code)
if __name__ == "__main__":
    di = DataIngestion()
    df = di.load_raw_data()
    print(df.head())
