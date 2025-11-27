import joblib
import pandas as pd
import os
from dotenv import load_dotenv
import warnings

warnings.filterwarnings('ignore')
load_dotenv()


class ModelPrediction:

    def __init__(self):
        self.model_path = os.getenv("RANDOM_FOREST_MODEL")
        self.encoder_path = os.getenv("LABEL_ENCODER_MODEL")
        self.model = None
        self.encoders = None
        self.input_df = None
        self.result = None

    # -------------------------------------------------------
    # 1️⃣ Load Model & Encoders
    # -------------------------------------------------------
    def load_model_and_encoders(self):
        self.model = joblib.load(self.model_path)
        self.encoders = joblib.load(self.encoder_path)
        print("✔ Model & Encoders Loaded Successfully")

    # -------------------------------------------------------
    # 2️⃣ Load User Input (dictionary)
    # -------------------------------------------------------
    def load_input(self, data: dict):
        self.input_df = pd.DataFrame([data])
        print("✔ Input Loaded")

    # -------------------------------------------------------
    # 3️⃣ Encode Categorical Features
    # -------------------------------------------------------
    def encode_features(self):
        categorical_cols = ['Company', 'TypeName', 'Cpu_name', 'Gpu_name', 'OpSys']

        for col in categorical_cols:
            self.input_df[col] = self.encoders[col].transform(self.input_df[col])

        print("✔ Categorical Encoding Applied")

    # -------------------------------------------------------
    # 4️⃣ Predict Price
    # -------------------------------------------------------
    def predict(self):
        self.result = self.model.predict(self.input_df)[0]
        return self.result


# ===================================================================
# HOW TO RUN THIS IN VSCODE
# ===================================================================

if __name__ == "__main__":
    # Example Input
    input_data = {
         
    "Company": "Dell",
    "TypeName": "Notebook",
        "Ram": 8,
    "OpSys": "Windows 10",
     "Weight": 1.8,
     'TouchScreen': 0,
     'IPS': 1,
    "X_res": 1920,
    "Y_res": 1080,
    "PPI": 141.21,
    "Cpu_name": "Intel Core i5",
    "Gpu_name": "Intel",
    "HDD": 0,
    "SSD": 512,
    "Hybrid": 0,
    "Flash_Storage": 0,
 
  }

    mp = ModelPrediction()
    mp.load_model_and_encoders()
    mp.load_input(input_data)
    mp.encode_features()
    prediction = mp.predict()

    print("\nPredicted Laptop Price: ₹", round(prediction, 2))
