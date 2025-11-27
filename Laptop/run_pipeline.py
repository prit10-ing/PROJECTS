# ================================
# run_pipeline.py
# Master ML Pipeline Runner
# ================================

from data_ingestion import DataIngestion
from data_preprocessing import PreprocessingPipeline
from model_training import ModelTrainingEvaluation
from model_prediction import ModelPrediction

print("\n🚀 Starting Full ML Pipeline...\n")

# --------------------------------
# 1️⃣ Data Ingestion
# --------------------------------
print("\n==============================")
print(" STEP 1: DATA INGESTION")
print("==============================")

di = DataIngestion()
raw_df = di.load_raw_data()


# --------------------------------
# 2️⃣ Data Preprocessing
# --------------------------------
print("\n==============================")
print(" STEP 2: PREPROCESSING")
print("==============================")

pp = PreprocessingPipeline()
pp.load_data()
pp.data_report()
pp.basic_cleaning()
pp.screen_features()
pp.cpu_cleaning()
pp.gpu_cleaning()
pp.memory_cleaning()
pp.drop_columns()
pp.save_clean()
pp.reload_and_save()


# --------------------------------
# 3️⃣ Model Training & Evaluation
# --------------------------------
print("\n==============================")
print(" STEP 3: MODEL TRAINING & EVALUATION")
print("==============================")

mt = ModelTrainingEvaluation()
mt.load_data()
mt.label_encode()
mt.split_data()
mt.scale_data()
mt.evaluate_models()
mt.evaluate_unscaled()
mt.tune_random_forest()
mt.save_models()


# --------------------------------
# 4️⃣ Model Prediction (Optional Test)
# --------------------------------
print("\n==============================")
print(" STEP 4: MODEL PREDICTION")
print("==============================")

sample_input = {
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
mp.load_input(sample_input)
mp.encode_features()
pred = mp.predict()

print("\n💰 Predicted Sample Laptop Price:", round(pred, 2))


print("\n✅ FULL PIPELINE COMPLETED SUCCESSFULLY!")
   