import streamlit as st
import joblib
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("RANDOM_FOREST_MODEL")
ENCODER_PATH = os.getenv("LABEL_ENCODER_MODEL")

# Load Model & Encoders
model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODER_PATH)

st.title("💻 Laptop Price Prediction App")

# ==========================
# Input Fields
# ==========================

company = st.selectbox("Company", ["Dell", "HP", "Lenovo", "Asus", "Acer"])
typename = st.selectbox("Laptop Type", ["Notebook", "Gaming", "Ultrabook", "Workstation"])
cpu = st.selectbox("CPU", ["Intel Core i3", "Intel Core i5", "Intel Core i7"])
gpu = st.selectbox("GPU", ["Intel", "Nvidia", "AMD"])
opsys = st.selectbox("Operating System", ["Windows 10", "Linux", "No OS", "macOS"])

ram = st.number_input("RAM (GB)", min_value=2, max_value=64)
weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0)

# Screen Inputs
touchscreen = st.selectbox("TouchScreen", [0, 1])
ips = st.selectbox("IPS Display", [0, 1])

x_res = st.number_input("X Resolution", min_value=800)
y_res = st.number_input("Y Resolution", min_value=600)

ppi = st.number_input("PPI", min_value=50.0)

# Storage Inputs
hdd = st.number_input("HDD (GB)", min_value=0)
ssd = st.number_input("SSD (GB)", min_value=0)
hybrid = st.number_input("Hybrid (GB)", min_value=0)
flash = st.number_input("Flash Storage (GB)", min_value=0)


# ==========================
# Safe Encoder
# ==========================

def safe_transform(encoder, value):
    if value not in encoder.classes_:
        st.warning(f"⚠️ '{value}' unseen in training → assigning -1.")
        return -1
    return encoder.transform([value])[0]


# ==========================
# Final Input DataFrame
# ORDER MUST MATCH TRAINING FEATURES
# ==========================

input_data = pd.DataFrame([[
    safe_transform(encoders['Company'], company),   # 1
    safe_transform(encoders['TypeName'], typename), # 2
    ram,                                            # 3
    safe_transform(encoders['OpSys'], opsys),       # 4
    weight,
    touchscreen,                                    # 15
    ips, 
                                                                                    # 5
    x_res,                                          # 6
    y_res,                                          # 7
    ppi,                                            # 8
    safe_transform(encoders['Cpu_name'], cpu),      # 9
    safe_transform(encoders['Gpu_name'], gpu),      # 10
    hdd,                                            # 11
    ssd,                                            # 12
    hybrid,                                         # 13
    flash                                          # 14
                                                # 16
]], columns=[
    'Company', 'TypeName', 'Ram', 'OpSys', 'Weight',
    'TouchScreen','IPS',
    'X_res', 'Y_res', 'PPI', 'Cpu_name', 'Gpu_name',
    'HDD', 'SSD', 'Hybrid', 'Flash_Storage'
     
])

# ==========================
# Predict Button
# ==========================

if st.button("Predict Price"):
    try:
        price = model.predict(input_data)[0]
        st.success(f"💰 Predicted Laptop Price: ₹{price:.2f}")
    except Exception as e:
        st.error(f"❌ Prediction Failed: {e}")
