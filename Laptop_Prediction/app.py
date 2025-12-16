import streamlit as st
import joblib
import pandas as pd
import os

# ==========================
# Paths (Cloud-safe)
# ==========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Models", "randomperfect_forest_classifier.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "Models", "all_label_encoders.pkl")

# ==========================
# Load model & encoders (cached)
# ==========================
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODER_PATH)
    return model, encoders

model, encoders = load_artifacts()

# ==========================
# App UI
# ==========================
st.title("💻 Laptop Price Prediction App")

company = st.selectbox("Company", ["Dell", "HP", "Lenovo", "Asus", "Acer"])
typename = st.selectbox("Laptop Type", ["Notebook", "Gaming", "Ultrabook", "Workstation"])
cpu = st.selectbox("CPU", ["Intel Core i3", "Intel Core i5", "Intel Core i7"])
gpu = st.selectbox("GPU", ["Intel", "Nvidia", "AMD"])
opsys = st.selectbox("Operating System", ["Windows 10", "Linux", "No OS", "macOS"])

ram = st.number_input("RAM (GB)", min_value=2, max_value=64)
weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0)

touchscreen = st.selectbox("TouchScreen", [0, 1])
ips = st.selectbox("IPS Display", [0, 1])

x_res = st.number_input("X Resolution", min_value=800)
y_res = st.number_input("Y Resolution", min_value=600)
ppi = st.number_input("PPI", min_value=50.0)

hdd = st.number_input("HDD (GB)", min_value=0)
ssd = st.number_input("SSD (GB)", min_value=0)
hybrid = st.number_input("Hybrid (GB)", min_value=0)
flash = st.number_input("Flash Storage (GB)", min_value=0)

# ==========================
# Safe encoder
# ==========================
def safe_transform(encoder, value):
    if value not in encoder.classes_:
        st.warning(f"⚠️ '{value}' unseen in training → assigning -1")
        return -1
    return encoder.transform([value])[0]

# ==========================
# Input DataFrame (order matters!)
# ==========================
input_data = pd.DataFrame([[
    safe_transform(encoders['Company'], company),
    safe_transform(encoders['TypeName'], typename),
    ram,
    safe_transform(encoders['OpSys'], opsys),
    weight,
    touchscreen,
    ips,
    x_res,
    y_res,
    ppi,
    safe_transform(encoders['Cpu_name'], cpu),
    safe_transform(encoders['Gpu_name'], gpu),
    hdd,
    ssd,
    hybrid,
    flash
]], columns=[
    'Company', 'TypeName', 'Ram', 'OpSys', 'Weight',
    'TouchScreen', 'IPS',
    'X_res', 'Y_res', 'PPI',
    'Cpu_name', 'Gpu_name',
    'HDD', 'SSD', 'Hybrid', 'Flash_Storage'
])

# ==========================
# Prediction
# ==========================
if st.button("Predict Price"):
    try:
        price = model.predict(input_data)[0]
        st.success(f"💰 Predicted Laptop Price: ₹{price:.2f}")
    except Exception as e:
        st.error(f"❌ Prediction Failed: {e}")
