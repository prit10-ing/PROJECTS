# 💻 Laptop Price Prediction – Machine Learning Project

A complete end-to-end machine learning pipeline that predicts laptop prices based on specifications such as RAM, storage, CPU, GPU, screen resolution, and display features.

This project includes:
- Data ingestion from MySQL
- Data cleaning & preprocessing
- Feature engineering
- Exploratory Data Analysis (EDA)
- Multiple ML model training & evaluation
- Hyperparameter tuning (RandomizedSearchCV)
- Final Random Forest model saving
- Production-ready `.env` configuration
- GitHub-ready project folder structure

---

## 🚀 Project Workflow

### **1. Data Ingestion**
- Connected to MySQL using a dedicated `DataIngestion` class.
- Extracted `laptop_data` table.
- Saved raw dataset to CSV (`laptop_data.csv`).

### **2. Data Cleaning**
Performed using a preprocessing script:
- Removed units (GB, kg)
- Fixed screen resolution formats
- Extracted numeric values from strings
- Removed unnecessary text from CPU & GPU columns
- Cleaned memory column and split into:
  - HDD
  - SSD
  - Hybrid
  - Flash Storage

### **3. Feature Engineering**
- Created new columns:
  - `TouchScreen`
  - `IPS`
  - `X_res`, `Y_res`
  - `PPI` (Pixels Per Inch)
  - `Cpu_name`
  - `Gpu_name`
- Scaled features using `StandardScaler`
- Applied `LabelEncoder` to categorical columns

### **4. Exploratory Data Analysis (EDA)**
- Distribution plots
- Value counts
- Heatmaps
- Price comparisons:
  - CPU vs Price
  - GPU vs Price
  - TouchScreen vs Price
  - IPS vs Price

### **5. Modeling**
Trained and evaluated multiple models:
- Linear Regression  
- KNN  
- SVR  
- Decision Tree  
- Random Forest  

Also evaluated tree models *without scaling*.

### **6. Hyperparameter Tuning**
Used `RandomizedSearchCV` to optimize:
- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `max_features`

Final model: **Optimized Random Forest Regressor**

### **7. Saving the Model**
Saved using joblib:
- `random_forest_classifier.pkl`
- `label_encoder.pkl`

---

## 🧠 Model Performance

| Model | R² Score | MAE |
|-------|---------|--------|
| Linear Regression | Moderate | Higher error |
| KNN | Low | Weak generalization |
| SVR | Low | Underperformed |
| Decision Tree | Good | Slight overfit |
| Random Forest | **Best** | **Lowest MAE** |
| Tuned Random Forest | **Highest R²** | **Best performer** |

✔ Tuned Random Forest is the **final selected model**.

---

## 📁 Project Structure

project/
│
├── data/
│ ├── laptop_data.csv
│ └── clean_data.csv
│
├── models/
│ ├── random_forest_classifier.pkl
│ └── label_encoder.pkl
│
├── src/
│ ├── ingestion.py
│ ├── preprocessing.py
│ ├── train_model.py
│ ├── predict.py
│ └── utils/
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md