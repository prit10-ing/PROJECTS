import os
from dotenv import load_dotenv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error
import joblib


class ModelTrainingEvaluation:

    def __init__(self):
        load_dotenv()
        self.clean_data_path = "D:\\Virtual\\Data\\clean_data.csv"
        self.model_path = "D://Virtual//Models//randomperfect_forest_classifier.pkl"
        self.encoder_path = "D://Virtual//Models//all_label_encoders.pkl"

        self.test_size = 0.20
        self.random_state = 42
        self.df = None
        self.encoders = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.best_rf = None



    # ------------------------------------------------------------
    # 1️⃣ Load Data
    # ------------------------------------------------------------
    def load_data(self):
        self.df = pd.read_csv(self.clean_data_path)



    # ------------------------------------------------------------
    # 2️⃣ Label Encoding
    # ------------------------------------------------------------
    def label_encode(self):
        cols_to_encode = ['Company', 'Cpu_name', 'Gpu_name', 'OpSys', 'TypeName']

        for col in cols_to_encode:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.encoders[col] = le



    # ------------------------------------------------------------
    # 3️⃣ Train-Test Split
    # ------------------------------------------------------------
    def split_data(self):
        X = self.df.drop('Price', axis=1)
        y = self.df['Price']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )



    # ------------------------------------------------------------
    # 4️⃣ Standard Scaling
    # ------------------------------------------------------------
    def scale_data(self):
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)



    # ------------------------------------------------------------
    # 5️⃣ Train + Evaluate Multiple Models
    # ------------------------------------------------------------
    def evaluate_models(self):
        models = {
            'Linear_Regression': LinearRegression(),
            'KNN': KNeighborsRegressor(),
            'SVR': SVR(),
            'Decision_Tree': DecisionTreeRegressor(),
            'Random_Forest': RandomForestRegressor()
        }

        results = []

        for name, model in models.items():
            model.fit(self.X_train_scaled, self.y_train)
            y_pred = model.predict(self.X_test_scaled)

            results.append({
                'Model': name,
                'R2_Score': r2_score(self.y_test, y_pred),
                'MAE': mean_absolute_error(self.y_test, y_pred)
            })

        print("\n===== Scaled Model Results =====")
        for r in results:
            print(r)



    # ------------------------------------------------------------
    # 6️⃣ Decision Tree + RF Without Scaling
    # ------------------------------------------------------------
    def evaluate_unscaled(self):
        models = {
            'decision_tree_no_scaling': DecisionTreeRegressor(),
            'random_forest_no_scaling': RandomForestRegressor()
        }

        results = []

        for name, model in models.items():
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)

            results.append({
                'Model': name,
                'R2_Score': r2_score(self.y_test, y_pred),
                'MAE': mean_absolute_error(self.y_test, y_pred)
            })

        print("\n===== Unscaled Model Results =====")
        for r in results:
            print(r)



    # ------------------------------------------------------------
    # 7️⃣ Hyperparameter Tuning of Random Forest
    # ------------------------------------------------------------
    def tune_random_forest(self):
        rf = RandomForestRegressor()

        param_dist = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }

        rnd_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_dist,
            n_iter=30,
            cv=5,
            n_jobs=-1,
            scoring='r2',
            verbose=2,
            random_state=42
        )

        rnd_search.fit(self.X_train, self.y_train)
        self.best_rf = rnd_search.best_estimator_

        y_pred_test = self.best_rf.predict(self.X_test)
        print("\nBest RF R2 Score:", r2_score(self.y_test, y_pred_test))



    # ------------------------------------------------------------
    # 8️⃣ Save Best Model + Encoders
    # ------------------------------------------------------------
    def save_models(self):
        joblib.dump(self.best_rf, self.model_path)
        joblib.dump(self.encoders, self.encoder_path)
        print("\n🎉 Model & Encoders Saved Successfully")



# ===================================================================
# HOW TO RUN THIS IN VSCODE
# ===================================================================

if __name__ == "__main__":
    mt = ModelTrainingEvaluation()
    mt.load_data()
    mt.label_encode()
    mt.split_data()
    mt.scale_data()
    mt.evaluate_models()
    mt.evaluate_unscaled()
    mt.tune_random_forest()
    mt.save_models()

    print("\n✅ Model Training + Evaluation Completed Successfully!")
