import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import warnings


warnings.filterwarnings('ignore')



class PreprocessingPipeline:

    def __init__(self):
        self.raw_path = "D:\\Virtual\\Data\\laptop_data.csv"
        self.clean_path = 'D:/Virtual/Data/clean_data.csv'
        self.df = None

    # -------------------------------------------------------
    # 1️⃣ Load Raw Data
    # -------------------------------------------------------
    def load_data(self):
        self.df = pd.read_csv(self.raw_path)

    # -------------------------------------------------------
    # 2️⃣ Full Data Report
    # -------------------------------------------------------
    def data_report(self):
        print(" DATA REPORT SUMMARY")
        print("="*50)

        print(f"\n Shape: {self.df.shape}")
        print("\n Data Types:")
        print(self.df.dtypes)

        print("\n Info:")
        self.df.info()

        print("\n Null Values:")
        print(self.df.isnull().sum())

        print("\n Duplicates:", self.df.duplicated().sum())

        print("\n Describe:")
        print(self.df.describe(include='all').transpose())

        print("\n Unique Values:")
        for col in self.df.columns:
            print(f"{col}: {self.df[col].nunique()}")

        print("\n Value Counts:")
        for i in self.df:
            print(f'Column: {i}')
            print(self.df[i].value_counts())
            print("")

    # -------------------------------------------------------
    # 3️⃣ Basic Cleaning
    # -------------------------------------------------------
    def basic_cleaning(self):
        self.df["Ram"] = self.df["Ram"].str.replace("GB", "").astype(int)
        self.df["Weight"] = self.df["Weight"].str.replace("kg", "").astype(float)

    # -------------------------------------------------------
    # 4️⃣ Screen Feature Engineering
    # -------------------------------------------------------
    def screen_features(self):
        self.df["TouchScreen"] = self.df["ScreenResolution"].apply(lambda x: 1 if "Touchscreen" in x else 0)
        self.df["IPS"] = self.df["ScreenResolution"].apply(lambda x: 1 if "IPS" in x else 0)

        splitdf = self.df["ScreenResolution"].str.split("x", n=1, expand=True)
        self.df['X_res'] = splitdf[0].str.replace(",", "").str.findall(r'(\d+\.?\d+)').apply(lambda x: x[0]).astype(int)
        self.df['Y_res'] = splitdf[1].astype(int)

        self.df["PPI"] = (self.df["X_res"]**2 + self.df["Y_res"]**2)**0.5 / self.df["Inches"]
        self.df.drop(["ScreenResolution", "Inches"], axis=1, inplace=True)

    # -------------------------------------------------------
    # 📊 PLOT 1: Screen vs Price
    # -------------------------------------------------------
    def plot_screen_vs_price(self):
        plt.figure(figsize=(12,5))

        plt.subplot(1, 2, 1)
        sn.barplot(data=self.df, x='TouchScreen', y='Price', palette='plasma')
        plt.title("TouchScreen vs Price")

        plt.subplot(1, 2, 2)
        sn.barplot(data=self.df, x='IPS', y='Price', palette='plasma')
        plt.title("IPS vs Price")

        plt.tight_layout()
        plt.show()

    # -------------------------------------------------------
    # 5️⃣ CPU Cleaning
    # -------------------------------------------------------
    def cpu_cleaning(self):

        self.df["Cpu"] = self.df["Cpu"].apply(lambda t: " ".join(t.split()[0:3]))

        def update_cpu_name(cpu):
            if 'Intel' in cpu:
                if 'Core i7' in cpu:
                    return 'Intel Core i7'
                elif 'Core i5' in cpu:
                    return 'Intel Core i5'
                elif 'Core i3' in cpu:
                    return 'Intel Core i3'
                else:
                    return 'Intel Processor'
            elif 'AMD' in cpu:
                if "A9-Series" in cpu:
                    return 'AMD  A9-Series'
                elif "A6-Series" in cpu:
                    return 'AMD  A6-Series'
                elif "A10-Series" in cpu:
                    return 'AMD  A10-Series'
                elif "A12-Series" in cpu:
                    return 'AMD  A12-Series'
                else:
                    return "AMD Processor"
            else:
                return cpu

        self.df["Cpu_name"] = self.df["Cpu"].apply(update_cpu_name)

    # -------------------------------------------------------
    # 6️⃣ GPU Cleaning
    # -------------------------------------------------------
    def gpu_cleaning(self):
        def gpu(G):
            if "Intel" in G:
                return "Intel"
            elif "Nvidia" in G:
                return "Nvidia"
            elif "AMD" in G:
                return "AMD"
            else:
                return "Other"

        self.df['Gpu_name'] = self.df['Gpu'].apply(gpu)

    # -------------------------------------------------------
    # 📊 PLOT 2: CPU vs Price / GPU vs Price
    # -------------------------------------------------------
    def plot_gpu_cpu_vs_price(self):
        plt.figure(figsize=(12,5))

        plt.subplot(1, 2, 1)
        sn.barplot(data=self.df, x='Cpu_name', y='Price', palette='plasma')
        plt.xticks(rotation=30)
        plt.title("CPU vs Price")

        plt.subplot(1, 2, 2)
        sn.barplot(data=self.df, x='Gpu_name', y='Price', palette='plasma')
        plt.xticks(rotation=30)
        plt.title("GPU vs Price")

        plt.tight_layout()
        plt.show()

    # -------------------------------------------------------
    # 7️⃣ Memory Cleaning
    # -------------------------------------------------------
    def memory_cleaning(self):
        df = self.df

        df['Memory'] = df['Memory'].astype(str).replace(r'\.0', '', regex=True)
        df['Memory'] = df['Memory'].str.replace("GB", "")

        newdf = df["Memory"].str.split("+", n=1, expand=True)
        df["first"] = newdf[0].str.strip()
        df["second"] = newdf[1].fillna("0").str.strip()

        types = ["HDD", "SSD", "Hybrid", "Flash Storage"]

        for t in types:
            df["layer1_" + t] = df["first"].apply(lambda x: 1 if t in x else 0)

        df["first"] = df["first"].str.replace(r'\D', "", regex=True)
        df["first"] = pd.to_numeric(df["first"], errors="coerce")

        for t in types:
            df["layer2_" + t] = df["second"].apply(lambda x: 1 if t in x else 0)

        df["second"] = df["second"].str.replace(r'\D', "", regex=True)
        df["second"] = pd.to_numeric(df["second"], errors="coerce").fillna(0)

        df["HDD"] = df["first"] * df["layer1_HDD"] + df["second"] * df["layer2_HDD"]
        df["SSD"] = df["first"] * df["layer1_SSD"] + df["second"] * df["layer2_SSD"]
        df["Hybrid"] = df["first"] * df["layer1_Hybrid"] + df["second"] * df["layer2_Hybrid"]
        df["Flash_Storage"] = df["first"] * df["layer1_Flash Storage"] + df["second"] * df["layer2_Flash Storage"]

        df.drop(columns=[
            'first', 'second',
            'layer1_HDD', 'layer1_SSD', 'layer1_Hybrid', 'layer1_Flash Storage',
            'layer2_HDD', 'layer2_SSD', 'layer2_Hybrid', 'layer2_Flash Storage'
        ], inplace=True)

        self.df = df

    # -------------------------------------------------------
    # 📊 PLOT 3: Correlation Heatmap
    # -------------------------------------------------------
    def plot_correlation_heatmap(self):
        plt.figure(figsize=(10,7))
        sn.heatmap(self.df.corr(numeric_only=True), annot=True, cmap='plasma')
        plt.title("Correlation Heatmap")
        plt.show()

    # -------------------------------------------------------
    # 8️⃣ Drop Columns
    # -------------------------------------------------------
    def drop_columns(self):
        cols = ["Gpu", "Cpu", "Memory"]
        for col in cols:
            if col in self.df.columns:
                self.df.drop(col, axis=1, inplace=True)

        if "ï»¿" in self.df.columns:
            self.df.drop("ï»¿", axis=1, inplace=True)

    # -------------------------------------------------------
    # 9️⃣ Save Cleaned Data
    # -------------------------------------------------------
    def save_clean(self):
        self.df.to_csv(self.clean_path, index=False)

    # -------------------------------------------------------
    # 🔟 Reload & Save Again
    # -------------------------------------------------------
    def reload_and_save(self):
        df_clean = pd.read_csv(self.clean_path)
        df_clean.to_csv(self.clean_path, index=False)
if __name__ == "__main__":
    pp = PreprocessingPipeline()
    pp.load_data()
    pp.data_report()
    pp.basic_cleaning()
    pp.screen_features()
    pp.plot_screen_vs_price()
    pp.cpu_cleaning()
    pp.gpu_cleaning()
    pp.plot_gpu_cpu_vs_price()
    pp.memory_cleaning()
    pp.plot_correlation_heatmap()
    pp.drop_columns()
    pp.save_clean()
    pp.reload_and_save()

    print("\n✅ Preprocessing Completed Successfully!")
