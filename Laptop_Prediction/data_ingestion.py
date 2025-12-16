import pandas as pd
import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv()

class DataIngestion:

    def __init__(self):
        self.host = os.getenv("DB_HOST")
        self.port = os.getenv("MYSQL_PORT")
        self.database = os.getenv("DB_NAME")
        self.user = os.getenv("DB_USER")
        self.password = os.getenv("DB_PASSWORD")
        self.table = os.getenv("MYSQL_TABLE")

        self.df = None
        self.connection = None

    # Create DB connection
    def connect_db(self):
        self.connection = mysql.connector.connect(
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password
        )
        print("✔ Connected to MySQL Database")

    # Load data from SQL table
    def load_raw_data(self):
        if self.connection is None:
            self.connect_db()

        query = f"SELECT * FROM {self.table}"
        self.df = pd.read_sql(query, self.connection)

        print("✔ Data Loaded Successfully from MySQL")
        return self.df

    # Optional: save backup
    def save_raw_copy(self, output_path):
        if self.df is not None:
            self.df.to_csv(output_path, index=False)
            print("✔ MySQL Data Backup Saved Successfully")

    # Close DB connection
    def close_connection(self):
        if self.connection:
            self.connection.close()
            print("✔ MySQL Connection Closed")


# Run manually (VS Code)
if __name__ == "__main__":
    di = DataIngestion()
    df = di.load_raw_data()
    print(df.head())
    di.close_connection()
di.save_raw_copy("D:\\Virtual\\Data\\laptop_data.csv")