from dotenv import load_dotenv
import os
import pandas as pd
load_dotenv()
ROOT = os.getenv("PROJECT_ROOT")

def load_data():
    path = f'{ROOT}/data/dev/sygnaly_dev_clean_3000.csv'
    return pd.read_csv(path)