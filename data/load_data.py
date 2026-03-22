from dotenv import load_dotenv
import os
import pandas as pd
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dev" / "sygnaly_dev_clean_3000.csv"

def load_data():
    return pd.read_csv(DATA_PATH)
