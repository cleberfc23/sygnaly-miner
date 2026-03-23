from dotenv import load_dotenv
import os
import pandas as pd
from pathlib import Path
# BASE_DIR = Path(__file__).resolve().parent
# DATA_PATH = BASE_DIR / "dev" / "sygnaly_dev_clean_3000.csv"

# def load_data():
#     path_url = 'https://drive.google.com/file/d/1F_ADRBo3cEBLdF4fHsv7mGCYZXjWad7l/view?usp=sharing'
#     return pd.read_csv(path_url)



def load_data():
    file_id = '1F_ADRBo3cEBLdF4fHsv7mGCYZXjWad7l'
    path_url = f'https://drive.google.com/uc?id={file_id}'
    return pd.read_csv(path_url)