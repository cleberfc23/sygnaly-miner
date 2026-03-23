from dotenv import load_dotenv
import os
import pandas as pd
from pathlib import Path
import streamlit as st

@st.cache_data
def load_data():
    file_id = "1F_ADRBo3cEBLdF4fHsv7mGCYZXjWad7l"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    return pd.read_csv(url)