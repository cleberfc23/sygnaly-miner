import pandas as pd
from src.vectorization import build_bow
from src.clustering import run_kmeans
from src.metrics import compute_metrics
from ui.ui import build_ui
from config.config import N_CLUSTERS
from IPython.display import display
import numpy as np
from data.load_data import load_data

# df = pd.read_csv(
#     "/Users/cleberfcarvalho/Documents/myGitHub/citizen-signals-miner/data/dev/sygnaly_dev_clean_3000.csv")

dataframe = load_data()

X_content_vectorized, vectorizer = build_bow(dataframe['content'])

kmeans, labels = run_kmeans(X_content_vectorized, N_CLUSTERS)
dataframe['cluster'] = labels

inertia, silhouette = compute_metrics(X_content_vectorized, labels, kmeans)

display(dataframe.head())
print(f'inertia: {inertia}')
print(f'silhouette: {np.mean(silhouette)} +/- {np.std(silhouette)}')
