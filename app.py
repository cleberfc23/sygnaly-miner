import pandas as pd
from src.vectorization import build_bow
from src.clustering import run_kmeans
from src.metrics import compute_metrics
from ui.ui import build_ui, build_figures
from config.config import N_CLUSTERS
from IPython.display import display
import numpy as np
from data.load_data import load_data
import streamlit as st


st.set_page_config(
    page_title="Sygnały Miner",
    layout="wide"
)
#
st.title("Sygnały Miner  - -TEST-TESTT-TEST - build_figures - fig_cluster")
st.caption("Baseline NLP clustering with CountVectorizer + KMeans")


dataframe = load_data()

X_content_vectorized, vectorizer = build_bow(dataframe['content'])

kmeans, labels = run_kmeans(X_content_vectorized, N_CLUSTERS)

dataframe['cluster'] = labels

inertia, silhouette_mean, silhouette_std = compute_metrics(
    X_content_vectorized, labels, kmeans)

fig_word_cloud, fig_top_words, fig_cluster = build_figures(
    dataframe, X_content_vectorized)

# build_ui(dataframe, inertia, silhouette_mean, silhouette_std,
#          fig_word_cloud, fig_cluster, fig_top_words)
