# Sygnały - Public Signals Mining 
![Status](https://img.shields.io/badge/status-active%20development-yellow?style=for-the-badge)

An exploration of how machine learning can discover meaningful signals in large collections of public text.

## The Problem

Large volumes of public text contain valuable information about events, trends, and emerging issues.  
However, extracting structure from thousands of documents is difficult without automated analysis.

## The Approach

This project transforms raw text into structured signals using unsupervised learning, combining vector representations with clustering to identify recurring themes.

## Engineering Focus

Building a clear, reproducible ML workflow evolving from data exploration into a modular and deployable system.

## Dataset

- **248,123 documents** (Polish news corpus)
- Fields: `title`, `headline`, `content`, `link`
- Development subset: **3,000 documents**
- Final cleaned dataset: **2,890 documents (~96.3% retained)**

Source:  
https://huggingface.co/datasets/WiktorS/polish-news

## Baseline Pipeline

- **Vectorization:** CountVectorizer  
- **Clustering:** KMeans (k = 8, empirically chosen)  
- **Evaluation:**  
  - Inertia  
  - Silhouette Score: **0.07**  

## Visualizations

- PCA projection of clusters  
- Cluster distribution  
- Top terms per cluster  

## 🌐 Live Demo
https://sygnaly-miner.streamlit.app/

## Key Insight

The baseline shows **low cluster separability (silhouette ≈ 0.07)**, indicating that Bag-of-Words is insufficient for capturing semantic structure.

## Current Status

- End-to-end baseline pipeline implemented  
- Clustering, evaluation, and visualization completed  
- Streamlit app under development  

## Next Steps

- Introduce TF-IDF and semantic embeddings  
- Improve clustering quality  
- Deploy interactive application  