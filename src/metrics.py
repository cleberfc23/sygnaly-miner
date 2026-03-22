from sklearn.metrics import silhouette_score
import numpy as np

def compute_metrics(X, labels, model):
    inertia = model.inertia_
    silhouette = silhouette_score(X, labels)
    return inertia, np.mean(silhouette), np.std(silhouette)
