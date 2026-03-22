from sklearn.metrics import silhouette_score


def compute_metrics(X, labels, model):
    inertia = model.inertia_
    silhouette = silhouette_score(X, labels)
    return inertia, silhouette
