from sklearn.cluster import KMeans
from config.config import RANDOM_STATE


def run_kmeans(X, number_clusters):
    kmeans = KMeans(n_clusters=number_clusters,
                    random_state=RANDOM_STATE,
                    n_init = 'auto').fit(X)
    
    return kmeans, kmeans.labels_
