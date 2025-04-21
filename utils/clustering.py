from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

def find_optimal_clusters(scaled_features, max_k=10):
    inertia = []
    silhouette_scores = []
    davies_bouldin_scores = []
    K_range = range(2, max_k+1)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(scaled_features)
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(scaled_features, labels))
        davies_bouldin_scores.append(davies_bouldin_score(scaled_features, labels))

    return {
        'inertia': inertia,
        'silhouette': silhouette_scores,
        'davies_bouldin': davies_bouldin_scores,
        'K_range': list(K_range)
    }

def perform_clustering(scaled_features, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_features)
    return cluster_labels