# utils/plots.py (обновленный)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


def plot_elbow_method(inertia, K_range):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(K_range, inertia, marker='o', linestyle='--')
    ax.set_xlabel('Number of clusters (k)')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Method')
    return fig


def plot_silhouette(silhouette_scores, K_range):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(K_range, silhouette_scores, marker='o', linestyle='--', color='green')
    ax.set_xlabel('Number of clusters (k)')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Analysis')
    return fig


def plot_davies_bouldin(davies_bouldin_scores, K_range):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(K_range, davies_bouldin_scores, marker='o', linestyle='--', color='orange')
    ax.set_xlabel('Number of clusters (k)')
    ax.set_ylabel('Davies-Bouldin Score')
    ax.set_title('Davies-Bouldin Analysis')
    return fig


def plot_pca_clusters(scaled_features, cluster_labels):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_features)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1],
                    hue=cluster_labels, palette='Set1', ax=ax)
    ax.set_title('Clusters Visualization (PCA)')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    return fig