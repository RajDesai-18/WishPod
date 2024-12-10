import os
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from src.data_processing.feature_engineering import prepare_clustering_features
from src.utils.common_utils import euclidean_distance, initialize_centroids
from src.evaluation.metrics import evaluate_model_with_auc

# Paths
PROCESSED_DATA_PATH = "D:/Classes/Machine Learning/Project/Wishpod/data/processed/preprocessed_podcasts.csv"
CLUSTER_OUTPUT_PATH = "D:/Classes/Machine Learning/Project/Wishpod/data/clustering/"
CLUSTERED_DATA_PATH = os.path.join(CLUSTER_OUTPUT_PATH, "clustered_podcasts.csv")

# Finalized number of clusters
N_CLUSTERS = 10

# K-Means Clustering Functions
def assign_clusters(X, centroids):
    """Assign each data point to the closest centroid."""
    clusters = []
    for x in X:
        distances = [euclidean_distance(x, centroid) for centroid in centroids]
        cluster = np.argmin(distances)  # Index of the closest centroid
        clusters.append(cluster)
    return np.array(clusters)

def update_centroids(X, clusters, k):
    """Recalculate centroids as the mean of points in each cluster."""
    new_centroids = []
    for i in range(k):
        cluster_points = X[clusters == i]
        if len(cluster_points) > 0:
            new_centroids.append(np.mean(cluster_points, axis=0))
        else:  # Handle empty cluster case
            new_centroids.append(np.zeros(X.shape[1]))
    return np.array(new_centroids)

def kmeans_from_scratch(X, k, max_iters=100, tol=1e-4):
    """K-Means clustering algorithm implemented from scratch."""
    centroids = initialize_centroids(X, k)
    for _ in range(max_iters):
        clusters = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, clusters, k)
        if np.all(np.abs(new_centroids - centroids) < tol):
            break
        centroids = new_centroids
    return clusters, centroids

def perform_clustering(df, n_clusters=N_CLUSTERS):
    """
    Perform K-Means clustering on the podcast dataset.
    """
    print(f"Performing K-Means clustering with {n_clusters} clusters...")
    X = prepare_clustering_features(df)
    clusters, centroids = kmeans_from_scratch(X, k=n_clusters)
    df['cluster'] = clusters
    return df, centroids

def recommend_from_cluster(df, podcast_name, n_recommendations=5):
    """
    Recommend podcasts from the same cluster as the input podcast.
    """
    if podcast_name not in df['name'].values:
        raise ValueError(f"Podcast '{podcast_name}' not found in the dataset.")

    podcast_cluster = df[df['name'] == podcast_name]['cluster'].iloc[0]
    print(f"Podcast '{podcast_name}' belongs to cluster {podcast_cluster}.")

    cluster_podcasts = df[df['cluster'] == podcast_cluster]
    cluster_podcasts = cluster_podcasts[cluster_podcasts['name'] != podcast_name]
    recommendations = cluster_podcasts.sort_values(by='total_episodes_scaled', ascending=False)
    return recommendations[['name', 'description', 'genre', 'publisher']].head(n_recommendations)

if __name__ == "__main__":
    print("Loading preprocessed data...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"Dataset loaded with {len(df)} rows.")

    # Perform clustering with the best number of clusters
    df, centroids = perform_clustering(df)

    # Save clustered data
    os.makedirs(CLUSTER_OUTPUT_PATH, exist_ok=True)
    df.to_csv(CLUSTERED_DATA_PATH, index=False)
    print(f"Clustered data saved to {CLUSTERED_DATA_PATH}")

    # Evaluate clustering performance
    print("\nEvaluating Clustering Performance...")
    X = prepare_clustering_features(df)
    silhouette = silhouette_score(X, df['cluster'])
    print(f"Silhouette Score: {silhouette:.4f}")

    # Example: Recommend podcasts
    podcast_name = "TED Tech"
    print(f"\nRecommendations for podcast '{podcast_name}':")
    try:
        recommendations = recommend_from_cluster(df, podcast_name, n_recommendations=5)
        print(recommendations)
    except ValueError as e:
        print(e)
