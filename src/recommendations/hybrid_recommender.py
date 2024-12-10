import pandas as pd
from src.data_processing.feature_engineering import prepare_knn_dataset
from src.models.clustering import perform_clustering, recommend_from_cluster
from src.models.content_based_filtering import content_based_recommendation_knn
from src.models.knn_model import knn_predict_top
from src.models.naive_bayes import predict_naive_bayes_recommendations, train_naive_bayes_model
from src.evaluation.metrics import evaluate_model_with_auc

# Paths
PROCESSED_DATA_PATH = "D:/Classes/Machine Learning/Project/Wishpod/data/processed/preprocessed_podcasts.csv"

def hybrid_recommendations(podcast_name, df, k=3, top_n=5):
    """
    Generate hybrid recommendations with dynamically adjusted weights and evaluate them.
    """
    print("=" * 50)
    print(f"Hybrid Recommendations for Podcast: '{podcast_name}'")
    print("=" * 50)

    # Perform clustering if not already done
    if 'cluster' not in df.columns:
        print("\n[INFO] Performing Clustering...")
        df, _ = perform_clustering(df, n_clusters=10)

    # Content-Based Recommendations
    print("\n[INFO] Fetching Content-Based Recommendations...")
    content_recommendations = content_based_recommendation_knn(df, podcast_name, top_n=top_n)
    content_scores = {name: top_n - idx for idx, name in enumerate(content_recommendations['name'])}

    # Clustering Recommendations
    print("\n[INFO] Fetching Clustering-Based Recommendations...")
    cluster_recommendations = recommend_from_cluster(df, podcast_name, n_recommendations=top_n)
    cluster_scores = {name: top_n - idx for idx, name in enumerate(cluster_recommendations['name'])}

    # Prepare dataset for kNN
    print("\n[INFO] Preparing Dataset for kNN...")
    df_knn = prepare_knn_dataset(df.copy(), input_podcast=podcast_name)

    # kNN Recommendations
    print("\n[INFO] Fetching kNN-Based Recommendations...")
    X = df_knn[['total_episodes', 'explicit', 'popularity', 'total_episodes_scaled', 'genre_encoded', 'language_encoded']]
    y = df_knn['is_recommended']
    _, knn_recommendations = knn_predict_top(X.values, y.values, X.values, k, df_knn, top_n=top_n)
    knn_scores = {name: top_n - idx for idx, name in enumerate(knn_recommendations)}

    # Naive Bayes Recommendations
    print("\n[INFO] Training Naive Bayes Model...")
    naive_bayes_model = train_naive_bayes_model(df_knn)
    print("\n[INFO] Fetching Naive Bayes-Based Recommendations...")
    naive_bayes_recommendations = predict_naive_bayes_recommendations(df_knn, naive_bayes_model, top_n=top_n)
    nb_scores = {row['name']: top_n - idx for idx, row in naive_bayes_recommendations.iterrows()}

    # Combine Scores
    combined_scores = {}
    for name in set(content_scores.keys()).union(cluster_scores.keys(), knn_scores.keys(), nb_scores.keys()):
        combined_scores[name] = (
            0.4 * content_scores.get(name, 0) +
            0.3 * cluster_scores.get(name, 0) +
            0.2 * knn_scores.get(name, 0) +
            0.1 * nb_scores.get(name, 0)
        )

    # Sort Recommendations
    final_recommendations = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # Evaluate Hybrid Recommendations
    y_true = y.values
    y_pred = [1 if name in dict(final_recommendations) else 0 for name in df_knn['name']]
    metrics = evaluate_model_with_auc(y_true, y_pred)

    print("\nFinal Recommendations:")
    for idx, (name, score) in enumerate(final_recommendations, start=1):
        print(f"{idx}. {name:<40} | Score: {score:.2f}")

    print("\nHybrid Recommender Evaluation Metrics:")
    print(metrics)

    return final_recommendations


if __name__ == "__main__":
    # Load the processed dataset
    try:
        print("Loading preprocessed data...")
        df = pd.read_csv(PROCESSED_DATA_PATH)
        print(f"Dataset loaded with {len(df)} rows.")

        # Input podcast name
        podcast_name = "The Joe Rogan Experience"

        # Generate hybrid recommendations
        hybrid_recommendations(podcast_name, df, k=3, top_n=5)

    except Exception as e:
        print(f"An error occurred: {e}")
