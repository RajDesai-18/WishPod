import pandas as pd
from src.data_processing.feature_engineering import prepare_knn_dataset, prepare_naive_bayes_dataset
from src.models.content_based_filtering import content_based_recommendation_knn
from src.models.clustering import perform_clustering, recommend_from_cluster
from src.models.knn_model import knn_predict_top
from src.models.naive_bayes import train_naive_bayes_model, predict_naive_bayes_recommendations

# Paths
PROCESSED_DATA_PATH = "D:/Classes/Machine Learning/Project/Wishpod/data/processed/preprocessed_podcasts.csv"


def main():
    """
    Single entry point for the Wishpod project without Hybrid Recommender.
    """
    print("==================================================")
    print("   Welcome to Wishpod Podcast Recommendation System!")
    print("==================================================")

    # Load dataset
    print("\n[INFO] Loading preprocessed data...")
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
        print(f"[INFO] Dataset loaded with {len(df)} rows.")
    except FileNotFoundError:
        print("[ERROR] Preprocessed dataset not found. Please check the file path.")
        return

    # Input podcast for recommendations
    podcast_name = input("\nEnter a podcast name for recommendations: ").strip()
    if podcast_name not in df['name'].values:
        print(f"[ERROR] Podcast '{podcast_name}' not found in the dataset.")
        return

    # Content-Based Recommendations
    print("\n[INFO] Running Content-Based Filtering...")
    try:
        content_recommendations = content_based_recommendation_knn(df, podcast_name, top_n=5)
        print("\nContent-Based Recommendations:")
        for idx, row in content_recommendations.iterrows():
            print(f"{idx + 1}. {row['name']} (Similarity: {row['similarity']:.2f})")
    except ValueError as e:
        print(f"[ERROR] {e}")

    # Clustering-Based Recommendations
    print("\n[INFO] Running Clustering...")
    try:
        df, _ = perform_clustering(df, n_clusters=10)
        clustering_recommendations = recommend_from_cluster(df, podcast_name, n_recommendations=5)
        print("\nClustering-Based Recommendations:")
        for idx, row in clustering_recommendations.iterrows():
            print(f"{idx + 1}. {row['name']} (Genre: {row['genre']}, Publisher: {row['publisher']})")
    except ValueError as e:
        print(f"[ERROR] {e}")

    # kNN Recommendations
    print("\n[INFO] Running k-Nearest Neighbors (kNN)...")
    prepared_knn_df = prepare_knn_dataset(df.copy(), input_podcast=podcast_name)
    knn_X = prepared_knn_df[['total_episodes', 'explicit', 'popularity', 'total_episodes_scaled', 'genre_encoded', 'language_encoded']].values
    knn_y = prepared_knn_df['is_recommended'].values
    _, knn_recommendations = knn_predict_top(knn_X, knn_y, knn_X, k=3, df=prepared_knn_df, top_n=5)
    print("\nKNN-Based Recommendations:")
    for idx, name in enumerate(knn_recommendations, start=1):
        print(f"{idx}. {name}")

    # Naive Bayes Recommendations
    print("\n[INFO] Running Naive Bayes...")
    prepared_nb_df = prepare_naive_bayes_dataset(df, input_podcast=podcast_name)
    naive_bayes_model = train_naive_bayes_model(prepared_nb_df)
    nb_recommendations = predict_naive_bayes_recommendations(prepared_nb_df, naive_bayes_model, top_n=5)
    print("\nNaive Bayes-Based Recommendations:")
    for idx, row in nb_recommendations.iterrows():
        print(f"{idx + 1}. {row['name']} (Probability: {row['probability']:.2f})")

    print("\n[INFO] Recommendation process completed. Thank you for using Wishpod!")


if __name__ == "__main__":
    main()
