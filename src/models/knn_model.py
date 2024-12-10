import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from src.evaluation.metrics import evaluate_model_with_auc, plot_confusion_matrix
from src.utils.common_utils import euclidean_distance
from src.data_processing.feature_engineering import prepare_knn_dataset

# Paths
PROCESSED_DATA_PATH = "D:/Classes/Machine Learning/Project/Wishpod/data/processed/preprocessed_podcasts.csv"

# Feature and Target Columns
FEATURE_COLUMNS = ['total_episodes', 'explicit', 'popularity',
                   'total_episodes_scaled', 'genre_encoded', 'language_encoded']
TARGET_COLUMN = 'is_recommended'

# Best k value from tuning
K = 3

def knn_predict(X_train, y_train, test_point, k=K):
    """Predict the class of a single test point using k-Nearest Neighbors."""
    distances = [(euclidean_distance(test_point, train_point), y_train[i]) for i, train_point in enumerate(X_train)]
    distances = sorted(distances, key=lambda x: x[0])
    k_neighbor_classes = [neighbor[1] for neighbor in distances[:k]]
    most_common = Counter(k_neighbor_classes).most_common(1)
    return most_common[0][0]

def knn_predict_top(X_train, y_train, X_test, k=K, df=None, top_n=5):
    """
    Predict the classes of all test points using k-Nearest Neighbors
    and return the top N recommended podcasts.
    """
    predictions = [knn_predict(X_train, y_train, test_point, k) for test_point in X_test]
    recommendations = [i for i, pred in enumerate(predictions) if pred == 1]

    recommended_podcasts = [
        df.iloc[idx]['name'] if 'name' in df.columns else 'Unknown Podcast'
        for idx in recommendations[:top_n]
    ]
    return predictions, recommended_podcasts

if __name__ == "__main__":
    # Load the processed dataset
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # Define the podcast name
    podcast_name = "The Joe Rogan Experience"
    print(f"Preparing dataset for podcast: {podcast_name}")

    # Prepare the dataset
    prepared_df = prepare_knn_dataset(df, input_podcast=podcast_name)

    X = prepared_df[FEATURE_COLUMNS].values
    y = prepared_df[TARGET_COLUMN].values

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Generate predictions
    y_pred, recommended_podcasts = knn_predict_top(X_train, y_train, X_test, K, df)

    # Evaluate the model
    print("\nEvaluating Model Performance...")
    knn_metrics = evaluate_model_with_auc(y_test, y_pred)
    print("\nEvaluation Metrics:")
    print(knn_metrics)

    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred)

    # Print recommended podcasts
    print("\nRecommended Podcasts:")
    for podcast in recommended_podcasts:
        print(f"- {podcast}")
