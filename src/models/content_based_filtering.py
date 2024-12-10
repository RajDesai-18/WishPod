import pandas as pd
from src.utils.common_utils import compute_similarity_cbf

# Paths
PROCESSED_DATA_PATH = "D:/Classes/Machine Learning/Project/Wishpod/data/processed/preprocessed_podcasts.csv"

# Define the best weights globally
WEIGHTS = {'genre': 0.5, 'language': 0.3, 'episodes': 0.2}

def content_based_recommendation_knn(df, podcast_name, top_n=5, weights=None):
    """
    Recommend podcasts based on similarity to a given podcast.
    """
    # Use the best weights if none are provided
    if weights is None:
        weights = WEIGHTS

    # Ensure input podcast exists
    if podcast_name not in df['name'].values:
        raise ValueError(f"Podcast '{podcast_name}' not found in the dataset.")

    # Extract target podcast features
    target_podcast = df[df['name'] == podcast_name].iloc[0]

    # Compute similarity for all rows using the utility function
    df['similarity'] = df.apply(
        lambda row: compute_similarity_cbf(row, target_podcast, weights), axis=1
    )

    # Sort by similarity and exclude the input podcast
    recommendations = df[df['name'] != podcast_name].sort_values(by='similarity', ascending=False).head(top_n)

    return recommendations[['name', 'similarity']]


if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # Define the podcast name for recommendations
    podcast_name = "The Joe Rogan Experience"

    # Generate recommendations using the best weights
    print(f"\nGenerating recommendations for podcast '{podcast_name}'...")
    recommendations = content_based_recommendation_knn(df, podcast_name, top_n=5)
    print("\nRecommendations:")
    print(recommendations)
