import numpy as np


def euclidean_distance(a, b):
    """Compute the Euclidean distance between two points."""
    return np.sqrt(np.sum((a - b) ** 2))


def initialize_centroids(X, k, random_state=42):
    """Randomly initialize centroids from the dataset."""
    np.random.seed(random_state)
    random_indices = np.random.choice(len(X), size=k, replace=False)
    return X[random_indices]

def compute_hamming_distance(a, b):
    """Compute Hamming distance between two categorical features."""
    return 0 if a == b else 1


def compute_euclidean_distance(a, b):
    """Compute Euclidean distance between two numerical features."""
    return np.abs(a - b)


def compute_similarity(row, input_row, weights=None):
    """
    Compute similarity between a given podcast and an input podcast.
    Combine distances across multiple features.
    """
    if weights is None:
        weights = {'genre': 0.5, 'language': 0.3, 'episodes': 0.2}

    genre_similarity = 1 - compute_hamming_distance(row['genre_encoded'], input_row['genre_encoded'])
    language_similarity = 1 - compute_hamming_distance(row['language_encoded'], input_row['language_encoded'])
    episode_similarity = 1 - compute_euclidean_distance(row['total_episodes_scaled'],
                                                        input_row['total_episodes_scaled'])

    total_similarity = (weights['genre'] * genre_similarity +
                        weights['language'] * language_similarity +
                        weights['episodes'] * episode_similarity)
    return total_similarity

def compute_similarity_cbf(row, target_podcast, weights):
    """
    Compute similarity between a podcast row and the target podcast.
    """
    genre_similarity = 1 if row['genre_encoded'] == target_podcast['genre_encoded'] else 0
    language_similarity = 1 if row['language_encoded'] == target_podcast['language_encoded'] else 0
    episode_similarity = 1 - abs(row['total_episodes_scaled'] - target_podcast['total_episodes_scaled'])

    # Weighted similarity
    return (
        weights['genre'] * genre_similarity +
        weights['language'] * language_similarity +
        weights['episodes'] * episode_similarity
    )

