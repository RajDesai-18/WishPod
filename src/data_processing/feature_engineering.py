from src.utils.common_utils import compute_similarity


def prepare_clustering_features(df):
    """
    Prepare the dataset for clustering.
    Selects features relevant to the clustering algorithm.
    """
    features = ['genre_encoded', 'language_encoded', 'total_episodes_scaled']
    return df[features].values


def prepare_knn_dataset(df, input_podcast, negative_sample_ratio=0.5):
    """
    Prepare the dataset for kNN training, preserving the 'name' column.
    """
    if input_podcast not in df['name'].values:
        raise ValueError(f"Podcast '{input_podcast}' not found in the dataset.")

    # Identify the input podcast row
    input_row = df[df['name'] == input_podcast].iloc[0]

    # Compute similarity for all rows with the input podcast
    df['similarity'] = df.apply(lambda row: compute_similarity(row, input_row), axis=1)

    # Generate the target column ('is_recommended') based on similarity
    df['is_recommended'] = (df['similarity'] > df['similarity'].quantile(1 - negative_sample_ratio)).astype(int)

    # Retain the 'name' column while dropping unnecessary columns
    columns_to_keep = ['name', 'total_episodes', 'explicit', 'popularity',
                       'total_episodes_scaled', 'genre_encoded', 'language_encoded',
                       'is_recommended']
    df = df[columns_to_keep]
    df = df.dropna()

    return df


def prepare_naive_bayes_dataset(df, input_podcast, negative_sample_ratio=0.5):
    """
    Prepare the dataset for Naive Bayes training.
    """
    if input_podcast not in df['name'].values:
        raise ValueError(f"Podcast '{input_podcast}' not found in the dataset.")

    input_row = df[df['name'] == input_podcast].iloc[0]

    df['similarity'] = df.apply(lambda row: compute_similarity(row, input_row), axis=1)

    df['is_recommended'] = (df['similarity'] > df['similarity'].quantile(1 - negative_sample_ratio)).astype(int)

    columns_to_drop = ['id', 'description', 'publisher', 'external_url', 'images', 'image_url',
                       'language', 'genre', 'similarity']
    df = df.drop(columns=columns_to_drop, errors='ignore')
    df = df.dropna()

    return df
