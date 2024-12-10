import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Paths
RAW_DATA_PATH = "D:/Classes/Machine Learning/Project/Wishpod/data/raw/podcasts.csv"
PROCESSED_DATA_PATH = "D:/Classes/Machine Learning/Project/Wishpod/data/processed/preprocessed_podcasts.csv"

def preprocess_podcast_data(raw_data_path, processed_data_path):
    """
    Preprocess the podcast dataset:
    - Remove duplicates
    - Handle missing values
    - Normalize numerical features
    - Encode categorical features
    - Save the preprocessed data
    """
    # Load the raw data
    print("Loading raw data...")
    df = pd.read_csv(raw_data_path)
    print(f"Raw dataset loaded with {len(df)} rows.")

    # Remove duplicates
    print("\nRemoving duplicates...")
    duplicates = df.duplicated(subset='id').sum()
    print(f"Number of duplicate rows: {duplicates}")
    df = df.drop_duplicates(subset='id')
    print(f"Dataset after removing duplicates: {len(df)} rows.")

    # Inspect and handle missing values
    print("\nHandling missing values...")
    print("Missing values before:\n", df.isnull().sum())
    df['language'] = df['language'].fillna('unknown')  # Fill missing languages
    df['description'] = df['description'].fillna('No description provided')  # Fill missing descriptions
    print("Missing values after:\n", df.isnull().sum())

    # Normalize numerical features
    print("\nNormalizing numerical features...")
    scaler = MinMaxScaler()
    df['total_episodes_scaled'] = scaler.fit_transform(df[['total_episodes']])
    print("Numerical features normalized.")

    # Encode categorical features
    print("\nEncoding categorical features...")
    encoder = LabelEncoder()
    df['genre_encoded'] = encoder.fit_transform(df['genre'])
    df['language_encoded'] = encoder.fit_transform(df['language'])
    print("Categorical features encoded.")

    # Save the preprocessed dataset
    print("\nSaving preprocessed data...")
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    df.to_csv(processed_data_path, index=False)
    print(f"Preprocessed data saved to {processed_data_path}")

    return df


if __name__ == "__main__":
    # Preprocess the data
    print("Starting preprocessing...")
    processed_df = preprocess_podcast_data(RAW_DATA_PATH, PROCESSED_DATA_PATH)
    print("\nPreprocessing complete. Here's a preview of the preprocessed data:")
    print(processed_df.head())
