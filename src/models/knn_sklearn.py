# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import classification_report
# from src.models.content_based_filtering import compute_similarity
#
# # Paths
# PROCESSED_DATA_PATH = "D:/Classes/Machine Learning/Project/Wishpod/data/processed/preprocessed_podcasts.csv"
#
# # Define the feature columns used for training and prediction
# FEATURE_COLUMNS = ['total_episodes', 'explicit', 'popularity',
#                    'total_episodes_scaled', 'genre_encoded', 'language_encoded']
#
# def prepare_knn_dataset(df, input_podcast, negative_sample_ratio=0.5):
#     """
#     Prepare the dataset for kNN training.
#     """
#     # Ensure the input podcast exists
#     if input_podcast not in df['name'].values:
#         raise ValueError(f"Podcast '{input_podcast}' not found in the dataset.")
#
#     # Get the input podcast row
#     input_row = df[df['name'] == input_podcast].iloc[0]
#
#     # Compute similarity scores for all podcasts
#     df['similarity'] = df.apply(lambda row: compute_similarity(row, input_row), axis=1)
#
#     # Label the top similar podcasts as recommended
#     df['is_recommended'] = (df['similarity'] > df['similarity'].quantile(1 - negative_sample_ratio)).astype(int)
#
#     # Drop unnecessary columns (but keep 'name' for reference)
#     columns_to_drop = ['id', 'description', 'publisher', 'external_url', 'images', 'image_url',
#                        'language', 'genre', 'similarity']
#     df = df.drop(columns=columns_to_drop, errors='ignore')
#
#     # Remove rows with NaN or invalid values
#     df = df.dropna()
#
#     # Debug: Check the dataset size and contents
#     print(f"Dataset size after processing: {df.shape}")
#     print(df.head())
#
#     return df
#
#
# def train_knn_model(df, k=5):
#     """
#     Train a k-Nearest Neighbors (kNN) model.
#     """
#     # Split dataset into features and target
#     X = df[FEATURE_COLUMNS]
#     y = df['is_recommended']
#
#     # Debug: Print features being used
#     print("Features used for training:")
#     print(X.dtypes)
#
#     # Split into train and test sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     # Train kNN model
#     model = KNeighborsClassifier(n_neighbors=k)
#     model.fit(X_train, y_train)
#
#     # Evaluate the model
#     y_pred = model.predict(X_test)
#     print("\nModel Evaluation:")
#     print(classification_report(y_test, y_pred))
#
#     return model
#
#
# def predict_knn_recommendations(df, model, top_n=5):
#     """
#     Predict top N podcast recommendations using the trained kNN model.
#     """
#     # Predict probabilities
#     df['probability'] = model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
#
#     # Filter and sort by probability
#     df['predicted'] = model.predict(df[FEATURE_COLUMNS])
#     recommendations = df[df['predicted'] == 1].sort_values(by='probability', ascending=False).head(top_n)
#
#     # Print recommended podcast names
#     print("\nRecommended Podcasts:")
#     for idx, row in recommendations.iterrows():
#         print(f"- {row['name']}")
#
#     return recommendations
#
# if __name__ == "__main__":
#     # Load processed dataset
#     df = pd.read_csv(PROCESSED_DATA_PATH)
#
#     # Input podcast
#     podcast_name = "The Joe Rogan Experience"
#     print(f"Preparing dataset for podcast: {podcast_name}")
#
#     # Prepare dataset
#     prepared_df = prepare_knn_dataset(df, input_podcast=podcast_name)
#
#     # Train kNN
#     model = train_knn_model(prepared_df, k=5)
#
#     # Predict recommendations
#     print(f"\nRecommendations for podcast '{podcast_name}':")
#     recommendations = predict_knn_recommendations(prepared_df, model, top_n=5)
#     print(recommendations)
