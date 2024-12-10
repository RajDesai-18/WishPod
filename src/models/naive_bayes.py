import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from src.data_processing.feature_engineering import prepare_naive_bayes_dataset
from src.evaluation.metrics import evaluate_model_with_auc, plot_confusion_matrix

# Paths
PROCESSED_DATA_PATH = "D:/Classes/Machine Learning/Project/Wishpod/data/processed/preprocessed_podcasts.csv"
FEATURE_COLUMNS = ['total_episodes', 'explicit', 'popularity',
                   'total_episodes_scaled', 'genre_encoded', 'language_encoded']
TARGET_COLUMN = 'is_recommended'

# Naive Bayes Training
def train_naive_bayes_model(df):
    """
    Train a Naive Bayes model to predict podcast recommendations.
    """
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GaussianNB()
    model.fit(X_train, y_train)

    # Evaluate the model
    print("\nEvaluating Model Performance...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of being recommended
    metrics = evaluate_model_with_auc(y_test, y_pred, y_pred_proba)
    print("\nEvaluation Metrics:")
    print(metrics)

    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred)

    return model


def predict_naive_bayes_recommendations(df, model, top_n=5):
    """
    Predict top N podcast recommendations using the trained Naive Bayes model.
    """
    df['probability'] = model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
    df['predicted'] = model.predict(df[FEATURE_COLUMNS])
    recommendations = df[df['predicted'] == 1].sort_values(by='probability', ascending=False).head(top_n)

    # print("\nRecommended Podcasts:")
    # for idx, row in recommendations.iterrows():
    #     print(f"- {row['name']}")

    return recommendations


if __name__ == "__main__":
    # Load the processed dataset
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # Define the podcast for which recommendations will be generated
    podcast_name = "TED Tech"
    print(f"Preparing dataset for podcast: {podcast_name}")
    prepared_df = prepare_naive_bayes_dataset(df, input_podcast=podcast_name)

    # Train the Naive Bayes model
    model = train_naive_bayes_model(prepared_df)

    # Generate recommendations
    print(f"\nRecommendations for podcast '{podcast_name}':")
    predict_naive_bayes_recommendations(prepared_df, model, top_n=5)
