import os
import sys
import streamlit as st
import pandas as pd

# Add the parent directory to sys.path for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import modules from src
from src.data_processing.feature_engineering import prepare_knn_dataset, prepare_naive_bayes_dataset
from src.models.knn_model import knn_predict_top
from src.models.clustering import perform_clustering, recommend_from_cluster
from src.models.naive_bayes import train_naive_bayes_model, predict_naive_bayes_recommendations
from src.models.content_based_filtering import content_based_recommendation_knn

# Paths
PROCESSED_DATA_PATH = "D:/Classes/Machine Learning/Project/Wishpod/data/processed/preprocessed_podcasts.csv"

# Define the Streamlit app
def main():
    st.set_page_config(page_title="Wishpod - Podcast Recommendations", layout="wide")

    # Apply custom font and styles
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Raleway:wght@300;400;600;700&display=swap');
            body {
                font-family: 'Raleway', sans-serif;
                color: #F8F8F2;
                background-color: #121212;
            }
            h1, h4 {
                text-align: center;
                color: #61bbff;
                font-family: 'Raleway', sans-serif;
            }
            .stTextInput > div > input {
                background-color: #121212;
                color: #61bbff;
                font-size: 25px;
                border: 1px solid #61bbff;
                border-radius: 5px;
            }
            .stButton button {
                background-color: #61bbff;
                color: #FFFFFF;
                border-radius: 5px;
                font-size: 14px;
                border: none;
            }
            .stButton button:hover {
                color: #FFFFFF;
                background-color: #61bbff;
            }
            .output-container {
                color: #61bbff;
                text-align: center;
                margin-top: 20px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header
    st.markdown("<h1>🎧 Wishpod</h1>", unsafe_allow_html=True)
    st.markdown("<h4>Your Personalized Podcast Recommendation Engine</h4>", unsafe_allow_html=True)

    # Input Section
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            podcast_name = st.text_input("Enter Podcast Name:", placeholder="e.g., The Joe Rogan Experience")
            st.markdown("<h6>Select Models:</h6>", unsafe_allow_html=True)
            content_based = st.checkbox("Content-Based")
            clustering_based = st.checkbox("Clustering-Based")
            knn_based = st.checkbox("kNN")
            naive_bayes = st.checkbox("Naive Bayes")
            top_n = st.slider("Number of Recommendations:", 1, 10, 5)
            submit_button = st.button("Submit")

    # Recommendation Results
    if submit_button:
        st.subheader("Recommendations")
        st.markdown("<div class='output-container'>", unsafe_allow_html=True)

        # Load dataset
        @st.cache_data
        def load_data():
            return pd.read_csv(PROCESSED_DATA_PATH)

        df = load_data()

        # Validate Podcast Name
        if podcast_name not in df['name'].values:
            st.error(f"Podcast '{podcast_name}' not found in the dataset!")
            return

        # Content-Based Recommendations
        if content_based:
            st.subheader("- Content-Based Recommendations:")
            content_recs = content_based_recommendation_knn(df, podcast_name, top_n=top_n)
            for idx, row in content_recs.iterrows():
                st.write(f"{idx + 1}. {row['name']} (Similarity: {row['similarity']:.2f})")

        # Clustering-Based Recommendations
        if clustering_based:
            st.subheader("- Clustering-Based Recommendations:")
            df, _ = perform_clustering(df, n_clusters=10)
            cluster_recs = recommend_from_cluster(df, podcast_name, n_recommendations=top_n)
            for idx, row in cluster_recs.iterrows():
                st.write(f"{idx + 1}. {row['name']} (Cluster: {row['genre']})")

        # kNN Recommendations
        if knn_based:
            st.subheader("- kNN Recommendations:")
            prepared_df = prepare_knn_dataset(df, input_podcast=podcast_name)
            X = prepared_df[['total_episodes', 'explicit', 'popularity', 'total_episodes_scaled', 'genre_encoded', 'language_encoded']].values
            y = prepared_df['is_recommended'].values
            _, knn_recs = knn_predict_top(X, y, X, k=3, df=prepared_df, top_n=top_n)
            for idx, rec in enumerate(knn_recs):
                st.write(f"{idx + 1}. {rec}")

        # Naive Bayes Recommendations
        if naive_bayes:
            st.subheader("- Naive Bayes Recommendations:")
            prepared_df = prepare_naive_bayes_dataset(df, input_podcast=podcast_name)
            model = train_naive_bayes_model(prepared_df)
            nb_recs = predict_naive_bayes_recommendations(prepared_df, model, top_n=top_n)
            for idx, row in nb_recs.iterrows():
                st.write(f"{idx + 1}. {row['name']} (Probability: {row['probability']:.2f})")

        st.markdown("</div>", unsafe_allow_html=True)

    # Footer
    st.markdown(
        """
        <p style='text-align: center;'>
        Made by Raj Desai <br>
        Developed for the Machine Learning Course, Fall 2024
        </p>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
