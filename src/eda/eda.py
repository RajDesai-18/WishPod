import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Paths
PROCESSED_DATA_PATH = "D:/Classes/Machine Learning/Project/Wishpod/data/processed/preprocessed_podcasts.csv"
EDA_OUTPUT_PATH = "D:/Classes/Machine Learning/Project/Wishpod/data/eda/"

def perform_eda(processed_data_path, eda_output_path):
    """
    Perform Exploratory Data Analysis (EDA) on the podcast dataset.
    - Generate visualizations and insights about the dataset.
    """
    # Load preprocessed data
    print("Loading preprocessed data...")
    df = pd.read_csv(processed_data_path)
    print(f"Dataset loaded with {len(df)} rows and {len(df.columns)} columns.")

    # Ensure output directory exists
    os.makedirs(eda_output_path, exist_ok=True)

    # 1. Basic statistics
    print("\nBasic statistics:")
    print(df.describe(include='all'))

    # 2. Genre distribution
    print("\nVisualizing genre distribution...")
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='genre', order=df['genre'].value_counts().index)
    plt.title("Podcast Genre Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(eda_output_path, "genre_distribution.png"))
    plt.show()

    # 3. Language distribution
    print("\nVisualizing language distribution...")
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='language', order=df['language'].value_counts().index)
    plt.title("Podcast Language Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(eda_output_path, "language_distribution.png"))
    plt.show()

    # 4. Episode count distribution
    print("\nVisualizing episode count distribution...")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['total_episodes'], bins=30, kde=True, color="blue")
    plt.title("Total Episodes Distribution")
    plt.xlabel("Total Episodes")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(eda_output_path, "episode_count_distribution.png"))
    plt.show()

    # 5. Correlation heatmap
    print("\nGenerating correlation heatmap...")
    plt.figure(figsize=(10, 8))
    numeric_columns = df.select_dtypes(include=['number'])
    correlation = numeric_columns.corr()
    sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(eda_output_path, "correlation_heatmap.png"))
    plt.show()

    # 6. Word cloud for descriptions
    print("\nGenerating word cloud for descriptions...")
    text = " ".join(df['description'].dropna().astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Word Cloud for Podcast Descriptions")
    plt.tight_layout()
    plt.savefig(os.path.join(eda_output_path, "word_cloud.png"))
    plt.show()

    print("\nEDA complete. Visualizations saved to:", eda_output_path)

if __name__ == "__main__":
    print("Starting EDA...")
    perform_eda(PROCESSED_DATA_PATH, EDA_OUTPUT_PATH)
    print("EDA completed.")
