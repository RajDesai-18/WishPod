# 🎧 Wishpod: Personalized Podcast Recommendation Engine

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)  
A machine learning-powered podcast recommendation system designed to provide personalized podcast suggestions based on user preferences and listening habits.

---

## 📖 Table of Contents

1. [Overview](#-overview)  
2. [Features](#-features)  
3. [Tech Stack](#-tech-stack)  
4. [Dataset](#-dataset)  
5. [Models Implemented](#-models-implemented)  
6. [Project Structure](#-project-structure)  
7. [How to Run](#-how-to-run)
8. [Future Enhancements](#-future-enhancements) 
9. [License](#-license)  

---

## 🔍 Overview

Wishpod is a podcast recommendation engine that uses machine learning algorithms to generate tailored podcast suggestions for users. It offers recommendations using various approaches, including content-based filtering, clustering, kNN, and Naive Bayes classification. Users can interact with the system via a modern and minimalistic web UI powered by Streamlit.

This project is developed as part of the **Machine Learning Course Project** by **Raj Desai**.

---

## ✨ Features

- **Multi-Model Support**: Users can choose from content-based, clustering, kNN, or Naive Bayes models for recommendations.
- **Dynamic Weights**: Combines multiple recommendation models using dynamic accuracy-based weighting for hybrid recommendations.
- **Interactive UI**: A sleek, modern UI for selecting input podcast and models.
- **Efficient Computation**: Implements k-means clustering and kNN from scratch for better customization and understanding.
- **Real-Time Results**: Provides recommendations and evaluates model performance with metrics like accuracy, precision, recall, and ROC-AUC.

---

## 🛠 Tech Stack

- **Programming Language**: Python 3.10
- **Web Framework**: Streamlit
- **Libraries**:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib

---

## 📊 Dataset

The system uses a preprocessed dataset containing podcast metadata, including features like:
- Podcast Name
- Genre
- Language
- Popularity
- Number of Episodes
- Explicit Content Indicator

**Path**: `data/processed/preprocessed_podcasts.csv`

---

## 🧠 Models Implemented

### 1. **Content-Based Filtering**:
   - Recommends podcasts based on similarity of metadata (e.g., genre, language, popularity).
   - Uses a weighted similarity computation.

### 2. **Clustering**:
   - Groups podcasts into clusters using k-means clustering.
   - Recommends podcasts from the same cluster as the selected podcast.

### 3. **k-Nearest Neighbors (kNN)**:
   - Predicts podcast recommendations based on proximity in feature space.

### 4. **Naive Bayes**:
   - Trains a probabilistic model to predict recommended podcasts.

---

## 📂 Project Structure

```
Wishpod/
├── data/
│   ├── clustering/
│   ├── processed/
│   ├── raw/
│   └── results/
├── notebooks/
├── src/
│   ├── data_processing/
│   ├── eda/
│   ├── evaluation/
│   ├── models/
│   ├── recommendations/
│   └── utils/
├── UI/
│   └── app.py
├── main.py
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### Prerequisites:
- Python 3.10+
- pip (Python package manager)

### Setup Instructions:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-github-username/wishpod.git
   cd wishpod
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv wishpod-env
   source wishpod-env/bin/activate  # On Windows: wishpod-env\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit App**:
   ```bash
   streamlit run UI/app.py
   ```

5. Open the app in your browser: `http://localhost:8501`.

---

## 🛠 Future Enhancements

- Add more models like collaborative filtering.
- Integrate real-time podcast data using APIs.
- Improve clustering model by experimenting with DBSCAN or hierarchical clustering.
- Add a user feedback mechanism for personalized recommendations.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
