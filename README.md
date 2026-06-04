# 🎬 Hybrid Movie Recommendation System

A machine learning-based movie recommendation system that combines content-based filtering techniques to suggest relevant movies based on user preferences and movie characteristics.

## 🚀 Features

* Content-Based Movie Recommendation using TF-IDF Vectorization
* Cosine Similarity-based Movie Matching
* Personalized User Recommendations
* Genre-Based Recommendations
* Exploratory Data Analysis (EDA)
* Recommendation Similarity Analysis
* Automated Report Generation
* Modular Code Structure for Future Hybrid Extensions

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn

## 📂 Project Structure

```text
Hybrid_Movie_Recommendation_System/
│
├── main.py
├── recommender.py
├── collaborative_filtering.py
├── svd_recommender.py
├── evaluation_metrics.py
├── streamlit_app.py
│
├── movies.csv
├── ratings.csv
│
├── eda_dashboard.png
├── similarity_analysis.png
├── recommendation_report.txt
│
├── requirements.txt
└── README.md
```

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/<your-username>/hybrid-movie-recommender.git
cd hybrid-movie-recommender
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Running the Project

Execute:

```bash
python main.py
```

The system will:

* Load movie and ratings datasets
* Build recommendation models
* Generate recommendations
* Create analysis visualizations
* Generate recommendation reports

## 📊 Methodology

### Content-Based Filtering

The recommendation engine uses:

1. Movie genre extraction
2. TF-IDF vectorization
3. Cosine similarity computation
4. Similar movie ranking

### Recommendation Pipeline

```text
Movie Metadata
      ↓
TF-IDF Vectorization
      ↓
Cosine Similarity Matrix
      ↓
Top-N Similar Movies
      ↓
Personalized Recommendations
```

## 📈 Outputs

### EDA Dashboard

Provides insights into:

* Rating distributions
* Popular genres
* User activity trends
* Movie frequency analysis

### Similarity Analysis

Visualizes:

* Movie similarity distributions
* Recommendation strength
* Feature relationships

### Recommendation Report

Generates a text report containing:

* Similar movie recommendations
* Genre recommendations
* User-based suggestions

## 🔮 Future Improvements

* User-User Collaborative Filtering
* Item-Item Collaborative Filtering
* Matrix Factorization (SVD)
* Hybrid Recommendation Engine
* Streamlit Web Application
* Movie Poster Integration using TMDB API
* Precision@K and Recall@K Evaluation
* Real-Time Recommendation Serving

## 🎯 Resume Highlights

* Built a machine learning recommendation system using TF-IDF vectorization and cosine similarity.
* Processed and analyzed movie metadata and user rating datasets.
* Developed recommendation pipelines for personalized and genre-based suggestions.
* Generated automated visual analytics and recommendation reports.
* Designed a modular architecture suitable for collaborative filtering and hybrid recommender extensions.

## 📜 License

This project is intended for educational and portfolio purposes.
