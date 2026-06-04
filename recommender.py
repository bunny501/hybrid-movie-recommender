"""
Content-Based Movie Recommendation System
Uses MovieLens-style data with cosine similarity on TF-IDF genre features.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


class MovieRecommender:
    """
    Content-Based Movie Recommendation System.
    
    Pipeline:
    1. Data Loading & Preprocessing
    2. Feature Extraction (TF-IDF on genres)
    3. Cosine Similarity Matrix Computation
    4. Personalized Recommendation Generation
    """

    def __init__(self, movies_path: str, ratings_path: str):
        self.movies_path = movies_path
        self.ratings_path = ratings_path
        self.movies_df = None
        self.ratings_df = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.indices = None
        self._load_and_preprocess()
        self._build_similarity_matrix()

    # ─────────────────────────────────────────────
    # Step 1: Data Loading & Preprocessing
    # ─────────────────────────────────────────────
    def _load_and_preprocess(self):
        """Load CSVs, clean genres, engineer features."""
        self.movies_df = pd.read_csv(self.movies_path)
        self.ratings_df = pd.read_csv(self.ratings_path)

        # Normalize genres: replace '|' with space for TF-IDF
        self.movies_df['genres_clean'] = (
            self.movies_df['genres']
            .fillna('Unknown')
            .str.replace('|', ' ', regex=False)
            .str.lower()
        )

        # Extract year from title
        self.movies_df['year'] = (
            self.movies_df['title']
            .str.extract(r'\((\d{4})\)')
            .astype(float)
        )

        # Compute average rating and rating count per movie
        avg_ratings = (
            self.ratings_df.groupby('movieId')['rating']
            .agg(['mean', 'count'])
            .rename(columns={'mean': 'avg_rating', 'count': 'rating_count'})
            .reset_index()
        )
        self.movies_df = self.movies_df.merge(avg_ratings, on='movieId', how='left')
        self.movies_df['avg_rating'].fillna(0, inplace=True)
        self.movies_df['rating_count'].fillna(0, inplace=True)

        # Popularity score (IMDB-style weighted rating)
        C = self.movies_df['avg_rating'].mean()
        m = self.movies_df['rating_count'].quantile(0.25)
        v = self.movies_df['rating_count']
        R = self.movies_df['avg_rating']
        self.movies_df['popularity_score'] = (v / (v + m)) * R + (m / (v + m)) * C

        # Index for fast lookup
        self.indices = pd.Series(
            self.movies_df.index,
            index=self.movies_df['title']
        ).drop_duplicates()

    # ─────────────────────────────────────────────
    # Step 2: Feature Extraction & Similarity Matrix
    # ─────────────────────────────────────────────
    def _build_similarity_matrix(self):
        """TF-IDF vectorize genres → cosine similarity matrix."""
        tfidf = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),
            min_df=1,
            stop_words='english'
        )
        self.tfidf_matrix = tfidf.fit_transform(self.movies_df['genres_clean'])
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

    # ─────────────────────────────────────────────
    # Step 3: Content-Based Recommendations
    # ─────────────────────────────────────────────
    def get_similar_movies(self, title: str, top_n: int = 10) -> pd.DataFrame:
        """
        Given a movie title, return top_n most similar movies
        ranked by cosine similarity score.
        """
        if title not in self.indices:
            raise ValueError(f"Movie '{title}' not found in dataset.")

        idx = self.indices[title]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n + 1]  # Exclude the movie itself

        movie_indices = [i[0] for i in sim_scores]
        scores = [round(i[1], 4) for i in sim_scores]

        result = self.movies_df.iloc[movie_indices][
            ['title', 'genres', 'avg_rating', 'rating_count', 'year', 'popularity_score']
        ].copy()
        result['similarity_score'] = scores
        result = result.sort_values('similarity_score', ascending=False)
        return result.reset_index(drop=True)

    # ─────────────────────────────────────────────
    # Step 4: Personalized User Recommendations
    # ─────────────────────────────────────────────
    def get_user_recommendations(self, user_id: int, top_n: int = 10) -> pd.DataFrame:
        """
        Generate personalized recommendations for a user:
        - Find movies they rated highly (>= 3.5)
        - Build a weighted genre preference profile
        - Score all unseen movies against that profile
        - Rank by weighted content score + popularity
        """
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        if user_ratings.empty:
            raise ValueError(f"No ratings found for user {user_id}.")

        # Movies they liked
        liked = user_ratings[user_ratings['rating'] >= 3.5]['movieId'].values
        seen_ids = user_ratings['movieId'].values

        if len(liked) == 0:
            liked = user_ratings.nlargest(3, 'rating')['movieId'].values

        # Build user profile: weighted average of liked movie TF-IDF vectors
        liked_indices = self.movies_df[self.movies_df['movieId'].isin(liked)].index
        weights = []
        for mid in liked:
            row = user_ratings[user_ratings['movieId'] == mid]
            weights.append(float(row['rating'].values[0]) if not row.empty else 3.5)

        weights = np.array(weights)
        if len(liked_indices) == 0:
            return pd.DataFrame()

        liked_vectors = self.tfidf_matrix[liked_indices].toarray()
        w = weights[:len(liked_vectors)] / weights[:len(liked_vectors)].sum()
        user_profile = np.average(liked_vectors, axis=0, weights=w)

        # Score all unseen movies
        all_vectors = self.tfidf_matrix.toarray()
        content_scores = cosine_similarity([user_profile], all_vectors)[0]

        # Combine with popularity
        scaler = MinMaxScaler()
        pop_scores = scaler.fit_transform(
            self.movies_df[['popularity_score']].values
        ).flatten()

        final_scores = 0.75 * content_scores + 0.25 * pop_scores

        # Filter out seen movies
        unseen_mask = ~self.movies_df['movieId'].isin(seen_ids)
        scored = self.movies_df.copy()
        scored['recommendation_score'] = final_scores
        scored = scored[unseen_mask].sort_values('recommendation_score', ascending=False)

        result = scored.head(top_n)[
            ['title', 'genres', 'avg_rating', 'rating_count',
             'year', 'popularity_score', 'recommendation_score']
        ].copy()
        result['recommendation_score'] = result['recommendation_score'].round(4)
        return result.reset_index(drop=True)

    def get_genre_recommendations(self, genres: list, top_n: int = 10) -> pd.DataFrame:
        """
        Recommend movies matching a list of preferred genres,
        ranked by genre overlap + popularity.
        """
        genre_query = ' '.join(g.lower() for g in genres)
        tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1)
        tfidf.fit(self.movies_df['genres_clean'])
        query_vec = tfidf.transform([genre_query])
        movie_vecs = tfidf.transform(self.movies_df['genres_clean'])
        scores = cosine_similarity(query_vec, movie_vecs)[0]

        result = self.movies_df.copy()
        result['genre_match_score'] = scores
        result = result.sort_values(
            ['genre_match_score', 'popularity_score'], ascending=False
        ).head(top_n)
        return result[['title', 'genres', 'avg_rating', 'genre_match_score']].reset_index(drop=True)

    def get_stats(self) -> dict:
        """Return dataset statistics for reporting."""
        all_genres = set()
        for g in self.movies_df['genres']:
            for genre in g.split('|'):
                all_genres.add(genre.strip())

        return {
            'total_movies': len(self.movies_df),
            'total_ratings': len(self.ratings_df),
            'total_users': self.ratings_df['userId'].nunique(),
            'unique_genres': sorted(all_genres),
            'avg_rating_overall': round(self.ratings_df['rating'].mean(), 2),
            'tfidf_matrix_shape': self.tfidf_matrix.shape,
            'cosine_sim_shape': self.cosine_sim.shape,
            'year_range': (
                int(self.movies_df['year'].min()),
                int(self.movies_df['year'].max())
            )
        }
