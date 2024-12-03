# recommender.py

import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error, ndcg_score
from sklearn.model_selection import train_test_split
import logging
from svdpp_recommender import SvdppRecommender
from datetime import datetime


class MovieRecommenderSystem:
    def __init__(self, base_cf_weight=0.5):
        """
        Initialize the enhanced hybrid recommender system
        """
        self.base_cf_weight = base_cf_weight
        self.name = "Enhanced Movie Recommender System"

        # Data attributes
        self.movies_df = None
        self.ratings_df = None
        self.tfidf_matrix = None

        # Models
        self.cf_model = None
        self.similarity_matrix = None
        self.global_mean_rating = None

        # Caches
        self.movie_neighbors_cache = {}
        self.user_genre_preferences = {}
        self.user_rating_patterns = {}
        self.movie_popularity_scores = {}
        self.temporal_weights = {}

        # Results and analysis
        self.evaluation_results = {}
        self.prediction_stats = {
            'cold_start': [],
            'regular': [],
            'item_cold_start': []
        }

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def fit(self, movies_df, ratings_df, tfidf_matrix):
        """
        Train the recommender system
        """
        self.logger.info("Starting model training...")
        self.movies_df = movies_df
        self.ratings_df = ratings_df
        self.tfidf_matrix = tfidf_matrix
        self.global_mean_rating = ratings_df['rating'].mean()

        # Initialize models
        self._train_collaborative_filtering()
        self._train_content_based()
        self._precompute_movie_neighbors()

        # Precompute user and movie features
        self._precompute_all_features()

        self.logger.info("Model training completed")

    def _train_collaborative_filtering(self):
        """
        Load and initialize SVD++ model
        """
        self.logger.info("Loading SVD++ model...")
        try:
            self.cf_model = SvdppRecommender.load_model('./model/svdpp_model.pkl')
            if self.cf_model is None:
                raise Exception("Failed to load SVD++ model")
            self.logger.info("SVD++ model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading SVD++ model: {e}")
            raise

    def _train_content_based(self):
        """
        Enhanced content-based model training
        """
        self.logger.info("Training content-based model...")
        try:
            # Text feature cosine similarity
            text_similarity = cosine_similarity(self.tfidf_matrix)

            # Genre Similarity
            genre_similarities = np.zeros((len(self.movies_df), len(self.movies_df)))
            for i, movie1 in self.movies_df.iterrows():
                for j, movie2 in self.movies_df.iterrows():
                    genres1 = set(movie1['genres_list'])
                    genres2 = set(movie2['genres_list'])
                    if genres1 and genres2:
                        genre_similarities[i, j] = len(genres1 & genres2) / len(genres1 | genres2)

            # Year similarity
            year_similarities = np.zeros_like(genre_similarities)
            years = self.movies_df['release_year'].values
            for i in range(len(years)):
                for j in range(len(years)):
                    year_diff = abs(years[i] - years[j])
                    year_similarities[i, j] = np.exp(-year_diff / 10)  # 10年半衰期

            # Combining similarity matrices
            self.similarity_matrix = (
                    0.6 * text_similarity +
                    0.3 * genre_similarities +
                    0.1 * year_similarities
            )

            self.logger.info("Content-based model trained successfully")

        except Exception as e:
            self.logger.error(f"Error in content-based training: {e}")
            raise

    def _precompute_movie_neighbors(self, k=10):
        """
        Pre-computing similar movies for each movie
        """
        self.logger.info("Precomputing movie neighbors...")
        for idx, movie_id in enumerate(self.movies_df['id']):
            similar_indices = np.argsort(self.similarity_matrix[idx])[-k - 1:][::-1]
            similar_scores = self.similarity_matrix[idx][similar_indices]
            self.movie_neighbors_cache[movie_id] = {
                'indices': similar_indices,
                'scores': similar_scores
            }

    def _precompute_all_features(self):
        """
        Precomputing features for all users and movies
        """
        self.logger.info("Precomputing features...")

        # User features
        for user_id in self.ratings_df['userId'].unique():
            self._calculate_user_genre_preferences(user_id)
            self._calculate_user_rating_pattern(user_id)

        # Movie features
        for movie_id in self.movies_df['id']:
            self._calculate_movie_popularity_score(movie_id)
            self._calculate_movie_temporal_weight(movie_id)

    def _calculate_user_genre_preferences(self, user_id):
        """
        Calculate user preference for different movie genres
        """
        if user_id in self.user_genre_preferences:
            return self.user_genre_preferences[user_id]

        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        genre_scores = {}

        for _, rating in user_ratings.iterrows():
            movie = self.movies_df[self.movies_df['id'] == rating['movieId']]
            if not movie.empty:
                for genre in movie.iloc[0]['genres_list']:
                    if genre not in genre_scores:
                        genre_scores[genre] = {'count': 0, 'sum_rating': 0.0}
                    genre_scores[genre]['count'] += 1
                    genre_scores[genre]['sum_rating'] += rating['rating']

        # Calculate average ratings and preferences
        for genre in genre_scores:
            count = genre_scores[genre]['count']
            sum_rating = genre_scores[genre]['sum_rating']
            genre_scores[genre] = {
                'avg_rating': sum_rating / count,
                'preference_score': (sum_rating / count) * np.log1p(count)
            }

        self.user_genre_preferences[user_id] = genre_scores
        return genre_scores

    def _calculate_user_rating_pattern(self, user_id):
        """
        Analyzing user rating patterns
        """
        if user_id in self.user_rating_patterns:
            return self.user_rating_patterns[user_id]

        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        if len(user_ratings) == 0:
            return None

        ratings_array = user_ratings['rating'].values
        timestamps = user_ratings['timestamp'].values

        pattern = {
            'mean': np.mean(ratings_array),
            'std': np.std(ratings_array),
            'median': np.median(ratings_array),
            'rating_count': len(ratings_array),
            'time_span': (timestamps.max() - timestamps.min()) / (24 * 60 * 60),
            'rating_entropy': self._calculate_rating_entropy(ratings_array),
            'genre_diversity': len(self._calculate_user_genre_preferences(user_id))
        }

        self.user_rating_patterns[user_id] = pattern
        return pattern

    def _calculate_rating_entropy(self, ratings):
        """
        Calculating rating entropy to measure the uncertainty of user ratings
        """
        unique, counts = np.unique(ratings, return_counts=True)
        probs = counts / len(ratings)
        return -np.sum(probs * np.log(probs))

    def _calculate_movie_temporal_weight(self, movie_id):
        """
        Movie temporal weight calculation
        """
        if movie_id in self.temporal_weights:
            return self.temporal_weights[movie_id]

        movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
        current_year = datetime.now().year

        years_diff = current_year - movie['release_year']
        temporal_weight = np.exp(-years_diff / 5)

        self.temporal_weights[movie_id] = temporal_weight
        return temporal_weight

    def _calculate_movie_popularity_score(self, movie_id):
        """
        Calculating movie popularity scores
        """
        if movie_id in self.movie_popularity_scores:
            return self.movie_popularity_scores[movie_id]

        movie_ratings = self.ratings_df[self.ratings_df['movieId'] == movie_id]
        if len(movie_ratings) == 0:
            return 0.0

        # Using Bayesian averaging
        C = 10  # Minimum rating number threshold
        m = self.global_mean_rating
        R = movie_ratings['rating'].mean()
        v = len(movie_ratings)

        popularity = (v * R + C * m) / (v + C)

        self.movie_popularity_scores[movie_id] = popularity
        return popularity

    def predict_collaborative(self, user_id, movie_id):
        """
        Make collaborative filtering prediction
        """
        try:
            pred = self.cf_model.predict(user_id, movie_id)
            return np.clip(pred, 1.0, 5.0) if pred is not None else None
        except Exception as e:
            self.logger.error(f"Error in collaborative prediction: {e}")
            return None

    def predict_content_based(self, movie_id):
        """
        Enhanced content-based prediction with improved diversity and accuracy
        """
        try:
            if movie_id not in self.movies_df['id'].values:
                return None

            if movie_id not in self.movie_neighbors_cache:
                return self.global_mean_rating

            neighbors = self.movie_neighbors_cache[movie_id]
            similar_movies = self.movies_df.iloc[neighbors['indices']]
            similar_scores = neighbors['scores']

            target_movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
            weighted_ratings = []
            weights = []
            genre_penalties = []

            for idx, movie in similar_movies.iterrows():
                movie_ratings = self.ratings_df[
                    self.ratings_df['movieId'] == movie['id']
                    ]
                if len(movie_ratings) > 0:
                    # 评分可信度调整
                    rating_count = len(movie_ratings)
                    avg_rating = movie_ratings['rating'].mean()
                    rating_confidence = 1.0 / (1.0 + np.exp(-rating_count / 50))  # Sigmoid函数

                    # 时间相关性
                    year_diff = abs(movie['release_year'] - target_movie['release_year'])
                    time_relevance = np.exp(-year_diff / 10)  # 10年半衰期

                    # 类型相似度
                    common_genres = set(movie['genres_list']) & set(target_movie['genres_list'])
                    genre_similarity = len(common_genres) / max(len(movie['genres_list']),
                                                                len(target_movie['genres_list']))

                    # 组合权重
                    combined_weight = (
                            similar_scores[len(weights)] * 0.4 +  # 相似度权重
                            rating_confidence * 0.3 +  # 评分可信度权重
                            time_relevance * 0.2 +  # 时间相关性权重
                            genre_similarity * 0.1  # 类型相似度权重
                    )

                    weighted_ratings.append(avg_rating)
                    weights.append(combined_weight)
                    genre_penalties.append(genre_similarity)

            if weighted_ratings:
                # 调整softmax温度
                temperature = 1.0  # 增大温度参数
                weights = np.exp(np.array(weights) / temperature)
                weights = weights / np.sum(weights)

                # 基础预测
                base_prediction = np.average(weighted_ratings, weights=weights)

                # 类型多样性调整
                genre_diversity = np.mean(genre_penalties)
                diversity_adjustment = 0.95 + 0.1 * genre_diversity  # 减小调整幅度

                # 时间因子
                temporal_weight = self._calculate_movie_temporal_weight(movie_id)
                time_factor = 0.9 + 0.1 * temporal_weight  # 进一步减小时间影响

                # 流行度调整
                popularity_score = self._calculate_movie_popularity_score(movie_id)
                popularity_factor = 0.97 + 0.06 * (popularity_score / 5.0)  # 减小流行度影响

                final_prediction = (
                        base_prediction *
                        diversity_adjustment *
                        time_factor *
                        popularity_factor
                )

                return np.clip(final_prediction, 2.5, 4.5)

        except Exception as e:
            self.logger.error(f"Error in content-based prediction: {e}")
            return None

    def _calculate_adaptive_weights(self, user_id, movie_id):
        """
        Calculate adaptive weights
        """
        try:
            user_pattern = self._calculate_user_rating_pattern(user_id)
            genre_prefs = self._calculate_user_genre_preferences(user_id)
            temporal_weight = self._calculate_movie_temporal_weight(movie_id)
            popularity_score = self._calculate_movie_popularity_score(movie_id)

            if user_pattern is None:
                return 0.3

            cf_weight = self.base_cf_weight * 1.2

            # User pattern adjustments
            if user_pattern['rating_count'] < 5:
                cf_weight *= 0.7
            elif user_pattern['genre_diversity'] > 5:
                cf_weight *= 1.1

            # Rating pattern adjustment
            rating_volatility = user_pattern['rating_entropy'] / np.log(5)
            cf_weight *= (1 + 0.2 * rating_volatility)

            # Genre preferences adjustment
            movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
            genre_match_score = 0
            for genre in movie['genres_list']:
                if genre in genre_prefs:
                    genre_match_score += genre_prefs[genre]['preference_score'] / 5.0

            if genre_match_score > 0:
                cf_weight *= (1 + 0.1 * genre_match_score)

            # Time and popularity adjustments
            cf_weight *= (0.9 + 0.1 * temporal_weight)
            if popularity_score > self.global_mean_rating:
                cf_weight *= 1.1

            return np.clip(cf_weight, 0.3, 0.7)

        except Exception as e:
            self.logger.error(f"Error in adaptive weight calculation: {e}")
            return self.base_cf_weight

    def predict_hybrid(self, user_id, movie_id):
        """
        Enhanced hybrid prediction with optimized weighting
        """
        try:
            cf_pred = self.predict_collaborative(user_id, movie_id)
            cb_pred = self.predict_content_based(movie_id)

            if cf_pred is None and cb_pred is None:
                return self.global_mean_rating
            if cf_pred is None:
                return cb_pred
            if cb_pred is None:
                return cf_pred

            # Calculate adaptive weights
            cf_weight = self._calculate_adaptive_weights(user_id, movie_id)

            # Combine predictions
            base_pred = cf_weight * cf_pred + (1 - cf_weight) * cb_pred

            # Temporal adjustment
            temporal_weight = self._calculate_movie_temporal_weight(movie_id)
            time_factor = 0.9 + 0.1 * temporal_weight

            # Noise injection based on user rating variance
            user_variance = self._get_user_rating_variance(user_id)
            noise_scale = max(0.05, user_variance / 20)
            noise = np.random.normal(0, noise_scale)

            # Final prediction
            final_pred = base_pred * time_factor + noise
            final_pred = np.clip(final_pred, 2.5, 4.5)

            # Record prediction stats
            user_ratings = len(self.ratings_df[self.ratings_df['userId'] == user_id])
            movie_ratings = len(self.ratings_df[self.ratings_df['movieId'] == movie_id])

            category = 'cold_start' if user_ratings < 5 else \
                'item_cold_start' if movie_ratings < 10 else 'regular'

            self.prediction_stats[category].append({
                'user_id': user_id,
                'movie_id': movie_id,
                'cf_weight': cf_weight,
                'cf_pred': cf_pred,
                'cb_pred': cb_pred,
                'final_pred': final_pred,
                'temporal_weight': temporal_weight,
                'rating_count': user_ratings
            })

            return final_pred

        except Exception as e:
            self.logger.error(f"Error in hybrid prediction: {e}")
            return None

    def _get_user_rating_variance(self, user_id):
        """
        Calculate user rating variance
        """
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]['rating']
        return user_ratings.var() if len(user_ratings) > 0 else 0.5

    def evaluate(self, test_size=0.2, random_state=42):
        """
        Enhanced model evaluation
        """
        self.logger.info("Starting model evaluation...")

        # Split test data
        _, test_data = train_test_split(
            self.ratings_df,
            test_size=test_size,
            random_state=random_state,
            stratify=self.ratings_df['userId'].map(
                lambda x: 'cold' if len(self.ratings_df[self.ratings_df['userId'] == x]) < 5 else 'regular'
            )
        )

        # Evaluation methods
        methods = {
            'Collaborative Filtering': self.predict_collaborative,
            'Content Based': lambda user_id, movie_id: self.predict_content_based(movie_id),
            'Hybrid': self.predict_hybrid
        }

        # Evaluation user types
        user_types = {
            'Cold Start': lambda x: len(self.ratings_df[self.ratings_df['userId'] == x]) < 5,
            'Regular': lambda x: len(self.ratings_df[self.ratings_df['userId'] == x]) >= 5
        }

        for method_name, predict_func in methods.items():
            self.evaluation_results[method_name] = {}

            for user_type, type_func in user_types.items():
                predictions = []
                actuals = []

                for _, row in test_data.iterrows():
                    if type_func(row['userId']):
                        pred = predict_func(row['userId'], row['movieId'])
                        if pred is not None:
                            predictions.append(pred)
                            actuals.append(row['rating'])

                if predictions:
                    predictions = np.array(predictions)
                    actuals = np.array(actuals)

                    self.evaluation_results[method_name][user_type] = {
                        'RMSE': np.sqrt(mean_squared_error(actuals, predictions)),
                        'MAE': mean_absolute_error(actuals, predictions),
                        'NDCG': ndcg_score(
                            actuals.reshape(1, -1),
                            predictions.reshape(1, -1)
                        )
                    }

        self.logger.info("Model evaluation completed")
        return self.evaluation_results

    def analyze_predictions(self):
        """
        Enhanced prediction analysis
        """
        print("\nDetailed Prediction Analysis:")
        for category in ['cold_start', 'regular', 'item_cold_start']:
            if not self.prediction_stats[category]:
                continue

            stats = pd.DataFrame(self.prediction_stats[category])
            print(f"\n{category.title()} Users:")
            print(f"Average CF weight: {stats['cf_weight'].mean():.3f}")
            print(f"Average temporal weight: {stats['temporal_weight'].mean():.3f}")
            print(f"Average CF prediction: {stats['cf_pred'].mean():.3f}")
            print(f"Average CB prediction: {stats['cb_pred'].mean():.3f}")
            print(f"Average final prediction: {stats['final_pred'].mean():.3f}")
            print(f"Average prediction difference (CF-CB): "
                  f"{(stats['cf_pred'] - stats['cb_pred']).abs().mean():.3f}")
            print(f"Rating count range: {stats['rating_count'].min()} - {stats['rating_count'].max()}")

            # Additional analysis for regular users
            if len(stats) > 1:
                print(f"Prediction diversity (std): {stats['final_pred'].std():.3f}")
                print(f"CF weight range: {stats['cf_weight'].min():.3f} - {stats['cf_weight'].max():.3f}")

    def get_recommendation_diversity_score(self, recommendations):
        """
        Calculate recommendation diversity score
        """
        if recommendations is None or len(recommendations) == 0:
            return 0.0

        # Genre diversity
        all_genres = []
        for genres in recommendations['genres_list']:
            all_genres.extend(genres)
        genre_diversity = len(set(all_genres)) / len(all_genres) if all_genres else 0

        # Time diversity
        year_diversity = np.std(recommendations['release_year']) / 10 if len(recommendations) > 1 else 0

        # Rating diversity
        rating_diversity = np.std(recommendations['predicted_rating']) / 5 if len(recommendations) > 1 else 0

        return (genre_diversity + year_diversity + rating_diversity) / 3

    def get_top_n_recommendations(self, user_id, n=10, method='hybrid', include_reasons=True):
        """
        Enhanced recommendation generation
        """
        try:
            rated_movies = set(self.ratings_df[
                                   self.ratings_df['userId'] == user_id
                                   ]['movieId'])
            candidate_movies = list(set(self.movies_df['id']) - rated_movies)

            # Prediction function
            predict_func = {
                'collaborative': self.predict_collaborative,
                'content': lambda user_id, movie_id: self.predict_content_based(movie_id),
                'hybrid': self.predict_hybrid
            }[method]

            # Consider diversity when obtaining prediction scores
            predictions = []
            for movie_id in candidate_movies:
                pred = predict_func(user_id, movie_id) if method != 'content' \
                    else predict_func(user_id, movie_id)
                if pred is not None:
                    # Add random noise for diversity
                    diversity_bonus = np.random.normal(0, 0.1)
                    predictions.append({
                        'movie_id': movie_id,
                        'predicted_rating': pred + diversity_bonus
                    })

            # Sort and get top N recommendations
            recommendations = sorted(
                predictions,
                key=lambda x: x['predicted_rating'],
                reverse=True
            )[:n]

            # Merge with movie details
            rec_df = pd.merge(
                pd.DataFrame([{
                    'id': r['movie_id'],
                    'predicted_rating': r['predicted_rating']
                } for r in recommendations]),
                self.movies_df[[
                    'id', 'original_title', 'genres_list', 'overview',
                    'rating_mean', 'rating_count', 'release_year'
                ]],
                on='id'
            )

            if include_reasons:
                rec_df['recommendation_reason'] = rec_df.apply(
                    lambda x: self._get_recommendation_reason(user_id, x['id']),
                    axis=1
                )

            # Calculate diversity score
            diversity_score = self.get_recommendation_diversity_score(rec_df)
            self.logger.info(f"Recommendation diversity score: {diversity_score:.3f}")

            return rec_df

        except Exception as e:
            self.logger.error(f"Error in generating recommendations: {e}")
            return None

    def _get_recommendation_reason(self, user_id, movie_id):
        """
        Generate personalized recommendation reasons
        """
        try:
            user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
            user_genres = []
            user_rated_movies = []
            reasons = []

            # Collect user genre preferences and rated movies
            for _, rating in user_ratings.iterrows():
                movie = self.movies_df[self.movies_df['id'] == rating['movieId']]
                if not movie.empty:
                    user_genres.extend(movie.iloc[0]['genres_list'])
                    if rating['rating'] >= 4.0:
                        user_rated_movies.append(movie.iloc[0])

            movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]

            # Recommend based on user genre preferences
            common_genres = set(movie['genres_list']) & set(user_genres)
            if common_genres:
                reasons.append(f"Based on your interest in {', '.join(common_genres)} movies")

            # Recommend similar movies to highly rated movies
            if movie_id in self.movie_neighbors_cache and user_rated_movies:
                neighbors = self.movies_df.iloc[self.movie_neighbors_cache[movie_id]['indices']]
                similar_rated = [m for m in user_rated_movies if m['id'] in neighbors['id'].values]
                if similar_rated:
                    reasons.append(f"Similar to {similar_rated[0]['original_title']} which you rated highly")

            # Recommend recent movies
            if movie['release_year'] >= datetime.now().year - 2:
                reasons.append("Recent release")

            # Recommend popular movies
            popularity = self._calculate_movie_popularity_score(movie_id)
            if popularity > self.global_mean_rating + 0.5:
                reasons.append("Highly rated by other users")

            return " & ".join(reasons) if reasons else "Based on your overall preferences"

        except Exception as e:
            self.logger.error(f"Error in generating recommendation reason: {e}")
            return "Based on your overall preferences"


def main():
    """
    Enhanced usage of the recommender system
    """
    from process import MovieDataPreprocessor
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Initialize preprocessor
    preprocessor = MovieDataPreprocessor()

    # Load and process data
    preprocessor.load_data(
        './dataset/tmdb_5000_movies.csv',
        './dataset/tmdb_5000_credits.csv',
        './dataset/ratings_small.csv'
    )
    preprocessor.clean_data()
    preprocessor.process_json_features()
    preprocessor.process_ratings()
    preprocessor.create_text_features()
    preprocessor.scale_numeric_features()

    # Get processed data
    movies_df, ratings_df, tfidf_matrix = preprocessor.get_processed_data()

    # Initialize recommender with optimal base weight
    recommender = MovieRecommenderSystem(base_cf_weight=0.45)

    # Train model
    recommender.fit(movies_df, ratings_df, tfidf_matrix)

    # Evaluate model
    evaluation_results = recommender.evaluate()

    # Visualize evaluation results
    metrics = ['RMSE', 'MAE', 'NDCG']
    plt.figure(figsize=(15, 5))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, 3, i)
        data = []
        for model, user_types in evaluation_results.items():
            for user_type, metrics in user_types.items():
                data.append({
                    'Model': model,
                    'User Type': user_type,
                    metric: metrics[metric]
                })
        df = pd.DataFrame(data)
        sns.barplot(x='Model', y=metric, hue='User Type', data=df)
        plt.title(f'{metric} by Model and User Type')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Analyze prediction patterns
    print("\nAnalyzing prediction patterns...")
    recommender.analyze_predictions()

    # Generate personalized recommendations
    print("\nGenerating personalized recommendations:")

    # User groups based on rating counts
    user_rating_counts = ratings_df['userId'].value_counts()
    user_groups = {
        'New Users (< 5 ratings)': user_rating_counts[user_rating_counts < 5],
        'Casual Users (5-20 ratings)': user_rating_counts[(user_rating_counts >= 5) & (user_rating_counts < 20)],
        'Active Users (20+ ratings)': user_rating_counts[user_rating_counts >= 20]
    }

    # Select a user from each group and get recommendations
    for group_name, users in user_groups.items():
        if not users.empty:
            user_id = users.index[0]
            print(f"\n{group_name}")
            print(f"User ID: {user_id}, Number of ratings: {users[user_id]}")

            # Get recommendations using different methods
            for method in ['hybrid', 'collaborative', 'content']:
                print(f"\nRecommendations using {method} method:")
                recommendations = recommender.get_top_n_recommendations(
                    user_id,
                    n=5,
                    method=method,
                    include_reasons=True
                )

                if recommendations is not None:
                    # Calculate diversity score
                    diversity_score = recommender.get_recommendation_diversity_score(recommendations)
                    print(f"Diversity Score: {diversity_score:.3f}")

                    # Display recommendations
                    pd.set_option('display.max_columns', None)
                    pd.set_option('display.max_colwidth', None)
                    pd.set_option('display.width', 1000)

                    # Display selected columns
                    display_columns = [
                        'original_title', 'predicted_rating', 'genres_list',
                        'release_year', 'recommendation_reason'
                    ]
                    print(recommendations[display_columns])

    # Save results
    try:
        os.makedirs('./results', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        evaluation_df = pd.DataFrame([
            {
                'model': model,
                'user_type': user_type,
                'metric': metric,
                'value': value
            }
            for model, user_types in evaluation_results.items()
            for user_type, metrics in user_types.items()
            for metric, value in metrics.items()
        ])
        evaluation_df.to_csv(f'./results/evaluation_{timestamp}.csv', index=False)

        with open(f'./results/recommendations_{timestamp}.txt', 'w') as f:
            for group_name, users in user_groups.items():
                if not users.empty:
                    user_id = users.index[0]
                    recommendations = recommender.get_top_n_recommendations(user_id, n=5)
                    if recommendations is not None:
                        f.write(f"\n{group_name} - User {user_id}:\n")
                        f.write(recommendations.to_string())
                        f.write("\n" + "=" * 80 + "\n")

        print("\nResults saved successfully!")

    except Exception as e:
        print(f"Error saving results: {e}")


if __name__ == "__main__":
    main()