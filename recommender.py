import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error, ndcg_score
from sklearn.model_selection import train_test_split
import logging
from svdpp_recommender import SvdppRecommender


class MovieRecommenderSystem:
    def __init__(self, cf_weight=0.7):
        """
        Initialize the hybrid recommender system
        cf_weight: weight for collaborative filtering predictions
        """
        self.cf_weight = cf_weight
        self.name = "Movie Recommender System"

        # Data attributes
        self.movies_df = None
        self.ratings_df = None
        self.tfidf_matrix = None

        # Models
        self.cf_model = None
        self.similarity_matrix = None

        # Results
        self.evaluation_results = {}

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.cold_start_threshold = 5
        self.warm_up_threshold = 20
        self.rating_min = 0.5
        self.rating_max = 5.0

    def fit(self, movies_df, ratings_df, tfidf_matrix):
        """
        Train the recommender system
        """
        self.logger.info("Starting model training...")
        self.movies_df = movies_df
        self.ratings_df = ratings_df
        self.tfidf_matrix = tfidf_matrix

        # Train collaborative filtering model
        self._train_collaborative_filtering()

        # Train content based model
        self._train_content_based()

        self.logger.info("Model training completed")

    def _train_collaborative_filtering(self):
        """
        Load and initialize SVD++ model
        """
        self.logger.info("Loading SVD++ model...")
        try:
            # self.cf_model = SvdppRecommender.load_model('./model/svdpp_model.pkl')
            self.cf_model = SvdppRecommender.load_model('./model/svdpp_model_enhanced.pkl')
            if self.cf_model is None:
                raise Exception("Failed to load SVD++ model")
            self.logger.info("SVD++ model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading SVD++ model: {e}")
            raise

    def _train_content_based(self):
        """
        Train content based model
        """
        self.logger.info("Training content-based model...")
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        self.logger.info("Content-based model trained")

    def predict_collaborative(self, user_id, movie_id):
        """
        Make collaborative filtering prediction
        """
        try:
            return self.cf_model.predict(user_id, movie_id)
        except Exception as e:
            self.logger.error(f"Error in collaborative prediction: {e}")
            return None

    def predict_content_based(self, movie_id):
        """
        Make content based prediction
        """
        try:
            if movie_id not in self.movies_df['id'].values:
                return None

            movie_idx = self.movies_df[self.movies_df['id'] == movie_id].index[0]
            similar_scores = self.similarity_matrix[movie_idx]

            # Get movie ratings
            movie_ratings = self.ratings_df[self.ratings_df['movieId'] == movie_id]['rating']
            if len(movie_ratings) > 0:
                return movie_ratings.mean()
            else:
                return self.ratings_df['rating'].mean()

        except Exception as e:
            self.logger.error(f"Error in content-based prediction: {e}")
            return None

    def clip_rating(self, rating):
        """
        确保评分在合法范围内
        """
        if rating is None:
            return None
        return max(self.rating_min, min(self.rating_max, rating))

    def get_dynamic_weight(self, user_id):
        """
        基于用户状态动态计算权重
        """
        user_ratings_count = len(self.ratings_df[self.ratings_df['userId'] == user_id])

        if user_ratings_count < self.cold_start_threshold:
            return 0.2
        elif user_ratings_count < self.warm_up_threshold:
            # 线性插值
            progress = (user_ratings_count - self.cold_start_threshold) / (
                        self.warm_up_threshold - self.cold_start_threshold)
            return 0.2 + 0.5 * progress
        else:
            return 0.7

    def predict_hybrid(self, user_id, movie_id):
        """
        使用动态权重的混合预测，并确保评分在合法范围内
        """
        try:
            alpha = self.get_dynamic_weight(user_id)

            cf_pred = self.predict_collaborative(user_id, movie_id)
            cb_pred = self.predict_content_based(movie_id)

            # 处理预测值
            if cf_pred is None:
                return self.clip_rating(cb_pred)
            if cb_pred is None:
                return self.clip_rating(cf_pred)

            # 加权融合并确保在范围内
            hybrid_pred = alpha * cf_pred + (1 - alpha) * cb_pred
            return self.clip_rating(hybrid_pred)

        except Exception as e:
            self.logger.error(f"Error in hybrid prediction: {e}")
            return None

    def evaluate(self, test_size=0.2, random_state=42):
        """评估模型性能"""
        _, test_data = train_test_split(
            self.ratings_df,
            test_size=test_size,
            random_state=random_state
        )

        methods = {
            'Collaborative Filtering': self.predict_collaborative,
            'Content Based': lambda u, m: self.predict_content_based(m),  # 修复这里
            'Hybrid': self.predict_hybrid
        }

        for method_name, predict_func in methods.items():
            predictions = []
            actuals = []

            for _, row in test_data.iterrows():
                pred = predict_func(row['userId'], row['movieId'])
                if pred is not None:
                    predictions.append(pred)
                    actuals.append(row['rating'])

            if predictions:
                predictions = np.array(predictions)
                actuals = np.array(actuals)

                self.evaluation_results[method_name] = {
                    'RMSE': np.sqrt(mean_squared_error(actuals, predictions)),
                    'MAE': mean_absolute_error(actuals, predictions),
                    'NDCG': ndcg_score(actuals.reshape(1, -1), predictions.reshape(1, -1))
                }

        return self.evaluation_results

    def get_top_n_recommendations(self, user_id, n=10, method='hybrid'):
        try:
            # Get user's rated movies
            rated_movies = set(self.ratings_df[
                                   self.ratings_df['userId'] == user_id
                                   ]['movieId'])
            all_movies = set(self.movies_df['id'])
            candidate_movies = list(all_movies - rated_movies)

            # Get prediction method
            predict_func = {
                'collaborative': self.predict_collaborative,
                'content': self.predict_content_based,
                'hybrid': self.predict_hybrid
            }[method]

            # Generate predictions
            predictions = []
            for movie_id in candidate_movies:
                pred = predict_func(user_id, movie_id)
                if pred is not None:
                    predictions.append((movie_id, pred))

            # Sort and get top-N
            recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]

            # Get movie details
            recommended_movies = pd.DataFrame(recommendations, columns=['id', 'predicted_rating'])

            recommended_movies = pd.merge(
                recommended_movies,
                self.movies_df[['id', 'original_title', 'genres_list']],
                on='id'
            )

            return recommended_movies
        except Exception as e:
            self.logger.error(f"Error in generating recommendations: {e}")
            return None

def main():
    """
    Example usage
    """
    from process import MovieDataPreprocessor

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

    # Initialize recommender
    recommender = MovieRecommenderSystem(cf_weight=0.7)

    # Train model
    recommender.fit(movies_df, ratings_df, tfidf_matrix)

    # Evaluate model
    evaluation_results = recommender.evaluate()
    print("\nModel Evaluation Results:")
    for model_name, metrics in evaluation_results.items():
        print(f"\n{model_name}:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

    # Generate recommendations for sample user
    user_id = ratings_df['userId'].iloc[0]
    recommendations = recommender.get_top_n_recommendations(user_id, n=5)
    if recommendations is not None:
        # Set pandas options to display all columns and widen the display width
        pd.set_option('display.max_columns', None)  # Display all columns
        pd.set_option('display.max_colwidth', None)  # Don't truncate column contents
        pd.set_option('display.width', 1000)  # Set the display width to a large number

        print(f"\nTop 5 recommendations for user {user_id}:")
        print(recommendations)


if __name__ == "__main__":
    main()