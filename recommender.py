# recommender.py

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error, ndcg_score
from sklearn.model_selection import train_test_split
import logging
import torch
import os

from svdpp_model import GPUSvdppRecommender
from process import MovieDataPreprocessor


class MovieRecommenderSystem:
    def __init__(self, cf_weight=0.7, model_path=None, svdpp_params=None):
        self.cf_weight = cf_weight
        self.name = "Enhanced Movie Recommender System"
        self.model_path = model_path
        self.svdpp_params = svdpp_params or {}

        # 数据属性
        self.movies_df = None
        self.ratings_df = None
        self.tfidf_matrix = None

        # 模型
        self.cf_recommender = None
        self.content_similarity = None
        self.metadata_similarity = None

        # 用户和物品映射
        self.user_mapping = {}
        self.item_mapping = {}

        # 结果
        self.evaluation_results = {}

        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # 参数设置
        self.cold_start_threshold = 5
        self.warm_up_threshold = 20
        self.rating_min = 0.5
        self.rating_max = 5.0

    def fit(self, movies_df, ratings_df, tfidf_matrix):
        """训练推荐系统"""
        self.logger.info("开始训练模型...")
        self.movies_df = movies_df
        self.ratings_df = ratings_df
        self.tfidf_matrix = tfidf_matrix

        # 确保ID类型一致
        self.movies_df['id'] = self.movies_df['id'].astype(str)
        self.ratings_df['movieId'] = self.ratings_df['movieId'].astype(str)

        # 训练协同过滤模型
        self._train_collaborative_filtering()

        # 训练基于内容的模型
        self._train_content_based()

        self.logger.info("模型训练完成")

    def _train_collaborative_filtering(self):
        """训练协同过滤模型"""
        self.logger.info("开始训练协同过滤模型...")

        # 准备训练数据
        train_data = self.ratings_df.merge(
            self.movies_df[
                ['id', 'weighted_rating', 'revenue', 'budget', 'popularity', 'roi', 'rating_mean', 'rating_count']],
            left_on='movieId',
            right_on='id',
            how='left'
        )

        # 填充可能的缺失值
        numerical_features = ['weighted_rating', 'revenue', 'budget', 'popularity', 'roi', 'rating_mean',
                              'rating_count']
        for col in numerical_features:
            train_data[col] = train_data[col].fillna(0)

        # 初始化SVD++推荐器
        self.cf_recommender = GPUSvdppRecommender(**self.svdpp_params)

        if self.model_path and os.path.exists(self.model_path):
            # 先使用少量轮次训练来初始化模型结构
            temp_params = self.svdpp_params.copy()
            temp_params['n_epochs'] = 1
            self.cf_recommender = GPUSvdppRecommender(**temp_params)
            self.cf_recommender.fit(train_data)

            # 加载预训练权重
            checkpoint = torch.load(self.model_path, map_location=self.cf_recommender.device)
            self.cf_recommender.model.load_state_dict(checkpoint['model_state_dict'])
            self.cf_recommender.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.user_mapping = checkpoint['user_mapping']
            self.item_mapping = checkpoint['item_mapping']
        else:
            # 训练新模型
            self.logger.info("训练新模型...")
            self.cf_recommender.fit(train_data)
            self.user_mapping = self.cf_recommender.user_mapping
            self.item_mapping = self.cf_recommender.item_mapping

    def _train_content_based(self):
        """训练基于内容的模型"""
        self.logger.info("训练基于内容的模型...")

        try:
            # 使用TF-IDF相似度
            self.content_similarity = cosine_similarity(self.tfidf_matrix)

            # 确保'soup'列存在
            if 'soup' not in self.movies_df.columns:
                self.logger.warning("找不到'soup'列，将只使用TF-IDF相似度")
                self.metadata_similarity = self.content_similarity
            else:
                # 创建元数据特征向量
                metadata_text = self.movies_df['soup'].fillna('')
                metadata_vectorizer = TfidfVectorizer(stop_words='english')
                metadata_matrix = metadata_vectorizer.fit_transform(metadata_text)
                self.metadata_similarity = cosine_similarity(metadata_matrix)

            self.logger.info(f"相似度矩阵大小: {self.content_similarity.shape}")

        except Exception as e:
            self.logger.error(f"训练基于内容的模型时出错: {str(e)}")
            # 如果出错，创建空的相似度矩阵
            self.content_similarity = np.zeros((len(self.movies_df), len(self.movies_df)))
            self.metadata_similarity = self.content_similarity.copy()

    def predict_collaborative(self, user_id, movie_id):
        """协同过滤预测"""
        if self.cf_recommender is None:
            return None

        try:
            user_idx = self.user_mapping.get(user_id)
            item_idx = self.item_mapping.get(movie_id)

            if user_idx is None or item_idx is None:
                return None

            # 获取用户的评分统计
            user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
            if len(user_ratings) == 0:
                return None

            user_avg_rating = user_ratings['rating'].mean()
            user_rating_count = len(user_ratings)

            # 获取电影特征
            movie_data = self.movies_df[self.movies_df['id'] == movie_id]
            if len(movie_data) == 0:
                return None

            movie_features = movie_data[
                ['weighted_rating', 'revenue', 'budget', 'popularity', 'roi', 'rating_mean', 'rating_count']].iloc[0]

            numerical_features = torch.tensor([
                user_avg_rating,  # 用户平均评分
                user_rating_count,  # 用户评分数量
                movie_features['rating_mean'],  # 电影平均评分
                movie_features['rating_count'],  # 电影评分数量
                movie_features['weighted_rating'],  # 加权评分
                movie_features['popularity'],  # 热度
                movie_features['roi']  # ROI
            ], dtype=torch.float32).unsqueeze(0)

            numerical_features = numerical_features.to(self.cf_recommender.device)
            user_tensor = torch.tensor([user_idx], dtype=torch.long).to(self.cf_recommender.device)
            item_tensor = torch.tensor([item_idx], dtype=torch.long).to(self.cf_recommender.device)

            with torch.no_grad():
                prediction = self.cf_recommender.model(
                    user_tensor,
                    item_tensor,
                    numerical_features=numerical_features
                )

            return prediction.item()

        except Exception as e:
            self.logger.error(f"协同过滤预测错误: {str(e)}")
            return None

    def predict_content_based(self, movie_id):
        """基于内容的预测"""
        try:
            movie_data = self.movies_df[self.movies_df['id'] == movie_id]
            if len(movie_data) == 0:
                return None

            movie_idx = movie_data.index[0]

            # 检查索引是否在相似度矩阵范围内
            if movie_idx >= self.content_similarity.shape[0] or movie_idx >= self.metadata_similarity.shape[0]:
                return None

            # 结合TF-IDF和元数据相似度
            content_scores = self.content_similarity[movie_idx]
            metadata_scores = self.metadata_similarity[movie_idx]

            # 加权组合相似度分数
            combined_scores = 0.6 * content_scores + 0.4 * metadata_scores

            # 获取最相似的电影（排除自身）
            N = 10
            similar_indices = np.argsort(combined_scores)[-N - 1:-1][::-1]

            # 检查是否有足够的相似电影
            if len(similar_indices) == 0:
                return None

            similar_movies = self.movies_df.iloc[similar_indices]

            # 检查是否有评分数据
            if 'weighted_rating' not in similar_movies.columns:
                return None

            # 计算加权评分
            weights = combined_scores[similar_indices]
            weighted_ratings = similar_movies['weighted_rating'].fillna(0).values * weights

            return weighted_ratings.sum() / weights.sum() if weights.sum() != 0 else None

        except Exception as e:
            self.logger.error(f"基于内容的预测错误: {str(e)}")
            return None

    def get_dynamic_weight(self, user_id):
        """基于用户状态获取动态权重"""
        user_ratings = len(self.ratings_df[self.ratings_df['userId'] == user_id])

        if user_ratings < self.cold_start_threshold:
            return 0.2  # 冷启动用户更依赖基于内容的推荐
        elif user_ratings < self.warm_up_threshold:
            progress = (user_ratings - self.cold_start_threshold) / (
                    self.warm_up_threshold - self.cold_start_threshold)
            return 0.2 + 0.5 * progress
        else:
            return 0.7

    def predict_hybrid(self, user_id, movie_id):
        """混合预测"""
        try:
            weight = self.get_dynamic_weight(user_id)
            cf_pred = self.predict_collaborative(user_id, movie_id)
            cb_pred = self.predict_content_based(movie_id)

            if cf_pred is None and cb_pred is None:
                return None
            elif cf_pred is None:
                return max(self.rating_min, min(self.rating_max, cb_pred))
            elif cb_pred is None:
                return max(self.rating_min, min(self.rating_max, cf_pred))

            hybrid_pred = weight * cf_pred + (1 - weight) * cb_pred
            return max(self.rating_min, min(self.rating_max, hybrid_pred))

        except Exception as e:
            self.logger.error(f"混合预测错误: {e}")
            return None

    def get_top_n_recommendations(self, user_id, n=10, method='hybrid'):
        """获取Top-N推荐"""
        try:
            rated_movies = set(self.ratings_df[self.ratings_df['userId'] == user_id]['movieId'])
            candidate_movies = list(set(self.movies_df['id']) - rated_movies)

            predict_func = {
                'collaborative': self.predict_collaborative,
                'content': self.predict_content_based,
                'hybrid': self.predict_hybrid
            }.get(method.lower())

            if predict_func is None:
                raise ValueError(f"未知的推荐方法: {method}")

            predictions = []
            for movie_id in candidate_movies:
                if method == 'content':
                    pred = predict_func(movie_id)
                else:
                    pred = predict_func(user_id, movie_id)

                if pred is not None:
                    predictions.append((movie_id, pred))

            recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]

            # 获取推荐电影的详细信息
            recommended_movies = pd.DataFrame(recommendations, columns=['movieId', 'predicted_rating'])
            recommended_movies = pd.merge(
                recommended_movies,
                self.movies_df[['id', 'original_title', 'genres_list', 'weighted_rating', 'popularity']],
                left_on='movieId',
                right_on='id'
            )

            return recommended_movies

        except Exception as e:
            self.logger.error(f"生成推荐时出错: {e}")
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
            'Content Based': lambda u, m: self.predict_content_based(m),
            'Hybrid': self.predict_hybrid
        }

        for method_name, predict_func in methods.items():
            predictions = []
            actuals = []

            for _, row in test_data.iterrows():
                if method_name == 'Content Based':
                    pred = predict_func(None, row['movieId'])
                else:
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


def main():
    """示例用法"""
    # 初始化预处理器
    preprocessor = MovieDataPreprocessor()

    # 加载和处理数据
    preprocessor.load_data(
        './dataset/tmdb_5000_movies.csv',
        './dataset/tmdb_5000_credits.csv',
        './dataset/ratings_small.csv'
    )

    # 执行所有预处理步骤
    preprocessor.clean_data()
    preprocessor.process_json_features()
    preprocessor.process_ratings()
    preprocessor.create_text_features()
    preprocessor.scale_numeric_features()

    # 获取处理后的数据
    movies_df, ratings_df, tfidf_matrix = preprocessor.get_processed_data()

    # 控制是否使用预训练模型
    # use_pretrained = True
    use_pretrained = False

    if use_pretrained:
        # 使用预训练模型的设置
        svdpp_params = {
            'n_factors': 100,
            'n_epochs': 1,  # 使用预训练模型时只需要1轮
            'batch_size': 2048,
            'learning_rate': 0.001,
            'weight_decay': 0.02,
            'num_workers': 8
        }
        model_path = './model/best_model.pt'
    else:
        # 训练新模型的设置
        svdpp_params = {
            'n_factors': 100,
            'n_epochs': 1000,  # 训练新模型时使用更多轮数
            'batch_size': 2048,
            'learning_rate': 0.001,
            'weight_decay': 0.02,
            'num_workers': 8
        }
        model_path = None

    # 初始化推荐器
    recommender = MovieRecommenderSystem(
        cf_weight=0.7,
        model_path=model_path,
        svdpp_params=svdpp_params
    )

    # 训练/加载模型
    recommender.fit(movies_df, ratings_df, tfidf_matrix)

    # 评估模型
    evaluation_results = recommender.evaluate()
    print("\n模型评估结果:")
    for model_name, metrics in evaluation_results.items():
        print(f"\n{model_name}:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

    # 为示例用户生成推荐
    user_id = ratings_df['userId'].iloc[0]
    print(f"\n为用户 {user_id} 生成推荐:")
    recommendations = recommender.get_top_n_recommendations(user_id, n=5)
    if recommendations is not None:
        # Set pandas options to display all columns and widen the display width
        pd.set_option('display.max_columns', None)  # Display all columns
        pd.set_option('display.max_colwidth', None)  # Don't truncate column contents
        pd.set_option('display.width', 1000)  # Set the display width to a large number
        print("\n推荐的电影:")
        print(recommendations[['original_title', 'genres_list', 'predicted_rating', 'weighted_rating']])


if __name__ == "__main__":
    main()