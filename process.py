import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer


class MovieDataPreprocessor:
    def __init__(self):
        self.movies_df = None
        self.ratings_df = None
        self.tfidf_matrix = None
        self.scaler = StandardScaler()

    def load_data(self, movies_metadata_path, credits_path, ratings_path):
        """加载所有需要的数据文件"""
        print("Loading datasets...")

        # 加载数据并处理ID类型
        credits_df = pd.read_csv(credits_path)
        movies_df = pd.read_csv(movies_metadata_path)
        ratings_df = pd.read_csv(ratings_path)

        # movieId的处理（ratings数据集中的movieId需要与tmdb的id对应）
        ratings_df['movieId'] = ratings_df['movieId'].astype(str)
        movies_df['id'] = movies_df['id'].astype(str)
        credits_df['movie_id'] = credits_df['movie_id'].astype(str)

        # 首先合并电影数据和演职员数据
        self.movies_df = movies_df.merge(credits_df, left_on='id', right_on='movie_id', how='inner')

        # 保存一个可用于协同过滤的评分数据副本
        self.ratings_df = ratings_df.copy()

        # 打印合并统计
        print(f"Movies before merge: {len(movies_df)}")
        print(f"Credits data: {len(credits_df)}")
        print(f"Movies after merge: {len(self.movies_df)}")
        print(f"Ratings data: {len(ratings_df)}")

    def clean_data(self):
        """数据清理"""
        print("Cleaning data...")

        # 处理缺失值
        self.movies_df['overview'] = self.movies_df['overview'].fillna('')
        self.movies_df['tagline'] = self.movies_df['tagline'].fillna('')

        # 处理数值型数据
        numeric_columns = ['budget', 'revenue', 'runtime', 'popularity']
        for col in numeric_columns:
            self.movies_df[col] = pd.to_numeric(self.movies_df[col], errors='coerce')
            # 使用中位数填充缺失值
            self.movies_df[col] = self.movies_df[col].fillna(self.movies_df[col].median())

        # 处理异常值
        self.movies_df = self.movies_df[
            (self.movies_df['budget'] > 1000) &  # 预算至少1000美元
            (self.movies_df['revenue'] >= 0)  # 收入非负
            ]

        # 处理日期
        self.movies_df['release_date'] = pd.to_datetime(
            self.movies_df['release_date'],
            errors='coerce'
        )

        # 提取时间特征
        self.movies_df['release_year'] = self.movies_df['release_date'].dt.year
        self.movies_df['release_month'] = self.movies_df['release_date'].dt.month
        self.movies_df['release_day'] = self.movies_df['release_date'].dt.day

        # 安全计算财务指标
        self.movies_df['roi'] = np.where(
            self.movies_df['budget'] > 0,
            (self.movies_df['revenue'] - self.movies_df['budget']) / self.movies_df['budget'],
            0
        )
        self.movies_df['profit'] = self.movies_df['revenue'] - self.movies_df['budget']

        # 裁剪异常ROI值
        roi_percentile = np.percentile(self.movies_df['roi'], 95)
        self.movies_df.loc[self.movies_df['roi'] > roi_percentile, 'roi'] = roi_percentile

    def process_json_features(self):
        """处理JSON格式的特征"""
        print("Processing JSON features...")

        # 扩展JSON特征列表
        features = ['cast', 'crew', 'keywords', 'genres',
                    'production_companies', 'production_countries', 'spoken_languages']

        for feature in features:
            if feature in self.movies_df.columns:
                self.movies_df[feature] = self.movies_df[feature].apply(
                    lambda x: literal_eval(str(x)) if isinstance(x, str) else []
                )

        # 提取更多特征
        self.movies_df['director'] = self.movies_df['crew'].apply(self._get_director)
        self.movies_df['main_cast'] = self.movies_df['cast'].apply(self._get_top_three)
        self.movies_df['genres_list'] = self.movies_df['genres'].apply(
            lambda x: [i['name'] for i in x] if isinstance(x, list) else []
        )
        self.movies_df['production_companies_list'] = self.movies_df['production_companies'].apply(
            lambda x: [i['name'] for i in x] if isinstance(x, list) else []
        )
        self.movies_df['production_countries_list'] = self.movies_df['production_countries'].apply(
            lambda x: [i['name'] for i in x] if isinstance(x, list) else []
        )
        self.movies_df['spoken_languages_list'] = self.movies_df['spoken_languages'].apply(
            lambda x: [i['name'] for i in x] if isinstance(x, list) else []
        )
        self.movies_df['keywords_list'] = self.movies_df['keywords'].apply(
            lambda x: [i['name'] for i in x] if isinstance(x, list) else []
        )

    def process_ratings(self):
        """处理评分数据"""
        print("Processing ratings data...")

        # 计算每部电影的评分统计
        rating_stats = self.ratings_df.groupby('movieId').agg({
            'rating': ['count', 'mean', 'std']
        }).reset_index()
        rating_stats.columns = ['movieId', 'rating_count', 'rating_mean', 'rating_std']

        # 将统计数据合并到电影数据中
        self.movies_df = self.movies_df.merge(
            rating_stats,
            left_on='id',
            right_on='movieId',
            how='left'
        )

        # 填充没有评分的电影
        self.movies_df['rating_count'] = self.movies_df['rating_count'].fillna(0)
        self.movies_df['rating_mean'] = self.movies_df['rating_mean'].fillna(0)
        self.movies_df['rating_std'] = self.movies_df['rating_std'].fillna(0)

        # 计算加权评分（IMDB公式）
        m = self.movies_df['rating_count'].quantile(0.90)
        C = self.movies_df['rating_mean'].mean()
        self.movies_df['weighted_rating'] = (
                (self.movies_df['rating_count'] / (self.movies_df['rating_count'] + m)) *
                self.movies_df['rating_mean'] +
                (m / (self.movies_df['rating_count'] + m)) * C
        )

    def create_text_features(self):
        """创建文本特征"""
        print("Creating text features...")

        # 合并所有文本字段
        self.movies_df['text_features'] = (
                self.movies_df['overview'].fillna('') + ' ' +
                self.movies_df['tagline'].fillna('')
        )

        # TF-IDF处理
        tfidf = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = tfidf.fit_transform(self.movies_df['text_features'])

        # 创建更全面的特征汤
        self.movies_df['soup'] = self.movies_df.apply(self._create_soup, axis=1)

    def scale_numeric_features(self):
        """标准化数值特征"""
        print("Scaling numeric features...")

        numeric_features = ['budget', 'revenue', 'runtime', 'popularity',
                            'rating_count', 'rating_mean', 'weighted_rating']

        # 确保数值特征存在
        existing_features = [f for f in numeric_features if f in self.movies_df.columns]

        if existing_features:
            # 移除包含nan的行
            self.movies_df = self.movies_df.dropna(subset=existing_features)

            # 标准化
            self.movies_df[existing_features] = self.scaler.fit_transform(
                self.movies_df[existing_features]
            )

    def analyze_data(self):
        """生成详细的数据分析报告"""
        print("\nDetailed Data Analysis Report:")

        # 基本统计
        print("\nDataset Statistics:")
        print(f"Total movies: {len(self.movies_df)}")
        print(f"Total ratings: {len(self.ratings_df)}")
        print(f"Unique users: {self.ratings_df['userId'].nunique()}")
        print(f"Rating range: {self.ratings_df['rating'].min()} - {self.ratings_df['rating'].max()}")

        # 电影统计
        print("\nMovie Statistics:")
        print(f"Years covered: {self.movies_df['release_year'].min()} - {self.movies_df['release_year'].max()}")
        print(f"Average budget: ${self.movies_df['budget'].mean():,.2f}")
        print(f"Average revenue: ${self.movies_df['revenue'].mean():,.2f}")
        print(f"Average ROI: {self.movies_df['roi'].mean():.2%}")

        # 类型分布
        genres_flat = [genre for genres in self.movies_df['genres_list'] for genre in genres]
        genres_dist = pd.Series(genres_flat).value_counts()
        print("\nTop 10 Genres:")
        print(genres_dist.head(10))

        # 可视化
        self._plot_ratings_distribution()
        self._plot_budget_revenue_relationship()
        self._plot_genres_distribution()
        self._plot_ratings_over_time()

    def _get_director(self, crew):
        """从crew中提取导演"""
        if isinstance(crew, list):
            for member in crew:
                if isinstance(member, dict) and member.get('job') == 'Director':
                    return member.get('name', '')
        return ''

    def _get_top_three(self, cast):
        """获取前三位演员"""
        if isinstance(cast, list):
            return [member.get('name', '') for member in cast[:3] if isinstance(member, dict)]
        return []

    def _create_soup(self, x):
        """创建特征汤，处理可能的缺失值"""
        features = []

        # 添加类型
        if isinstance(x.get('genres_list'), list):
            features.extend(x['genres_list'])

        # 添加演员
        if isinstance(x.get('main_cast'), list):
            features.extend(x['main_cast'])

        # 添加导演
        if x.get('director'):
            features.append(x['director'])

        # 添加关键词
        if isinstance(x.get('keywords_list'), list):
            features.extend(x['keywords_list'])

        # 添加概述词
        if isinstance(x.get('overview'), str):
            features.extend(x['overview'].split())

        # 处理所有特征
        features = [str(f).lower().strip() for f in features if f]
        return ' '.join(features)

    def _plot_ratings_distribution(self):
        """绘制评分分布图"""
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.ratings_df, x='rating', bins=20)
        plt.title('Rating Distribution')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.show()

    def _plot_budget_revenue_relationship(self):
        """绘制预算-收入关系图"""
        plt.figure(figsize=(10, 6))
        plt.scatter(self.movies_df['budget'], self.movies_df['revenue'], alpha=0.5)
        plt.xlabel('Budget ($)')
        plt.ylabel('Revenue ($)')
        plt.title('Budget vs Revenue')
        plt.show()

    def _plot_genres_distribution(self):
        """绘制电影类型分布图"""
        genres_flat = [genre for genres in self.movies_df['genres_list'] for genre in genres]
        genres_dist = pd.Series(genres_flat).value_counts().head(10)

        plt.figure(figsize=(12, 6))
        genres_dist.plot(kind='bar')
        plt.title('Top 10 Movie Genres')
        plt.xlabel('Genre')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def _plot_ratings_over_time(self):
        """绘制随时间变化的评分趋势"""
        self.ratings_df['timestamp'] = pd.to_datetime(self.ratings_df['timestamp'], unit='s')
        ratings_by_time = self.ratings_df.set_index('timestamp')['rating'].resample('M').mean()

        plt.figure(figsize=(12, 6))
        ratings_by_time.plot(kind='line')
        plt.title('Average Rating Over Time')
        plt.xlabel('Time')
        plt.ylabel('Average Rating')
        plt.grid(True)
        plt.show()

    def get_processed_data(self):
        """返回处理后的数据"""
        return self.movies_df, self.ratings_df, self.tfidf_matrix


if __name__ == "__main__":
    preprocessor = MovieDataPreprocessor()

    # 加载数据
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
    preprocessor.analyze_data()

    # 获取处理后的数据
    movies_df, ratings_df, tfidf_matrix = preprocessor.get_processed_data()