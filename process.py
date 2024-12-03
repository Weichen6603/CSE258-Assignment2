# process.py

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
        """Load all required data files."""
        print("Loading datasets...")

        # Load datasets and handle ID types
        credits_df = pd.read_csv(credits_path)
        movies_df = pd.read_csv(movies_metadata_path)
        ratings_df = pd.read_csv(ratings_path)

        # Handle movieId to match between ratings and tmdb IDs
        ratings_df['movieId'] = ratings_df['movieId'].astype(str)
        movies_df['id'] = movies_df['id'].astype(str)
        credits_df['movie_id'] = credits_df['movie_id'].astype(str)

        # Merge movie data and credits data
        self.movies_df = movies_df.merge(credits_df, left_on='id', right_on='movie_id', how='inner')

        # Save a copy of the ratings data for collaborative filtering
        self.ratings_df = ratings_df.copy()

        # Print merge statistics
        print(f"Movies before merge: {len(movies_df)}")
        print(f"Credits data: {len(credits_df)}")
        print(f"Movies after merge: {len(self.movies_df)}")
        print(f"Ratings data: {len(ratings_df)}")

    def clean_data(self):
        """Clean the data."""
        print("Cleaning data...")

        # Handle missing values
        self.movies_df['overview'] = self.movies_df['overview'].fillna('')
        self.movies_df['tagline'] = self.movies_df['tagline'].fillna('')

        # Handle numeric data
        numeric_columns = ['budget', 'revenue', 'runtime', 'popularity']
        for col in numeric_columns:
            self.movies_df[col] = pd.to_numeric(self.movies_df[col], errors='coerce')

        # Remove movies with budget or revenue equal to zero
        self.movies_df = self.movies_df[
            (self.movies_df['budget'] > 0) &
            (self.movies_df['revenue'] > 0)
        ]

        # Handle dates
        self.movies_df['release_date'] = pd.to_datetime(
            self.movies_df['release_date'],
            errors='coerce'
        )

        # Extract temporal features
        self.movies_df['release_year'] = self.movies_df['release_date'].dt.year
        self.movies_df['release_month'] = self.movies_df['release_date'].dt.month
        self.movies_df['release_day'] = self.movies_df['release_date'].dt.day

        # Calculate financial metrics
        self.movies_df['roi'] = (self.movies_df['revenue'] - self.movies_df['budget']) / self.movies_df['budget']
        self.movies_df['profit'] = self.movies_df['revenue'] - self.movies_df['budget']

    def process_json_features(self):
        """Process JSON format features."""
        print("Processing JSON features...")

        # List of JSON features to expand
        features = ['cast', 'crew', 'keywords', 'genres',
                    'production_companies', 'production_countries', 'spoken_languages']

        for feature in features:
            if feature in self.movies_df.columns:
                self.movies_df[feature] = self.movies_df[feature].apply(
                    lambda x: literal_eval(str(x)) if isinstance(x, str) else []
                )

        # Extract additional features
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
        """Process ratings data."""
        print("Processing ratings data...")

        # Calculate rating statistics for each movie
        rating_stats = self.ratings_df.groupby('movieId').agg({
            'rating': ['count', 'mean', 'std']
        }).reset_index()
        rating_stats.columns = ['movieId', 'rating_count', 'rating_mean', 'rating_std']

        # Merge statistics with movie data
        self.movies_df = self.movies_df.merge(
            rating_stats,
            left_on='id',
            right_on='movieId',
            how='left'
        )

        # Fill movies with no ratings
        self.movies_df['rating_count'] = self.movies_df['rating_count'].fillna(0)
        self.movies_df['rating_mean'] = self.movies_df['rating_mean'].fillna(0)
        self.movies_df['rating_std'] = self.movies_df['rating_std'].fillna(0)

        # Calculate weighted rating (IMDB formula)
        m = self.movies_df['rating_count'].quantile(0.90)
        C = self.movies_df['rating_mean'].mean()
        self.movies_df['weighted_rating'] = (
                (self.movies_df['rating_count'] / (self.movies_df['rating_count'] + m)) *
                self.movies_df['rating_mean'] +
                (m / (self.movies_df['rating_count'] + m)) * C
        )

    def create_text_features(self):
        """Create text-based features."""
        print("Creating text features...")

        # Combine text fields
        self.movies_df['text_features'] = (
                self.movies_df['overview'].fillna('') + ' ' +
                self.movies_df['tagline'].fillna('')
        )

        # TF-IDF processing
        tfidf = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = tfidf.fit_transform(self.movies_df['text_features'])

        # Create a comprehensive soup feature
        self.movies_df['soup'] = self.movies_df.apply(self._create_soup, axis=1)

    def scale_numeric_features(self):
        """Scale numeric features."""
        print("Scaling numeric features...")

        numeric_features = ['budget', 'revenue', 'runtime', 'popularity',
                            'rating_count', 'rating_mean', 'weighted_rating']

        # Ensure numeric features exist
        existing_features = [f for f in numeric_features if f in self.movies_df.columns]

        if existing_features:
            # Drop rows with NaN in numeric features
            self.movies_df = self.movies_df.dropna(subset=existing_features)

            # Scale the features
            self.movies_df[existing_features] = self.scaler.fit_transform(
                self.movies_df[existing_features]
            )

    def analyze_data(self):
        """Generate a detailed data analysis report."""
        print("\nDetailed Data Analysis Report:")

        # Basic statistics
        print("\nDataset Statistics:")
        print(f"Total movies: {len(self.movies_df)}")
        print(f"Total ratings: {len(self.ratings_df)}")
        print(f"Unique users: {self.ratings_df['userId'].nunique()}")
        print(f"Rating range: {self.ratings_df['rating'].min()} - {self.ratings_df['rating'].max()}")

        # Movie statistics
        print("\nMovie Statistics:")
        print(f"Years covered: {self.movies_df['release_year'].min()} - {self.movies_df['release_year'].max()}")
        print(f"Average budget: ${self.movies_df['budget'].mean():,.2f}")
        print(f"Average revenue: ${self.movies_df['revenue'].mean():,.2f}")
        print(f"Average ROI: {self.movies_df['roi'].mean():.2%}")

        # Genre distribution
        genres_flat = [genre for genres in self.movies_df['genres_list'] for genre in genres]
        genres_dist = pd.Series(genres_flat).value_counts()
        print("\nTop 10 Genres:")
        print(genres_dist.head(10))

        # Visualization
        self._plot_ratings_distribution()
        self._plot_budget_revenue_relationship()
        self._plot_genres_distribution()
        self._plot_ratings_over_time()

    def _get_director(self, crew):
        """Extract director from crew."""
        if isinstance(crew, list):
            for member in crew:
                if isinstance(member, dict) and member.get('job') == 'Director':
                    return member.get('name', '')
        return ''

    def _get_top_three(self, cast):
        """Get the top three actors."""
        if isinstance(cast, list):
            return [member.get('name', '') for member in cast[:3] if isinstance(member, dict)]
        return []

    def _create_soup(self, x):
        """Create a soup of features, handling missing values."""
        features = []

        # Add genres
        if isinstance(x.get('genres_list'), list):
            features.extend(x['genres_list'])

        # Add main cast
        if isinstance(x.get('main_cast'), list):
            features.extend(x['main_cast'])

        # Add director
        if x.get('director'):
            features.append(x['director'])

        # Add keywords
        if isinstance(x.get('keywords_list'), list):
            features.extend(x['keywords_list'])

        # Add overview words
        if isinstance(x.get('overview'), str):
            features.extend(x['overview'].split())

        # Process all features
        features = [str(f).lower().strip() for f in features if f]
        return ' '.join(features)

    def _plot_ratings_distribution(self):
        """Plot rating distribution."""
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.ratings_df, x='rating', bins=20)
        plt.title('Rating Distribution')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.show()

    def _plot_budget_revenue_relationship(self):
        """Plot budget-revenue relationship."""
        plt.figure(figsize=(10, 6))
        plt.scatter(self.movies_df['budget'], self.movies_df['revenue'], alpha=0.5)
        plt.xlabel('Budget ($)')
        plt.ylabel('Revenue ($)')
        plt.title('Budget vs Revenue')
        plt.show()

    def _plot_genres_distribution(self):
        """Plot genre distribution."""
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
        """Plot rating trends over time."""
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
        """Return processed data."""
        return self.movies_df, self.ratings_df, self.tfidf_matrix


if __name__ == "__main__":
    preprocessor = MovieDataPreprocessor()

    # Load data
    preprocessor.load_data(
        './dataset/tmdb_5000_movies.csv',
        './dataset/tmdb_5000_credits.csv',
        './dataset/ratings_small.csv'
    )

    # Execute all preprocessing steps
    preprocessor.clean_data()
    preprocessor.process_json_features()
    preprocessor.process_ratings()
    preprocessor.create_text_features()
    preprocessor.scale_numeric_features()
    preprocessor.analyze_data()

    # Get processed data
    movies_df, ratings_df, tfidf_matrix = preprocessor.get_processed_data()
