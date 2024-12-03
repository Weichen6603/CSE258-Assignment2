# CSE258-Assignment2

## Introduction

This project is a movie recommendation system that includes data preprocessing, collaborative filtering, content-based recommendation, and a hybrid recommendation model. The system is designed to provide accurate and diverse movie recommendations based on user preferences and movie features.

It also saves evaluation results and recommendation outputs for further analysis and review.

---

## Usage

If you have not downloaded the pre-trained file, first run the `svdpp_recommender.py`:

- This will train a new optimized model.
- Save it to `./model/svdpp_model.pkl`.
- Display the training process and evaluation metrics.

Then run `recommender.py`:

- This will:
  - Correctly load the optimized model.
  - Perform hybrid recommendations.
  - Display evaluation results.
  - Save **evaluation results** to `evaluation_<timestamp>.csv`:  
    Includes metrics like RMSE, MAE, and NDCG for different user groups (e.g., regular users and/or cold-start users ) and recommendation methods (e.g., hybrid, collaborative filtering, content-based).
  - Save **recommendation outputs** to `recommendations_<timestamp>.txt`:  
    Includes the recommended movies for different user groups, with details like movie title, predicted rating, genres, release year, and recommendation reasons.

Both files will be stored in the `./results/` folder for further review and analysis.