import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds

# Load Data
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Preprocess genres
movies['genres'] = movies['genres'].str.replace('|', ' ')

# Content-Based Filtering Setup
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
movie_id_to_idx = pd.Series(movies.index, index=movies['movieId'])

def content_based_recommend(movie_id, top_n=10):
    if movie_id not in movie_id_to_idx:
        return pd.DataFrame()
    idx = movie_id_to_idx[movie_id]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices][['movieId', 'title', 'genres']]

# Collaborative Filtering Setup
user_ids = ratings['userId'].unique()
movie_ids = ratings['movieId'].unique()
user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
movie_id_to_idx_mat = {mid: i for i, mid in enumerate(movie_ids)}

num_users = len(user_ids)
num_movies = len(movie_ids)
rating_matrix = np.zeros((num_users, num_movies))

for row in ratings.itertuples():
    uidx = user_id_to_idx[row.userId]
    midx = movie_id_to_idx_mat[row.movieId]
    rating_matrix[uidx, midx] = row.rating

user_ratings_mean = np.mean(rating_matrix, axis=1).reshape(-1, 1)
rating_matrix_norm = rating_matrix - user_ratings_mean
U, sigma, Vt = svds(rating_matrix_norm, k=50)
sigma = np.diag(sigma)
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean

def predict_rating(user_id, movie_id):
    if user_id not in user_id_to_idx or movie_id not in movie_id_to_idx_mat:
        return np.mean(ratings['rating'])
    uidx = user_id_to_idx[user_id]
    midx = movie_id_to_idx_mat[movie_id]
    return all_user_predicted_ratings[uidx, midx]

def hybrid_recommend(user_id, movie_id, alpha=0.5, top_n=10):
    if movie_id not in movie_id_to_idx:
        return pd.DataFrame()

    idx = movie_id_to_idx[movie_id]
    content_scores = cosine_sim[idx]
    
    collab_scores = [predict_rating(user_id, mid) for mid in movies['movieId']]
    collab_scores = np.array(collab_scores)

    content_scores_norm = (content_scores - content_scores.min()) / (content_scores.max() - content_scores.min() + 1e-8)
    collab_scores_norm = (collab_scores - collab_scores.min()) / (collab_scores.max() - collab_scores.min() + 1e-8)

    hybrid_scores = alpha * collab_scores_norm + (1 - alpha) * content_scores_norm
    indices = hybrid_scores.argsort()[::-1]
    top_indices = [i for i in indices if movies.iloc[i]['movieId'] != movie_id][:top_n]

    return movies.iloc[top_indices][['movieId', 'title', 'genres']]

def get_all_users():
    return sorted(ratings['userId'].unique())

def get_all_movies():
    return movies[['movieId', 'title']].sort_values('title')

