import streamlit as st
import pandas as pd
from intellegent hybrid.recommender import content_based_recommend, hybrid_recommend

# Load movies and ratings data
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

st.set_page_config(page_title="Hybrid Movie Recommender", layout="wide")
st.title("🎬 Hybrid Movie Recommendation System")

st.sidebar.header("🔍 Search Options")

# Select a user ID
user_ids = ratings['userId'].unique()
user_id = st.sidebar.selectbox("Select a User ID", user_ids)

# Select a movie title to base recommendations on
movie_titles = movies['title'].tolist()
movie_title = st.sidebar.selectbox("Select a Movie Title", movie_titles)

# Get movieId from title
movie_id = movies[movies['title'] == movie_title]['movieId'].values[0]

# Choose alpha for hybrid weighting
alpha = st.sidebar.slider("Hybrid Weight (Higher = More Collaborative)", 0.0, 1.0, 0.5, 0.1)

# Recommend
if st.sidebar.button("Get Recommendations"):
    st.subheader(f"📘 Content-Based Recommendations for '{movie_title}':")
    cb_recs = content_based_recommend(movie_id)
    st.dataframe(cb_recs)

    st.subheader(f"🤝 Hybrid Recommendations for User {user_id} based on '{movie_title}':")
    hybrid_recs = hybrid_recommend(user_id, movie_id, alpha)
    st.dataframe(hybrid_recs)

st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ using Streamlit")

