import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests


movies = pd.read_csv("/Users/anands/DATA_SCIENCE/movie_reco/tmdb_5000_movies.csv")
credits = pd.read_csv("/Users/anands/DATA_SCIENCE/movie_reco/tmdb_5000_credits.csv")
df = movies.merge(credits, on="title", how="left")


def convert(text):
    try:
        return " ".join([i['name'] for i in ast.literal_eval(text)])
    except:
        return ""

def get_cast(text):
    try:
        return " ".join([i['name'] for i in ast.literal_eval(text)[:3]])
    except:
        return ""

def get_director(text):
    try:
        for i in ast.literal_eval(text):
            if i['job'] == 'Director':
                return i['name']
    except:
        return ""

# Preprocessing
df['genres'] = df['genres'].apply(convert)
df['cast'] = df['cast'].apply(get_cast)
df['crew'] = df['crew'].apply(get_director)
df['overview'] = df['overview'].fillna('')
df['combined'] = df['overview'] + " " + df['genres'] + " " + df['cast'] + " " + df['crew']
df.fillna('', inplace=True)


tfidf = TfidfVectorizer(stop_words='english')
d_matrix = tfidf.fit_transform(df['combined'])
similarity = cosine_similarity(d_matrix)


TMDB_API_KEY = "99b9cd21f1b052c2fcd57d74efffcd10"  # Replace with your TMDB API Key
IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
    data = requests.get(url).json()
    if data.get('poster_path'):
        return IMAGE_BASE_URL + data['poster_path']
    return None


def recommend(movie_title, top_n=25):
    movie_title_lower = movie_title.lower()
    df_titles_lower = df['title'].str.lower()

    if movie_title_lower not in df_titles_lower.values:
        return [], None

    idx = df_titles_lower[df_titles_lower == movie_title_lower].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    recommended_movies = []
    for i in scores:
        movie = df.iloc[i[0]]
        poster = fetch_poster(movie['id'])
        recommended_movies.append((movie['title'], poster))

    searched_movie_poster = fetch_poster(df.iloc[idx]['id'])
    return recommended_movies, searched_movie_poster


st.set_page_config(layout="wide")
st.title("🎬 Movie Recommendation System")


selected_movie = st.sidebar.selectbox("Search Movie", df['title'].values)

if st.sidebar.button("Show Recommendations"):
    recommendations, searched_poster = recommend(selected_movie)

    if not recommendations and not searched_poster:
        st.sidebar.warning("Movie not found!")
    else:
      
        st.sidebar.subheader("🎯 You Selected:")
        if searched_poster:
            st.sidebar.image(searched_poster, use_container_width=True)
        st.sidebar.caption(selected_movie)

    
        st.subheader("⭐ Top 25 Recommendations")
        rec_cols = st.columns(5)  # 5 posters per row
        for i, (title, poster) in enumerate(recommendations):
            with rec_cols[i % 5]:
                if poster:
                    st.image(poster, use_container_width=True)
                st.caption(title)
