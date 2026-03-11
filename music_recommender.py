import streamlit as st
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@st.cache_data
def load_data():
    try:
        df = pd.read_csv("tcc_ceds_music.csv", encoding="latin1")
        return df
    except Exception as e:
        st.error(e)

df = load_data()
MOOD_GENRE_MAP = {
    "Romantic": ["jazz", "pop"],
    "Chill": ["jazz", "blues"],
    "Energetic": ["rock", "hiphop"],
    "Happy": ["pop"],
    "Sad": ["country", "blues"]
}
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    return text

df['combined_features'] = (
    df['track_name'] + " " +
    df['artist_name'] + " " +
    df['genre']
)

df['combined_features'] = df['combined_features'].apply(clean_text)

tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

feature_names = tfidf.get_feature_names_out()
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

df["song_clean"] = df["track_name"].str.lower().str.strip()
indices = pd.Series(df.index, index=df["song_clean"]).drop_duplicates()

def cold_start_recommend(df, top_n=5):
    """
    Fallback recommendation when input song is not found.
    Currently returns top-N songs from dataset.
    """
    return df[['track_name', 'artist_name', 'genre']].head(top_n)

def get_top_keywords(song_index, tfidf_matrix, feature_names, top_n=3):
    vector = tfidf_matrix[song_index].toarray()[0]
    top_indices = np.argsort(vector)[-top_n:][::-1]
    return [feature_names[i] for i in top_indices if vector[i] > 0]

def recommend_song(track_name, df, cosine_sim, indices, mood=None, top_n=5):
    song_name = track_name.lower().strip()

    top_n = min(top_n, len(df) - 1)

    if song_name in indices:
        idx = indices[song_name]
    else:
        matches = df[df["song_clean"].str.contains(song_name, na=False)]
        if matches.empty:
            return {
                "type": "cold_start",
                "message": "Song not found. Showing popular recommendations.",
                "results": cold_start_recommend(df, top_n)
            }
        idx = matches.index[0]

    scores = list(enumerate(cosine_sim[idx].ravel()))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = scores[1 : top_n + 1]

    if not scores:
        return {
            "type": "cold_start",
            "message": "Not enough similar songs found. Showing popular recommendations.",
            "results": cold_start_recommend(df, top_n)
        }

    results = []

    for i, score in scores:

        if i >= len(df):
         continue

        genre = str(df.iloc[i]["genre"]).lower()

        # 🎭 Mood filter
        if mood and mood in MOOD_GENRE_MAP:
            allowed = MOOD_GENRE_MAP[mood]
            if not any(g in genre for g in allowed):
                continue

        explanation = get_top_keywords(
            i,
            tfidf_matrix,
            feature_names,
            top_n=3
        )

        results.append({
            "track_name": df.iloc[i]["track_name"],
            "artist_name": df.iloc[i]["artist_name"],
            "genre": df.iloc[i]["genre"],
            "similarity (%)": round(score * 100, 2),
            "why": explanation
        })

    if not results:
        return {
            "type": "cold_start",
            "message": f"No {mood} songs found. Showing general recommendations.",
            "results": cold_start_recommend(df, top_n)
        }

    return {
        "type": "normal",
        "message": f"Similar {mood if mood else ''} songs found.",
        "results": results[:top_n]
    }