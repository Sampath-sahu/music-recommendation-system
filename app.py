import streamlit as st
import pandas as pd
import numpy as np
import base64
from music_recommender import (
    recommend_song,
    tfidf_matrix,
    feature_names,
    cosine_sim,
    indices,
    df
)

st.set_page_config(
    page_title="Music Recommender",
    page_icon="🎵",
    layout="wide"
)

# ── Session State Defaults ──────────────────────────────────────────
if "random_song" not in st.session_state:
    st.session_state.random_song = ""

# ── Background ──────────────────────────────────────────────────────
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    bg_css = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    /* Dark overlay */
    [data-testid="stAppViewContainer"]::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.45);
        z-index: -1;
    }}

    /* Mobile adjustment */
    @media (max-width: 768px) {{
        [data-testid="stAppViewContainer"] {{
            background-position: center top;
            background-attachment: scroll;
        }}
    }}
    </style>
    """
    st.markdown(bg_css, unsafe_allow_html=True)


# ── Sidebar CSS ─────────────────────────────────────────────────────
sidebar_css = """
<style>
/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
}
[data-testid="stSidebar"] * {
    color: #e0e0e0 !important;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #ffffff !important;
}

/* Sidebar divider override */
[data-testid="stSidebar"] hr {
    border-color: rgba(233, 69, 96, 0.4) !important;
    margin: 0.8rem 0;
}

/* Metric cards */
[data-testid="stSidebar"] [data-testid="stMetric"] {
    background: rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 10px 14px;
    border: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    font-size: 1.3rem !important;
    color: #e94560 !important;
    font-weight: 700 !important;
}
[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
    font-size: 0.78rem !important;
    color: #aaaaaa !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Buttons */
[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.45rem 1rem;
    transition: all 0.25s ease;
    border: 1px solid rgba(255,255,255,0.15);
    background: rgba(233, 69, 96, 0.15);
    color: #ffffff !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(233, 69, 96, 0.45);
    border-color: #e94560;
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(233, 69, 96, 0.25);
}

/* Slider accent */
[data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"] {
    color: #e94560 !important;
}
</style>
"""
st.markdown(sidebar_css, unsafe_allow_html=True)


# ── Background & Header ────────────────────────────────────────────
set_background("background.jpg")

st.markdown("""
<h1 style='
    text-align: center; 
    color: #ffffff;
    font-size: 2.8rem;
    font-weight: 800;
    text-shadow: 0 4px 15px rgba(0, 0, 0, 0.6), 0 2px 8px rgba(0, 0, 0, 0.8);
    letter-spacing: 1px;
    margin-bottom: 0.5rem;
'>
🎧 Music Recommendation System
</h1>
""", unsafe_allow_html=True)
st.markdown("""
<p style='
    text-align: center; 
    color: #e0e0e0;
    font-size: 1rem;
    text-shadow: 0 2px 6px rgba(2, 2, 2, 1.4);
    margin-top: -0.5rem;
    margin-bottom: 1rem;
'>
Find songs matching your vibe 🎵
</p>
""", unsafe_allow_html=True)
st.divider()


# ═══════════════════════════════════════════════════════════════════
#  SIDEBAR

# ── 1. Branding ─────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style='text-align:center; padding: 0.5rem 0 0.2rem;'>
    <span style='font-size:2.4rem;'>🎧</span>
    <h2 style='margin:0; font-size:1.35rem; letter-spacing:0.5px;'>Music Recommender</h2>
    <p style='margin:0; font-size:0.8rem; color:#aaa; font-style:italic;'>Discover your next favourite track</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.divider()

# ── 2. Recommendation Settings ─────────────────────────────────────
st.sidebar.markdown("### 🎚️ Filters")

top_n = st.sidebar.slider(
    "Number of Results",
    min_value=1,
    max_value=20,
    value=5,
    help="How many recommendations to show"
)

st.sidebar.divider()

# ── 3. Quick Actions ───────────────────────────────────────────────
st.sidebar.markdown("### ⚡ Quick Actions")

qa_col1, qa_col2 = st.sidebar.columns(2)

with qa_col1:
    if st.button("🎲 Random Song", use_container_width=True):
        random_row = df.sample(1).iloc[0]
        st.session_state.random_song = random_row["track_name"]
        st.rerun()

with qa_col2:
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.random_song = ""
        st.rerun()

st.sidebar.divider()

# ── 4. Dataset Stats ───────────────────────────────────────────────
st.sidebar.markdown("### 📊 Dataset Stats")

stat_c1, stat_c2, stat_c3 = st.sidebar.columns(3)
stat_c1.metric("Songs",   f"{len(df):,}")
stat_c2.metric("Artists", f"{df['artist_name'].nunique():,}")
stat_c3.metric("Genres",  f"{df['genre'].nunique()}")

st.sidebar.divider()

# ── 5. About (Expander) ───────────────────────────────────────────
with st.sidebar.expander("ℹ️ About this App", expanded=False):
    st.markdown("""
**Music Recommender** finds songs similar to any track
you type in, using **TF-IDF** vectorisation and
**cosine similarity**.

**Tech Stack**
- 🐍 Python · Streamlit
- 📐 scikit-learn (TF-IDF)
- 📊 Pandas · NumPy

**How it works**
1. Songs are represented as TF-IDF vectors
2. Cosine similarity scores are computed
3. Top-N most similar songs are returned
4. Optional mood filter narrows results by genre
""")


# ═══════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════

col1, col2 = st.columns([2, 1])

with col1:
    song_name = st.text_input(
        "🎶 Enter Song Name",
        value=st.session_state.random_song
    )

with col2:
    mood = st.selectbox(
        "🎭 Mood",
        ["None", "Happy", "Sad", "Romantic", "Energetic", "Chill"]
    )

selected_mood = None if mood == "None" else mood

if st.button("🎵 Recommend"):

    if song_name.strip() == "":
        st.warning("Please enter a song name.")
    else:

        with st.spinner("Finding similar songs... 🎵"):
            result = recommend_song(
                song_name,
                df,
                cosine_sim,
                indices,
                mood=selected_mood,
                top_n=top_n
            )
        if not isinstance(result, dict):
            result = {
            "type": "normal",
            "message": "Recommendations generated.",
            "results": result
    }

        st.info(result.get("message", "Recommendations generated."))

        st.subheader("🎶 Recommendations")

        results_df = pd.DataFrame(result["results"])
        st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True
            )