import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
df = pd.read_csv("tcc_ceds_music.csv")
df = df.reset_index(drop=True)

df['features'] = (
    df['track_name'].fillna('') + ' ' +
    df['artist_name'].fillna('') + ' ' +
    df['genre'].fillna('')
)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['features'])

# Cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix)

# Index mapping
indices = pd.Series(df.index, index=df['track_name'].str.lower()).drop_duplicates()

# Save everything
with open("cosine_sim.pkl", "wb") as f:
    pickle.dump(cosine_sim, f)

with open("indices.pkl", "wb") as f:
    pickle.dump(indices, f)

with open("df.pkl", "wb") as f:
    pickle.dump(df, f)

print("✅ Precomputation done. Files saved.")