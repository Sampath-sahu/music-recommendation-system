\# 🎵 Music Recommendation System



\## 📌 Project Overview



This project is a \*\*Content-Based Music Recommendation System\*\* that suggests songs similar to a selected track based on textual metadata such as \*\*track name, artist name, and genre\*\*.



The system applies \*\*Natural Language Processing (NLP)\*\* techniques using \*\*TF-IDF Vectorization\*\* and \*\*Cosine Similarity\*\* to compute song similarity and generate recommendations.



An interactive web interface is built using \*\*Streamlit\*\*, allowing users to search for a song and receive similar music recommendations instantly.



\---



\## 🚀 Live Demo



Deployed using Streamlit Cloud.



Live App:

https://music-recommendation-system1902.streamlit.app



\---



\## 🛠️ Tech Stack



\* Python

\* Pandas

\* NumPy

\* Scikit-learn

\* Streamlit

\* Natural Language Processing (NLP)



\---



\## ⚙️ How It Works



1\. The dataset containing music metadata is loaded.

2\. Important textual features such as \*\*track name, artist name, and genre\*\* are combined.

3\. Text data is converted into numerical vectors using \*\*TF-IDF Vectorization\*\*.

4\. \*\*Cosine Similarity\*\* is calculated between songs.

5\. When a user selects a song, the system returns \*\*Top-N most similar songs\*\*.



\---



\## 📂 Project Structure



```

music-recommendation-system

│

├── app.py

├── music\_recommender.py

├── requirements.txt

├── tcc\_ceds\_music.csv

├── background.jpg

└── README.md

```



\---



\## ⚙️ Installation \& Running Locally



Clone the repository:



```

git clone https://github.com/yourusername/music-recommendation-system.git

cd music-recommendation-system

```



Install dependencies:



```

pip install -r requirements.txt

```



Run the application:



```

streamlit run app.py

```



\---



\## 🎯 Future Improvements



\* Add album artwork and music posters

\* Integrate Spotify API for real-time music data

\* Improve UI/UX design

\* Add collaborative filtering recommendations



\---



\## 👨‍💻 Author



Sampath Sahu



