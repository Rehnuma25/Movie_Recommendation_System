from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ---------------- Load & Merge Data ----------------
movies = pd.read_csv('movies.csv')
tags = pd.read_csv('tags.csv')

# Aggregate tags
tags_agg = tags.groupby('movieId')['tag'].apply(lambda x: " ".join(x)).reset_index()
movies_tags = pd.merge(movies, tags_agg, on='movieId', how='left')

# Fill NaN tags
movies_tags['tag'] = movies_tags['tag'].fillna('')

# Combine content features
movies_tags['content'] = movies_tags['genres'] + " " + movies_tags['tag']

# TF-IDF + Cosine Similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_tags['content'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# ---------------- Recommendation Function ----------------
def recommend_movie(title):
    # Partial match search
    matches = movies_tags[movies_tags['title'].str.contains(title, case=False, na=False)]
    if matches.empty:
        return f"No match found for '{title}'", []
    
    matched_title = matches.iloc[0]['title']
    idx = matches.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    movie_indices = [i[0] for i in sim_scores]
    recommendations = movies_tags['title'].iloc[movie_indices].to_list()
    return matched_title, recommendations

# ---------------- Routes ----------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_title = request.form['movie_title']
    matched_title, recommendations = recommend_movie(movie_title)
    return render_template('result.html', 
                           input_title=movie_title, 
                           matched_title=matched_title,
                           recommendations=recommendations)

# ---------------- Run ----------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
