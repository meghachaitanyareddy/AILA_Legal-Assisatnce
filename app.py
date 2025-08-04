from flask import Flask, render_template, request
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load documents
docs_path = "dataset/Object_casedocs"
documents = []
filenames = []

for file in os.listdir(docs_path):
    with open(os.path.join(docs_path, file), 'r', encoding='utf-8') as f:
        documents.append(f.read())
        filenames.append(file)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
doc_vectors = vectorizer.fit_transform(documents)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, doc_vectors).flatten()
    top_indices = similarity.argsort()[-5:][::-1]

    results = [(filenames[i], documents[i][:300], round(similarity[i], 2)) for i in top_indices]
    return render_template('results.html', query=query, results=results)

if __name__ == '__main__':
    app.run(debug=True)
