import streamlit as st
import numpy as np
import pandas as pd
import re
import collections
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import openai

# --------------------------
# Helper Functions
# --------------------------
def initialize_sentence_transformer():
    """Initialize and return the SentenceTransformer model."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

def get_embedding(text, model):
    """Return the embedding for the provided text."""
    return model.encode(text)

def preprocess_text(text):
    """Basic text preprocessing (for demonstration, here we just lowercase the text)."""
    return text.lower()

# --------------------------
# Semantic Gap Analyzer
# --------------------------
def semantic_gap_analyzer(competitor_texts, target_text, n_value=2, top_n=10, min_df=1, max_df=1.0):
    """
    For each competitor, compute TF-IDF scores for n-grams and then identify gap candidates 
    where competitor n-gram importance exceeds that of the target.
    """
    # Preprocess texts
    competitor_texts = [preprocess_text(text) for text in competitor_texts]
    target_text = preprocess_text(target_text)
    
    # Vectorize competitor content
    vectorizer = TfidfVectorizer(ngram_range=(n_value, n_value), min_df=min_df, max_df=max_df, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(competitor_texts)
    feature_names = vectorizer.get_feature_names_out()
    df_tfidf_competitors = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
    
    # Vectorize target content using the same vocabulary
    target_tfidf_vector = vectorizer.transform([target_text])
    df_tfidf_target = pd.DataFrame(target_tfidf_vector.toarray(), columns=feature_names)
    
    gap_candidates = []
    model = initialize_sentence_transformer()
    target_embedding = get_embedding(target_text, model)
    
    # For each competitor, get top n-grams and compute the TF-IDF difference
    for idx in range(len(competitor_texts)):
        row = df_tfidf_competitors.iloc[idx]
        sorted_ngrams = row.sort_values(ascending=False).head(top_n)
        for ngram, comp_tfidf in sorted_ngrams.items():
            # Only consider n-grams present in target text
            if ngram in df_tfidf_target.columns:
                target_tfidf = df_tfidf_target.iloc[0][ngram]
                diff = comp_tfidf - target_tfidf
                if diff > 0:
                    gap_candidates.append((ngram, diff))
    
    # Remove duplicates and sort by gap score
    gap_candidates = list(set(gap_candidates))
    gap_candidates.sort(key=lambda x: x[1], reverse=True)
    return gap_candidates

# --------------------------
# Keyword Clustering
# --------------------------
def keyword_clustering(gap_candidates, n_clusters=5):
    """
    Cluster the gap n-grams using SentenceTransformer embeddings and KMeans.
    Returns a dictionary mapping each cluster label to its list of n-grams.
    """
    gap_ngrams = [ngram for ngram, score in gap_candidates]
    if not gap_ngrams:
        return {}
    model = initialize_sentence_transformer()
    embeddings = np.vstack([get_embedding(ngram, model) for ngram in gap_ngrams])
    
    # Cluster using KMeans
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(embeddings)
    
    clusters = {}
    for label, ngram, score in zip(labels, gap_ngrams, [score for _, score in gap_candidates]):
        clusters.setdefault(label, []).append((ngram, score))
    return clusters

# --------------------------
# ChatGPT API Integration for Recommendations
# --------------------------
def get_recommendations(semantic_gap_results, keyword_clusters, api_key):
    """
    Compose a prompt that includes the gap analysis and clustering results,
    then call the ChatGPT API to obtain SEO recommendations.
    """
    openai.api_key = api_key
    
    # Format the results as text
    gap_text = "\n".join([f"{ngram}: {score:.3f}" for ngram, score in semantic_gap_results])
    clusters_text = ""
    for cluster, items in keyword_clusters.items():
        clusters_text += f"\nCluster {cluster}: " + ", ".join([ngram for ngram, _ in items])
    
    prompt = (
        "You are an SEO expert. Based on the following analysis results, "
        "provide detailed recommendations for improving the website's content strategy.\n\n"
        "Semantic Gap Analysis Results (n-gram and gap score):\n"
        f"{gap_text}\n\n"
        "Keyword Clusters:\n"
        f"{clusters_text}\n\n"
        "Recommendations:"
    )
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an SEO expert."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=300
    )
    return response.choices[0].message["content"]

# --------------------------
# Streamlit App Interface
# --------------------------
def main():
    st.title("Semantic Gap Analyzer & Keyword Clustering with Recommendations")
    
    st.header("Input Data")
    competitor_input = st.text_area(
        "Enter competitor content (for multiple entries, separate with '---'):",
        placeholder="Competitor 1 content --- Competitor 2 content"
    )
    target_input = st.text_area("Enter target website content:")
    
    if st.button("Analyze"):
        if not competitor_input or not target_input:
            st.error("Please provide both competitor and target content.")
            return
        
        # Split competitor content if multiple entries are provided
        competitor_texts = [text.strip() for text in competitor_input.split('---') if text.strip()]
        
        # Run Semantic Gap Analysis
        gap_candidates = semantic_gap_analyzer(competitor_texts, target_input)
        st.subheader("Semantic Gap Analysis Results")
        st.write(gap_candidates)
        
        # Run Keyword Clustering
        clusters = keyword_clustering(gap_candidates)
        st.subheader("Keyword Clusters")
        for cluster, items in clusters.items():
            st.write(f"**Cluster {cluster}:** " + ", ".join([ngram for ngram, score in items]))
        
        st.header("ChatGPT SEO Recommendations")
        api_key = st.text_input("Enter your OpenAI API Key:", type="password")
        if st.button("Get Recommendations"):
            if not api_key:
                st.error("Please enter your OpenAI API key.")
                return
            recommendations = get_recommendations(gap_candidates, clusters, api_key)
            st.subheader("SEO Recommendations")
            st.write(recommendations)

if __name__ == "__main__":
    main()
