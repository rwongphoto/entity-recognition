import streamlit as st
import numpy as np
import pandas as pd
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import openai
import plotly.express as px

# --------------------------
# Helper Functions
# --------------------------
def initialize_sentence_transformer():
    """Initialize and return the SentenceTransformer model."""
    return SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text, model):
    """Return the embedding for the provided text."""
    return model.encode(text)

def preprocess_text(text):
    """Simple text preprocessing: convert to lowercase."""
    return text.lower()

# --------------------------
# Semantic Gap Analyzer
# --------------------------
def semantic_gap_analyzer(competitor_texts, target_text, n_value=2, top_n=10, min_df=1, max_df=1.0):
    """
    Process competitor and target texts to compute TF‑IDF scores for n‑grams.
    Returns a sorted list of gap candidates where competitor n‑gram importance exceeds that of the target.
    """
    competitor_texts = [preprocess_text(text) for text in competitor_texts]
    target_text = preprocess_text(target_text)
    
    vectorizer = TfidfVectorizer(ngram_range=(n_value, n_value), min_df=min_df, max_df=max_df, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(competitor_texts)
    feature_names = vectorizer.get_feature_names_out()
    df_tfidf_competitors = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
    
    target_tfidf_vector = vectorizer.transform([target_text])
    df_tfidf_target = pd.DataFrame(target_tfidf_vector.toarray(), columns=feature_names)
    
    gap_candidates = []
    model = initialize_sentence_transformer()
    for idx in range(len(competitor_texts)):
        row = df_tfidf_competitors.iloc[idx]
        sorted_ngrams = row.sort_values(ascending=False).head(top_n)
        for ngram, comp_tfidf in sorted_ngrams.items():
            if ngram in df_tfidf_target.columns:
                target_tfidf = df_tfidf_target.iloc[0][ngram]
                diff = comp_tfidf - target_tfidf
                if diff > 0:
                    gap_candidates.append((ngram, diff))
    # Remove duplicates and sort by gap score (highest first)
    gap_candidates = list(set(gap_candidates))
    gap_candidates.sort(key=lambda x: x[1], reverse=True)
    return gap_candidates

# --------------------------
# Keyword Clustering
# --------------------------
def keyword_clustering(gap_candidates, n_clusters=5):
    """
    Cluster the gap n‑grams using SentenceTransformer embeddings and KMeans.
    Returns a tuple: (clusters dictionary, embeddings array, cluster labels).
    """
    gap_ngrams = [ngram for ngram, score in gap_candidates]
    if not gap_ngrams:
        return {}, None, None
    model = initialize_sentence_transformer()
    embeddings = np.vstack([get_embedding(ngram, model) for ngram in gap_ngrams])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(embeddings)
    
    clusters = {}
    for label, ngram, score in zip(labels, gap_ngrams, [score for _, score in gap_candidates]):
        clusters.setdefault(label, []).append((ngram, score))
    return clusters, embeddings, labels

def plot_keyword_clusters(embeddings, labels, gap_ngrams):
    """
    Reduce embedding dimensions to 2 with PCA and generate a scatter plot of the clusters using Plotly.
    """
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    df_plot = pd.DataFrame({
        "x": embeddings_2d[:, 0],
        "y": embeddings_2d[:, 1],
        "Cluster": [f"Cluster {label}" for label in labels],
        "Keyword": gap_ngrams
    })
    fig = px.scatter(df_plot, x="x", y="y", color="Cluster", text="Keyword",
                     title="Keyword Clusters (PCA Projection)")
    fig.update_traces(textposition="top center")
    return fig

# --------------------------
# ChatGPT API Integration
# --------------------------
def get_recommendations(semantic_gap_results, keyword_clusters, api_key):
    """
    Compose a prompt that includes the gap analysis and clustering results,
    then call the ChatGPT API to obtain SEO recommendations.
    """
    openai.api_key = api_key
    gap_text = "\n".join([f"{ngram}: {score:.3f}" for ngram, score in semantic_gap_results])
    clusters_text = ""
    for cluster, items in keyword_clusters.items():
        clusters_text += f"\nCluster {cluster}: " + ", ".join([ngram for ngram, _ in items])
    
    prompt = (
        "You are an SEO expert. Based on the following analysis results, provide detailed recommendations "
        "for improving the website's content strategy.\n\n"
        "Semantic Gap Analysis Results (n-gram and gap score):\n"
        f"{gap_text}\n\n"
        "Keyword Clusters:\n"
        f"{clusters_text}\n\n"
        "Recommendations:"
    )
    
    try:
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
    except Exception as e:
        st.error(f"Error calling ChatGPT API: {e}. Please ensure you're using the latest version of the openai Python package.")
        return "No recommendations generated due to API error."

# --------------------------
# Main Streamlit App Interface
# --------------------------
def main():
    st.title("Semantic Gap Analyzer & Keyword Clustering with ChatGPT Recommendations")
    
    st.header("Input Data")
    competitor_input = st.text_area(
        "Enter competitor content (separate multiple entries with '---'):",
        placeholder="Competitor content 1 --- Competitor content 2"
    )
    target_input = st.text_area("Enter target website content:")
    
    n_value = st.selectbox("Select n‑gram size", options=[1, 2, 3, 4, 5], index=1)
    top_n = st.slider("Select top n n‑grams to consider", min_value=1, max_value=20, value=10)
    
    if st.button("Analyze"):
        if not competitor_input or not target_input:
            st.error("Please provide both competitor and target content.")
            return
        
        competitor_texts = [text.strip() for text in competitor_input.split('---') if text.strip()]
        
        with st.spinner("Running Semantic Gap Analysis..."):
            gap_candidates = semantic_gap_analyzer(competitor_texts, target_input, n_value=n_value, top_n=top_n)
        st.session_state["gap_candidates"] = gap_candidates
        
        if gap_candidates:
            df_gap = pd.DataFrame(gap_candidates, columns=["n‑gram", "Gap Score"])
            st.subheader("Semantic Gap Analysis Results")
            st.dataframe(df_gap)
        else:
            st.warning("No gap candidates found.")
        
        with st.spinner("Running Keyword Clustering..."):
            clusters, embeddings, labels = keyword_clustering(gap_candidates)
        st.session_state["clusters"] = clusters
        
        if clusters and embeddings is not None and labels is not None:
            st.subheader("Keyword Clusters")
            cluster_data = []
            for cluster, items in clusters.items():
                for ngram, score in items:
                    cluster_data.append({"Cluster": f"Cluster {cluster}", "n‑gram": ngram, "Gap Score": score})
            df_clusters = pd.DataFrame(cluster_data)
            st.dataframe(df_clusters)
            
            gap_ngrams = [ngram for ngram, _ in gap_candidates]
            fig_clusters = plot_keyword_clusters(embeddings, labels, gap_ngrams)
            st.plotly_chart(fig_clusters)
        else:
            st.warning("No keyword clusters generated.")
    
    st.header("ChatGPT SEO Recommendations")
    
    # Persist API key in session state
    if "api_key" not in st.session_state:
        st.session_state["api_key"] = ""
    api_key = st.text_input("Enter your OpenAI API Key:", type="password", value=st.session_state["api_key"])
    st.session_state["api_key"] = api_key
    
    if st.button("Get Recommendations"):
        if not api_key:
            st.error("Please enter your API key.")
            return
        
        if "gap_candidates" not in st.session_state or "clusters" not in st.session_state:
            st.error("Please run the analysis first.")
            return
        
        gap_candidates = st.session_state["gap_candidates"]
        clusters = st.session_state["clusters"]
        
        with st.spinner("Generating recommendations using ChatGPT..."):
            recommendations = get_recommendations(gap_candidates, clusters, api_key)
        st.subheader("SEO Recommendations")
        st.write(recommendations)

if __name__ == "__main__":
    if "gap_candidates" not in st.session_state:
        st.session_state["gap_candidates"] = []
    if "clusters" not in st.session_state:
        st.session_state["clusters"] = {}
    main()


