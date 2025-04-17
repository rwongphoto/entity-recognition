import os
import streamlit as st
from streamlit import cache_resource  # Add this line if it's not there
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
import spacy
import collections
from collections import Counter
from typing import List, Tuple, Dict
import textstat

from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.cluster import KMeans, AgglomerativeClustering  # Keep KMeans

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, WebDriverException, NoSuchElementException

# Import SentenceTransformer from sentence_transformers
from sentence_transformers import SentenceTransformer

# NEW IMPORTS for GSC Update
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer  # Still potentially useful elsewhere

# NEW Imports for GSC Update (KMeans/GPT)
from openai import OpenAI
from sklearn.metrics import silhouette_score
import math  # For ceiling function

import plotly.graph_objects as go
import plotly.figure_factory as ff

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

import random  # Import the random module

# NEW: Import SPARQLWrapper for querying Wikidata
from SPARQLWrapper import SPARQLWrapper, JSON

# NEW IMPORT: Import Hugging Face transformers for BERT-based NER
from transformers import pipeline

import seaborn as sns

# NEW IMPORT: Import UMAP for dimension reduction
import umap

import networkx as nx

from urllib.parse import urlparse

# --- Set Hugging Face Token from Streamlit Secrets ---
if "huggingface" in st.secrets and "api_token" in st.secrets["huggingface"]:
    hf_token = st.secrets["huggingface"]["api_token"]
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    print("Hugging Face token set from Streamlit secrets.")
else:
    print("Hugging Face token not found in Streamlit secrets. Proceeding without token.")

# --- Set OpenAI Token from Streamlit Secrets (for GSC Analyzer) ---
if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
    print("OpenAI API key found in Streamlit secrets.")
else:
    print("OpenAI API key not found in Streamlit secrets. GSC GPT labeling will be disabled.")

# ------------------------------------
# Global Variables & Utility Functions
# ------------------------------------
logo_url = "https://theseoconsultant.ai/wp-content/uploads/2024/12/cropped-theseoconsultant-logo-2.jpg"

REQUEST_INTERVAL = 2.0
last_request_time = 0

def enforce_rate_limit():
    global last_request_time
    now = time.time()
    elapsed = now - last_request_time
    if elapsed < REQUEST_INTERVAL:
        time.sleep(REQUEST_INTERVAL - elapsed)
    last_request_time = time.time()

nlp = None

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.2151.97",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPad; CPU OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Android 14; Mobile; rv:121.0) Gecko/121.0 Firefox/121.0",
    "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
]

def get_random_user_agent():
    """Returns a randomly selected user agent from the list."""
    return random.choice(USER_AGENTS)

@st.cache_resource
def load_spacy_model():
    global nlp
    if nlp is None:
        try:
            nlp = spacy.load("en_core_web_lg")
            print("spaCy model loaded successfully")
        except OSError:
            print("Downloading en_core_web_lg model...")
            spacy.cli.download("en_core_web_lg")
            nlp = spacy.load("en_core_web_lg")
            print("en_core_web_lg downloaded and loaded")
        except Exception as e:
            st.error(f"Failed to load spaCy model: {e}")
            return None
    return nlp

@st.cache_resource
def initialize_sentence_transformer():
    print("Attempting to load Sentence Transformer model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("Sentence Transformer model loaded successfully.")
    return model

def get_embedding(text, model):
    return model.encode(text)

@st.cache_data(ttl=86400)
def extract_text_from_url(url):
    try:
        enforce_rate_limit()
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        user_agent = get_random_user_agent()
        chrome_options.add_argument(f"user-agent={user_agent}")
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        wait = WebDriverWait(driver, 20)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "body")))
        page_source = driver.page_source
        driver.quit()
        soup = BeautifulSoup(page_source, "html.parser")
        body = soup.find('body')
        if not body:
            return None
        for tag in body.find_all(['header', 'footer']):
            tag.decompose()
        text = body.get_text(separator='\n', strip=True)
        return text
    except (TimeoutException, WebDriverException) as e:
        error_str = str(e)
        if "HTTPConnectionPool" in error_str or "timed out" in error_str:
            st.error(f"Timeout error fetching {url}: {error_str}. Please check your network connection or increase the timeout setting.")
        else:
            st.error(f"Selenium error fetching {url}: {error_str}")
        return None
    except Exception as e:
        error_str = str(e)
        if "HTTPConnectionPool" in error_str or "timed out" in error_str:
            st.error(f"HTTP Connection Timeout while fetching {url}: {error_str}. Please check your network connection or increase the timeout setting.")
        else:
            st.error(f"Unexpected error fetching {url}: {error_str}")
        return None

@st.cache_data(ttl=86400)
def extract_relevant_text_from_url(url):
    try:
        enforce_rate_limit()
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        user_agent = get_random_user_agent()
        chrome_options.add_argument(f"user-agent={user_agent}")
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        wait = WebDriverWait(driver, 20)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "body")))
        page_source = driver.page_source
        driver.quit()
        soup = BeautifulSoup(page_source, "html.parser")
        for tag in soup.find_all(["header", "footer"]):
            tag.decompose()
        tags = []
        tags.extend(soup.find_all("p"))
        tags.extend(soup.find_all("ol"))
        tags.extend(soup.find_all("ul"))
        for header in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            tags.extend(soup.find_all(header))
        tags.extend(soup.find_all("table"))
        texts = [tag.get_text(separator=" ", strip=True) for tag in tags]
        return " ".join(texts)
    except Exception as e:
        error_str = str(e)
        if "HTTPConnectionPool" in error_str or "timed out" in error_str:
            st.error(f"HTTP Connection Timeout while extracting relevant content from {url}: {error_str}. Please check your network connection or increase the timeout setting.")
        else:
            st.error(f"Error extracting relevant content from {url}: {error_str}")
        return None

@st.cache_data
def count_videos(_soup):
    video_count = len(_soup.find_all("video"))
    iframe_videos = len([
        iframe for iframe in _soup.find_all("iframe")
        if any(domain in (iframe.get("src") or "") for domain in ["youtube.com", "youtube-nocookie.com", "vimeo.com"])
    ])
    return video_count + iframe_videos

def preprocess_text(text, nlp_model):
    doc = nlp_model(text)
    lemmatized_tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
    return " ".join(lemmatized_tokens)

def create_navigation_menu(logo_url):
    menu_options = {
        "Home": "https://theseoconsultant.ai/",
        "About": "https://theseoconsultant.ai/about/",
        "Services": "https://theseoconsultant.ai/seo-services/",
        "Blog": "https://theseoconsultant.ai/blog/",
        "Contact": "https://theseoconsultant.ai/contact/"
    }
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center;">
            <img src="{logo_url}" width="350">
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <style>
        .topnav {
          overflow: hidden;
          background-color: #f1f1f1;
          display: flex;
          justify-content: center;
          margin-bottom: 35px;
        }
        .topnav a {
          float: left;
          display: block;
          color: black;
          text-align: center;
          padding: 14px 16px;
          text-decoration: none;
        }
        .topnav a:hover {
          background-color: #ddd;
          color: black;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    menu_html = "<div class='topnav'>"
    for key, value in menu_options.items():
        menu_html += f"<a href='{value}' target='_blank'>{key}</a>"
    menu_html += "</div>"
    st.markdown(menu_html, unsafe_allow_html=True)

@st.cache_resource
def load_bert_ner_pipeline():
    return pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

def identify_entities(text, nlp_model):
    ner_pipe = load_bert_ner_pipeline()
    bert_entities = ner_pipe(text)
    entities = []
    for ent in bert_entities:
        entity_text = ent["word"].strip()
        entity_label = ent["entity_group"]
        entities.append((entity_text, entity_label))
    return entities

def count_entities(entities: List[Tuple[str, str]], nlp_model) -> Counter:
    entity_counts = Counter()
    seen_entities = set()
    for entity, label in entities:
        entity = entity.replace('\n', ' ').replace('\r', '')
        if len(entity) > 2:
            doc = nlp_model(entity)
            lemma = " ".join([token.lemma_ for token in doc])
            if (lemma, label) not in seen_entities:
                entity_counts[(lemma, label)] += 1
                seen_entities.add((lemma, label))
    return entity_counts

def count_entities_total(entities: List[Tuple[str, str]], nlp_model) -> Counter:
    entity_counts = Counter()
    for entity, label in entities:
        entity = entity.replace('\n', ' ').replace('\r', '')
        if len(entity) > 2:
            doc = nlp_model(entity)
            lemma = " ".join([token.lemma_ for token in doc])
            entity_counts[(lemma, label)] += 1
    return entity_counts

def display_entity_barchart(entity_counts, top_n=30):
    filtered_entity_counts = {
        (entity, label): count
        for (entity, label), count in entity_counts.items()
        if label != "CARDINAL"
    }
    entity_data = pd.DataFrame.from_dict(filtered_entity_counts, orient='index', columns=['count'])
    entity_data.index.names = ['entity']
    entity_data = entity_data.sort_values('count', ascending=False).head(top_n).reset_index()
    entity_names = [e[0] for e in entity_data['entity']]
    counts = entity_data['count']
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(entity_names, counts)
    ax.set_xlabel("Entities")
    ax.set_ylabel("Frequency")
    ax.set_title("Entity Frequency")
    plt.xticks(rotation=45, ha="right")
    for bar, count in zip(bars, counts):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, int(count), ha='center', va='bottom')
    plt.tight_layout()
    st.pyplot(fig)

def display_entity_wordcloud(entity_counts):
    aggregated = {}
    for key, count in entity_counts.items():
        k = key[0] if isinstance(key, tuple) else key
        aggregated[k] = aggregated.get(k, 0) + count
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(aggregated)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout()
    st.pyplot(fig)

nltk.download('stopwords')
stop_words_nltk = set(nltk.corpus.stopwords.words('english'))

def generate_topic_label(queries_in_topic):
    words = []
    for query in queries_in_topic:
        tokens = query.lower().split()
        filtered = [t for t in tokens if t not in stop_words_nltk]
        words.extend(filtered)
    if words:
        freq = collections.Counter(words)
        common = freq.most_common(2)
        label = ", ".join([word for word, count in common])
        return label.capitalize()
    else:
        return "N/A"

def calculate_overall_similarity(urls, search_term, model):
    search_term_embedding = get_embedding(search_term, model)
    results = []
    for url in urls:
        text = extract_text_from_url(url)
        if text:
            text_embedding = get_embedding(text, model)
            similarity = cosine_similarity([text_embedding], [search_term_embedding])[0][0]
            results.append((url, similarity))
        else:
            results.append((url, None))
    return results

def calculate_similarity(text, search_term, model):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_embeddings = [get_embedding(sentence, model) for sentence in sentences]
    search_term_embedding = get_embedding(search_term, model)
    similarities = [cosine_similarity([emb], [search_term_embedding])[0][0] for emb in sentence_embeddings]
    return sentences, similarities

def rank_sentences_by_similarity(text, search_term):
    model = initialize_sentence_transformer()
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_embeddings = [get_embedding(sentence, model) for sentence in sentences]
    search_term_embedding = get_embedding(search_term, model)
    similarities = [cosine_similarity([emb], [search_term_embedding])[0][0] for emb in sentence_embeddings]
    min_similarity = min(similarities)
    max_similarity = max(similarities)
    normalized_similarities = ([0.0] * len(similarities) if max_similarity == min_similarity
                               else [(s - min_similarity) / (max_similarity - min_similarity) for s in similarities])
    return list(zip(sentences, normalized_similarities))

def highlight_text(text, search_term):
    sentences_with_similarity = rank_sentences_by_similarity(text, search_term)
    highlighted_text = ""
    for sentence, similarity in sentences_with_similarity:
        color = "red" if similarity < 0.35 else "green" if similarity >= 0.65 else "black"
        highlighted_text += f'<p style="color:{color};">{sentence}</p>'
    return highlighted_text

def rank_sections_by_similarity_bert(text, search_term, top_n=10):
    model = initialize_sentence_transformer()
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_embeddings = [get_embedding(sentence, model) for sentence in sentences]
    search_term_embedding = get_embedding(search_term, model)
    similarities = [cosine_similarity([emb], [search_term_embedding])[0][0] for emb in sentence_embeddings]
    section_scores = list(zip(sentences, similarities))
    sorted_sections = sorted(section_scores, key=lambda item: item[1], reverse=True)
    top_sections = sorted_sections[:top_n]
    bottom_sections = sorted_sections[-top_n:]
    return top_sections, bottom_sections

@st.cache_data(ttl=86400)
def get_wikidata_link(entity_name: str) -> str:
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    safe_entity = entity_name.replace('"', '\\"')
    query = f"""
    SELECT ?item WHERE {{
      ?item rdfs:label "{safe_entity}"@en.
    }} LIMIT 1
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])
        if bindings:
            item_url = bindings[0]["item"]["value"]
            return item_url
    except Exception as e:
        st.error(f"Error querying Wikidata for '{entity_name}': {e}")
    return None

@st.cache_resource
def get_openai_client():
    api_key = None
    if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
        api_key = st.secrets["openai"]["api_key"]
    else:
        print("OpenAI API key not found in Streamlit Secrets.")
        return None

    if api_key:
        try:
            client = OpenAI(api_key=api_key)
            print("OpenAI client initialized successfully.")
            return client
        except Exception as e:
            st.error(f"Failed to initialize OpenAI client: {e}")
            return None
    else:
        return None

@st.cache_data(ttl=3600)
def get_gpt_cluster_label(_client, queries_in_cluster: list, cluster_id: int) -> str:
    if not isinstance(_client, OpenAI):
        return f"Cluster {cluster_id + 1} (No API Key)"

    sample_queries = list(np.random.choice(queries_in_cluster, size=min(15, len(queries_in_cluster)), replace=False))
    if not sample_queries:
        return f"Cluster {cluster_id + 1} (Empty)"

    prompt = (
        "The following search queries represent related user searches: " +
        ", ".join(sample_queries) +
        ". Give a short 3–5 word topic label in Title Case for this group."
    )
    try:
        response = _client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert SEO assistant. Summarize search query clusters into concise 3–5 word topic labels, using Title Case."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=20,
            timeout=20.0,
        )
        topic = response.choices[0].message.content.strip().replace('"', '')
        topic = re.sub(r"^(Topic|Label|Cluster) ?\d*: ?", "", topic, flags=re.IGNORECASE).strip()
        return topic if topic else f"Cluster {cluster_id + 1} (GPT Failed)"
    except Exception as e:
        st.warning(f"❌ OpenAI API Error labeling cluster {cluster_id + 1}: {e}")
        return f"Cluster {cluster_id + 1} (GPT Error)"

def google_search_console_analysis_page():
    st.header("Google Search Console Data Analysis")
    st.markdown(
        """
        Compare GSC query data from two periods to identify performance changes.
        This tool now uses **KMeans clustering** on query embeddings (SentenceTransformer) and **GPT-based labeling** to group queries into topics.
        Upload CSV files (one for the 'Before' period and one for the 'After' period), and the tool will:
        - Calculate overall performance changes (based on full input data).
        - Merge data using an outer join to preserve all queries.
        - Compute embeddings for each query.
        - Cluster queries using KMeans (optimal K suggested via Silhouette Score).
        - Generate descriptive topic labels for each cluster using OpenAI's GPT.
        - Display the original merged data table with GPT topic labels (includes queries unique to one period).
        - Aggregate metrics by topic.
        - Visualize the YOY % change by topic for each metric.
        **Note:** Requires an OpenAI API key set in Streamlit Secrets for topic labeling.
        """
    )

    st.markdown("### Upload GSC Data")
    uploaded_file_before = st.file_uploader("Upload GSC CSV for 'Before' period", type=["csv"], key="gsc_before")
    uploaded_file_after = st.file_uploader("Upload GSC CSV for 'After' period", type=["csv"], key="gsc_after")

    if uploaded_file_before is not None and uploaded_file_after is not None:
        # --- NEW: clustering mode selector ---
        st.markdown("### Clustering Method")
        mode = st.radio(
            "Choose how to group queries:",
            ("Default (Embedding + KMeans)", "NER‑based")
        )

        openai_client = get_openai_client()
        progress_bar = st.progress(0)
        status_text = st.empty()
        try:
            status_text.text("Reading, Cleaning, Initial Metrics...")
            df_before = pd.read_csv(uploaded_file_before)
            df_after = pd.read_csv(uploaded_file_after)
            required_query_col = "Top queries"
            required_pos_col = "Position"
            def find_col_name(df, potential_names):
                for name in potential_names:
                    for col in df.columns:
                        if col.strip().lower() == name.strip().lower():
                            return col
                return None
            query_col_before = find_col_name(df_before, [required_query_col, "Query"])
            pos_col_before   = find_col_name(df_before, [required_pos_col, "Average position", "Position"])
            query_col_after  = find_col_name(df_after,  [required_query_col, "Query"])
            pos_col_after    = find_col_name(df_after,  [required_pos_col, "Average position", "Position"])
            clicks_col_before      = find_col_name(df_before, ["Clicks"])
            impressions_col_before= find_col_name(df_before, ["Impressions", "Impr."])
            ctr_col_before         = find_col_name(df_before, ["CTR"])
            clicks_col_after       = find_col_name(df_after,  ["Clicks"])
            impressions_col_after = find_col_name(df_after,  ["Impressions", "Impr."])
            ctr_col_after          = find_col_name(df_after,  ["CTR"])
            if not query_col_before or not pos_col_before or not query_col_after or not pos_col_after:
                st.error("Query/Position column missing in one of the files.")
                return
            rename_map_before = {query_col_before: "Query", pos_col_before: "Average Position"}
            rename_map_after  = {query_col_after:  "Query", pos_col_after:  "Average Position"}
            if clicks_col_before:       rename_map_before[clicks_col_before]       = "Clicks"
            if impressions_col_before:  rename_map_before[impressions_col_before]  = "Impressions"
            if ctr_col_before:          rename_map_before[ctr_col_before]          = "CTR"
            if clicks_col_after:        rename_map_after[clicks_col_after]         = "Clicks"
            if impressions_col_after:   rename_map_after[impressions_col_after]    = "Impressions"
            if ctr_col_after:           rename_map_after[ctr_col_after]           = "CTR"
            df_before = df_before.rename(columns=rename_map_before)
            df_after  = df_after.rename(columns=rename_map_after)

            def clean_metric(series):
                if pd.api.types.is_numeric_dtype(series):
                    return series
                series_str = series.astype(str)
                cleaned = series_str.str.replace('%', '', regex=False).str.replace('<|>|,', '', regex=True).str.strip()
                cleaned = cleaned.replace('', np.nan).replace('N/A', np.nan).replace('--', np.nan)
                return pd.to_numeric(cleaned, errors='coerce')

            potential_metrics = ["Average Position", "Clicks", "Impressions", "CTR"]
            df_before_cleaned = df_before.copy()
            df_after_cleaned  = df_after.copy()
            for df in [df_before_cleaned, df_after_cleaned]:
                for col in potential_metrics:
                    if col in df.columns:
                        df[col] = clean_metric(df[col])

            st.markdown("## Dashboard Summary")
            cols = st.columns(4)
            def calculate_weighted_average(values, weights):
                if values is None or weights is None:
                    return np.nan
                valid = values.notna() & weights.notna() & (weights > 0)
                if not valid.any():
                    return values.mean() if values.notna().any() else np.nan
                try:
                    return np.average(values[valid], weights=weights[valid])
                except ZeroDivisionError:
                    return values.mean() if values.notna().any() else np.nan

            if "Clicks" in df_before_cleaned.columns and "Clicks" in df_after_cleaned.columns:
                before_sum = df_before_cleaned["Clicks"].sum()
                after_sum  = df_after_cleaned["Clicks"].sum()
                diff = after_sum - before_sum
                pct  = (diff / before_sum * 100) if before_sum not in (0, np.nan) else 0
                cols[0].metric("Clicks Change", f"{diff:,.0f}", f"{pct:.1f}%")
            else:
                cols[0].metric("Clicks Change", "N/A")

            if "Impressions" in df_before_cleaned.columns and "Impressions" in df_after_cleaned.columns:
                before_sum = df_before_cleaned["Impressions"].sum()
                after_sum  = df_after_cleaned["Impressions"].sum()
                diff = after_sum - before_sum
                pct  = (diff / before_sum * 100) if before_sum not in (0, np.nan) else 0
                cols[1].metric("Impressions Change", f"{diff:,.0f}", f"{pct:.1f}%")
            else:
                cols[1].metric("Impressions Change", "N/A")

            avg_before = np.nan
            if "Average Position" in df_before_cleaned.columns and "Impressions" in df_before_cleaned.columns:
                avg_before = calculate_weighted_average(df_before_cleaned["Average Position"], df_before_cleaned["Impressions"])
            avg_after = np.nan
            if "Average Position" in df_after_cleaned.columns and "Impressions" in df_after_cleaned.columns:
                avg_after = calculate_weighted_average(df_after_cleaned["Average Position"], df_after_cleaned["Impressions"])
            if not np.isnan(avg_before) and not np.isnan(avg_after):
                change = avg_before - avg_after
                pct    = (change / avg_before * 100) if avg_before not in (0, np.nan) else 0
                cols[2].metric("Avg. Position Change", f"{change:.1f}", f"{pct:.1f}%", delta_color="inverse")
            else:
                cols[2].metric("Avg. Position Change", "N/A")

            ctr_before = np.nan
            if "CTR" in df_before_cleaned.columns and "Impressions" in df_before_cleaned.columns:
                ctr_before = calculate_weighted_average(df_before_cleaned["CTR"], df_before_cleaned["Impressions"])
            ctr_after = np.nan
            if "CTR" in df_after_cleaned.columns and "Impressions" in df_after_cleaned.columns:
                ctr_after = calculate_weighted_average(df_after_cleaned["CTR"], df_after_cleaned["Impressions"])
            if not np.isnan(ctr_before) and not np.isnan(ctr_after):
                change = ctr_after - ctr_before
                pct    = (change / ctr_before * 100) if ctr_before not in (0, np.nan) else 0
                cols[3].metric("Avg. CTR Change", f"{change:.2f}% pts", f"{pct:.1f}%")
            else:
                cols[3].metric("Avg. CTR Change", "N/A")

            progress_bar.progress(10)

            # --- Step 5: Merge Data using OUTER JOIN ---
            status_text.text("Merging data (Outer Join)...")
            cols_before = ["Query"] + [c for c in potential_metrics if c in df_before_cleaned.columns]
            cols_after  = ["Query"] + [c for c in potential_metrics if c in df_after_cleaned.columns]
            merged_df = pd.merge(
                df_before_cleaned[cols_before],
                df_after_cleaned[cols_after],
                on="Query",
                suffixes=("_before", "_after"),
                how="outer"
            )
            if merged_df.empty:
                st.error("Merge failed. No matching queries found.")
                return
            progress_bar.progress(15)

            # --- Step 6: Calculate YOY changes ---
            status_text.text("Calculating YOY changes...")
            def pct_change(abs_change, before):
                if pd.isna(abs_change) or pd.isna(before):
                    return np.nan
                if before == 0:
                    return np.inf if abs_change != 0 else 0.0
                return (abs_change / before) * 100

            if "Clicks_before" in merged_df and "Clicks_after" in merged_df:
                merged_df["Clicks_YOY"]     = merged_df["Clicks_after"]     - merged_df["Clicks_before"]
                merged_df["Clicks_YOY_pct"] = merged_df.apply(lambda r: pct_change(r["Clicks_YOY"],     r["Clicks_before"]), axis=1)
            if "Impressions_before" in merged_df and "Impressions_after" in merged_df:
                merged_df["Impressions_YOY"]     = merged_df["Impressions_after"]     - merged_df["Impressions_before"]
                merged_df["Impressions_YOY_pct"] = merged_df.apply(lambda r: pct_change(r["Impressions_YOY"], r["Impressions_before"]), axis=1)
            if "CTR_before" in merged_df and "CTR_after" in merged_df:
                merged_df["CTR_YOY"]     = merged_df["CTR_after"]     - merged_df["CTR_before"]
                merged_df["CTR_YOY_pct"] = merged_df.apply(lambda r: pct_change(r["CTR_YOY"], r["CTR_before"]), axis=1)
            if "Average Position_before" in merged_df and "Average Position_after" in merged_df:
                merged_df["Position_YOY"]     = merged_df["Average Position_before"] - merged_df["Average Position_after"]
                merged_df["Position_YOY_pct"] = merged_df.apply(lambda r: pct_change(r["Position_YOY"], r["Average Position_before"]), axis=1)

            progress_bar.progress(20)

            if mode == "Default (Embedding + KMeans)":
                # --- Step 7: Embeddings ---
                status_text.text("Computing query embeddings...")
                model = initialize_sentence_transformer()
                if model is None:
                    st.error("Sentence Transformer model failed to load.")
                    return

                queries = merged_df["Query"].astype(str).unique().tolist()
                if not queries:
                    st.error("No queries found in the merged data.")
                    return
                with st.spinner(f"Generating embeddings for {len(queries)} unique queries..."):
                    try:
                        embeddings_unique = model.encode(queries, show_progress_bar=True)
                    except Exception as e:
                        st.error(f"Embedding generation failed: {e}")
                        return

                query_to_emb = dict(zip(queries, embeddings_unique))
                merged_df["query_embedding"] = merged_df["Query"].map(query_to_emb)
                mask = merged_df["query_embedding"].notna()
                if not mask.any():
                    st.error("No valid embeddings generated.")
                    return
                matrix = np.vstack(merged_df.loc[mask, "query_embedding"].values)
                progress_bar.progress(35)

                # --- Step 8: Clustering ---
                status_text.text("Performing KMeans clustering...")
                n_items = matrix.shape[0]
                max_k = min(30, n_items - 1) if n_items > 1 else 1
                min_k = 3
                if max_k < min_k:
                    optimal = max_k if max_k > 0 else 1
                else:
                    sil_scores = {}
                    for k in range(min_k, max_k + 1):
                        try:
                            km_temp = KMeans(n_clusters=k, random_state=42, n_init='auto')
                            labels_temp = km_temp.fit_predict(matrix)
                            sil_scores[k] = silhouette_score(matrix, labels_temp)
                        except Exception:
                            continue
                    optimal = max(sil_scores, key=sil_scores.get) if sil_scores else max(min_k, math.ceil(n_items/50))
                    optimal = min(optimal, max_k)

                slider_min = max(1, min_k if n_items >= min_k else 1)
                slider_max = max(1, max_k)
                default_k = 10 if slider_max >= 10 else slider_max
                n_clusters = st.slider("Select number of query clusters (K):", slider_min, slider_max, default_k, key="kmeans_clusters_gsc")

                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
                labels = kmeans.fit_predict(matrix)
                merged_df.loc[mask, "Cluster_ID"] = labels
                merged_df["Cluster_ID"] = merged_df["Cluster_ID"].astype('Int64')
                progress_bar.progress(55)

                # --- Step 9: GPT Labeling ---
                status_text.text("Generating topic labels with GPT...")
                cluster_topics = {}
                valid_ids = merged_df["Cluster_ID"].dropna().unique()
                if openai_client:
                    gpt_bar = st.progress(0)
                    for i, cid in enumerate(sorted(valid_ids)):
                        qs = merged_df[merged_df["Cluster_ID"] == cid]["Query"].unique().tolist()
                        if qs:
                            cluster_topics[cid] = get_gpt_cluster_label(openai_client, qs, cid)
                        else:
                            cluster_topics[cid] = f"Cluster {cid + 1} (Empty)"
                        time.sleep(0.1)
                        gpt_bar.progress((i + 1) / len(valid_ids))
                    gpt_bar.empty()
                else:
                    st.warning("OpenAI client not initialized. Using default labels.")
                    for cid in sorted(valid_ids):
                        cluster_topics[cid] = f"Cluster {cid + 1}"

                cluster_topics[pd.NA] = "Unclustered / No Embedding"
                merged_df["Query_Topic"] = merged_df["Cluster_ID"].map(cluster_topics).fillna("Unclustered")
                progress_bar.progress(70)

            else:
                # --- NER‑based grouping branch ---
                status_text.text("Extracting entities via NER…")
                nlp = load_spacy_model()
                merged_df["Entities"] = merged_df["Query"].astype(str).apply(
                    lambda q: identify_entities(q, nlp)
                )
                def pick_topic(ents):
                    if not ents:
                        return "No Entity"
                    return max(ents, key=lambda x: len(x[0]))[0]
                merged_df["Query_Topic"] = merged_df["Entities"].apply(pick_topic)
                merged_df["Cluster_ID"] = merged_df["Query_Topic"]
                progress_bar.progress(70)

            # --- Display Merged Data Table ---
            st.markdown("### Combined Data with Topic Labels")
            st.markdown("Merged data (outer join) with cluster ID and topic labels.")
            display_cols = ["Query", "Cluster_ID", "Query_Topic"]
            metrics_ordered = ["Average Position", "Clicks", "Impressions", "CTR"]
            for metric in metrics_ordered:
                for suffix in ["_before", "_after", "_YOY", "_YOY_pct"]:
                    if metric == "Average Position":
                        col = {
                            "_before": "Average Position_before",
                            "_after":  "Average Position_after",
                            "_YOY":    "Position_YOY",
                            "_YOY_pct":"Position_YOY_pct"
                        }[suffix]
                    else:
                        col = f"{metric}{suffix}"
                    if col in merged_df.columns:
                        display_cols.append(col)
            df_display = merged_df[[c for c in display_cols if c in merged_df.columns]]
            fmt = {}
            def add_fmt(col, fstr):
                if col_name in merged_df_display.columns and pd.api.types.is_numeric_dtype(merged_df_display[col_name]):
                    fmt[col] = fstr
            add_fmt("Cluster_ID", "{:.0f}")
            add_fmt("Average Position_before", "{:.1f}")
            add_fmt("Average Position_after", "{:.1f}")
            add_fmt("Position_YOY", "{:+.1f}")
            add_fmt("Position_YOY_pct", "{:+.1f}%")
            add_fmt("Clicks_before", "{:,.0f}")
            add_fmt("Clicks_after", "{:,.0f}")
            add_fmt("Clicks_YOY", "{:+.0f}")
            add_fmt("Clicks_YOY_pct", "{:+.1f}%")
            add_fmt("Impressions_before", "{:,.0f}")
            add_fmt("Impressions_after", "{:,.0f}")
            add_fmt("Impressions_YOY", "{:+.0f}")
            add_fmt("Impressions_YOY_pct", "{:+.1f}%")
            add_fmt("CTR_before", "{:.2f}%")
            add_fmt("CTR_after", "{:.2f}%")
            add_fmt("CTR_YOY", "{:+.2f}%")
            add_fmt("CTR_YOY_pct", "{:+.1f}%")
            st.dataframe(df_display.style.format(fmt, na_rep="N/A"))

            # --- Step 10: Aggregated Metrics by Topic ---
            status_text.text("Aggregating metrics by topic...")
            agg_map = {}
            if "Average Position_before" in merged_df and "Impressions_before" in merged_df:
                agg_map["Average Position_before"] = lambda x: calculate_weighted_average(x, merged_df.loc[x.index, "Impressions_before"])
            elif "Average Position_before" in merged_df:
                agg_map["Average Position_before"] = "mean"
            if "Average Position_after" in merged_df and "Impressions_after" in merged_df:
                agg_map["Average Position_after"] = lambda x: calculate_weighted_average(x, merged_df.loc[x.index, "Impressions_after"])
            elif "Average Position_after" in merged_df:
                agg_map["Average Position_after"] = "mean"
            if "Clicks_before" in merged_df:
                agg_map["Clicks_before"] = "sum"
            if "Clicks_after" in merged_df:
                agg_map["Clicks_after"] = "sum"
            if "Impressions_before" in merged_df:
                agg_map["Impressions_before"] = "sum"
            if "Impressions_after" in merged_df:
                agg_map["Impressions_after"] = "sum"
            if "CTR_before" in merged_df and "Impressions_before" in merged_df:
                agg_map["CTR_before"] = lambda x: calculate_weighted_average(x, merged_df.loc[x.index, "Impressions_before"])
            elif "CTR_before" in merged_df:
                agg_map["CTR_before"] = "mean"
            if "CTR_after" in merged_df and "Impressions_after" in merged_df:
                agg_map["CTR_after"] = lambda x: calculate_weighted_average(x, merged_df.loc[x.index, "Impressions_after"])
            elif "CTR_after" in merged_df:
                agg_map["CTR_after"] = "mean"

            aggregated = merged_df.groupby("Query_Topic").agg(agg_map).reset_index().rename(columns={"Query_Topic":"Topic"})
            if "Average Position_before" in aggregated and "Average Position_after" in aggregated:
                aggregated["Position_YOY"] = aggregated["Average Position_before"] - aggregated["Average Position_after"]
            if "Clicks_before" in aggregated and "Clicks_after" in aggregated:
                aggregated["Clicks_YOY"] = aggregated["Clicks_after"] - aggregated["Clicks_before"]
            if "Impressions_before" in aggregated and "Impressions_after" in aggregated:
                aggregated["Impressions_YOY"] = aggregated["Impressions_after"] - aggregated["Impressions_before"]
            if "CTR_before" in aggregated and "CTR_after" in aggregated:
                aggregated["CTR_YOY"] = aggregated["CTR_after"] - aggregated["CTR_before"]

            if "Position_YOY" in aggregated:
                aggregated["Position_YOY_pct"] = aggregated.apply(lambda r: pct_change(r["Position_YOY"], r["Average Position_before"]), axis=1)
            if "Clicks_YOY" in aggregated:
                aggregated["Clicks_YOY_pct"] = aggregated.apply(lambda r: pct_change(r["Clicks_YOY"], r["Clicks_before"]), axis=1)
            if "Impressions_YOY" in aggregated:
                aggregated["Impressions_YOY_pct"] = aggregated.apply(lambda r: pct_change(r["Impressions_YOY"], r["Impressions_before"]), axis=1)
            if "CTR_YOY" in aggregated:
                aggregated["CTR_YOY_pct"] = aggregated.apply(lambda r: pct_change(r["CTR_YOY"], r["CTR_before"]), axis=1)

            progress_bar.progress(85)

            # Reorder columns
            cols_order = ["Topic"]
            yoy_cols = []
            for metric in ["Average Position", "Clicks", "Impressions", "CTR"]:
                if metric == "Average Position":
                    before, after, yoy, yoy_pct = "Average Position_before", "Average Position_after", "Position_YOY", "Position_YOY_pct"
                else:
                    before, after, yoy, yoy_pct = f"{metric}_before", f"{metric}_after", f"{metric}_YOY", f"{metric}_YOY_pct"
                for c in (before, after, yoy, yoy_pct):
                    if c in aggregated.columns:
                        cols_order.append(c)
                        if c.endswith("_YOY_pct"):
                            yoy_cols.append((c, metric))
            aggregated = aggregated[[c for c in cols_order if c in aggregated.columns]]

            fmt_agg = {}
            def add_agg_format(col_name, fmt_str):
                if col_name in aggregated.columns \
                   and pd.api.types.is_numeric_dtype(aggregated[col_name]):
                    format_dict_agg[col_name] = fmt_str
            add_fmt_agg("Average Position_before", "{:.1f}")
            add_fmt_agg("Average Position_after", "{:.1f}")
            add_fmt_agg("Position_YOY", "{:+.1f}")
            add_fmt_agg("Position_YOY_pct", "{:+.1f}%")
            add_fmt_agg("Clicks_before", "{:,.0f}")
            add_fmt_agg("Clicks_after", "{:,.0f}")
            add_fmt_agg("Clicks_YOY", "{:+.0f}")
            add_fmt_agg("Clicks_YOY_pct", "{:+.1f}%")
            add_fmt_agg("Impressions_before", "{:,.0f}")
            add_fmt_agg("Impressions_after", "{:,.0f}")
            add_fmt_agg("Impressions_YOY", "{:+.0f}")
            add_fmt_agg("Impressions_YOY_pct", "{:+.1f}%")
            add_fmt_agg("CTR_before", "{:.2f}%")
            add_fmt_agg("CTR_after", "{:.2f}%")
            add_fmt_agg("CTR_YOY", "{:+.2f}%")
            add_fmt_agg("CTR_YOY_pct", "{:+.1f}%")

            display_count = st.number_input("Number of aggregated topics to display:", min_value=1, value=min(aggregated.shape[0],50), max_value=aggregated.shape[0])
            sort_col = "Impressions_after" if "Impressions_after" in aggregated.columns else "Topic"
            aggregated_sorted = aggregated.sort_values(by=sort_col, ascending=False, na_position='last')
            st.dataframe(aggregated_sorted.head(display_count).style.format(fmt_agg, na_rep="N/A"))

            progress_bar.progress(90)

            # --- Step 11: Visualization ---
            status_text.text("Generating visualizations...")
            st.markdown("### YOY % Change by Topic for Each Metric")
            defaults = [t for t in aggregated["Topic"].unique() if t != "Unclustered"]
            topics = aggregated["Topic"].unique().tolist()
            if not defaults and topics:
                defaults = topics
            sel = st.multiselect("Select topics to display on the chart:", topics, default=defaults)
            vis = []
            if not yoy_cols:
                st.warning("Could not determine which YOY % columns to plot.")
            else:
                for _, row in aggregated_sorted.iterrows():
                    t = row["Topic"]
                    if t not in sel:
                        continue
                    for col, mname in yoy_cols:
                        val = row[col]
                        if pd.notna(val) and np.isfinite(val):
                            vis.append({"Topic": t, "Metric": mname, "YOY % Change": val})
            if vis:
                vis_df = pd.DataFrame(vis)
                vis_df["Metric"] = pd.Categorical(
                    vis_df["Metric"],
                    categories=["Clicks", "Impressions", "Average Position", "CTR"],
                    ordered=True
                )
                order_topics = sorted([t for t in aggregated_sorted["Topic"] if t in sel])
                fig = px.bar(
                    vis_df,
                    x="Topic",
                    y="YOY % Change",
                    color="Metric",
                    barmode="group",
                    category_orders={"Topic": order_topics, "Metric": ["Clicks", "Impressions", "Average Position", "CTR"]},
                    labels={"YOY % Change": "YOY Change (%)", "Topic": "Topic"}
                )
                fig.update_layout(height=600, yaxis_title="YOY Change (%)", legend_title_text="Metric")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No valid YOY % change data available to plot for the selected topics.")

            progress_bar.progress(100)
            status_text.text("Analysis Complete!")

        except FileNotFoundError:
            st.error("Error: CSV file not found.")
        except pd.errors.EmptyDataError:
            st.error("Error: CSV file is empty.")
        except KeyError as e:
            st.error(f"Error: Column mismatch: {e}.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
        finally:
            progress_bar.empty()
            status_text.empty()
    else:
        st.info("Please upload both GSC CSV files.")

def main():
    st.set_page_config(
        page_title="Semantic Search SEO Analysis Tools | The SEO Consultant.ai",
        page_icon="✏️",
        layout="wide"
    )
    hide_streamlit_elements = """
        <style>
        #MainMenu {visibility: hidden !important;}
        header {visibility: hidden !important;}
        footer {visibility: hidden !important;}
        .stApp > header > div:nth-child(3) { display: none !important; }
        button[title="View fullscreen"] { display: none !important; }
        div[data-testid="stToolbar"] { display: none !important; }
        div[data-testid="stDecoration"] { display: none !important; }
        a[href*='streamlit.io/cloud'],
        div[class*='_profileContainer'],
        div[class*='stActionButton'] > button[title*='Manage app'],
        div[data-testid="stStatusWidget"] { display: none !important; visibility: hidden !important; }
        div.block-container {padding-top: 1rem;}
        </style>
        """
    st.markdown(hide_streamlit_elements, unsafe_allow_html=True)
    create_navigation_menu(logo_url)
    st.sidebar.header("Semantic Search SEO Analysis Tools")
    tool = st.sidebar.selectbox("Select Tool:", ["Google Search Console Analyzer"])
    if tool == "Google Search Console Analyzer":
        google_search_console_analysis_page()
    st.markdown("---")
    st.markdown("<div style='text-align: center; margin-top: 20px;'>Powered by <a href='https://theseoconsultant.ai' target='_blank'>The SEO Consultant.ai</a></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

    print("NLTK checks complete. Starting Streamlit app...")
    main()

