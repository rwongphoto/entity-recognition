import os
import streamlit as st
from streamlit import cache_resource # Add this line if it's not there
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
from sklearn.cluster import KMeans, AgglomerativeClustering # Keep KMeans

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
# from sklearn.decomposition import LatentDirichletAllocation # No longer needed for GSC
from sklearn.feature_extraction.text import CountVectorizer # Still potentially useful elsewhere

# NEW Imports for GSC Update (KMeans/GPT)
from openai import OpenAI
from sklearn.cluster import KMeans # Already imported, but confirming
from sklearn.metrics import silhouette_score
import numpy as np # Already imported, but confirming
import math # For ceiling function

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
# Check if the secret exists before trying to access it
if "huggingface" in st.secrets and "api_token" in st.secrets["huggingface"]:
    hf_token = st.secrets["huggingface"]["api_token"]
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    print("Hugging Face token set from Streamlit secrets.") # Optional: for confirmation
else:
    print("Hugging Face token not found in Streamlit secrets. Proceeding without token.") # Optional

# --- Set OpenAI Token from Streamlit Secrets (for GSC Analyzer) ---
# Check if the secret exists before trying to access it
# Note: The get_openai_client function handles the actual check later
if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
    print("OpenAI API key found in Streamlit secrets.") # Optional: for confirmation
else:
    print("OpenAI API key not found in Streamlit secrets. GSC GPT labeling will be disabled.") # Optional

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

# User Agent List (Expanded)
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.2151.97", #Edge
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1", #Safari iPhone
    "Mozilla/5.0 (iPad; CPU OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1", #Safari iPad
    "Mozilla/5.0 (Android 14; Mobile; rv:121.0) Gecko/121.0 Firefox/121.0", #Firefox Android
    "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36", #Chrome Android
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
            print("en_core_web_md downloaded and loaded")
        except Exception as e:
            st.error(f"Failed to load spaCy model: {e}")
            return None
    return nlp

@st.cache_resource  # <-- ADD THIS DECORATOR
def initialize_sentence_transformer():
    """Loads the Sentence Transformer model using Streamlit's resource caching."""
    print("Attempting to load Sentence Transformer model...") # Optional: For debugging
    # It's slightly better practice to use the full repo name:
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("Sentence Transformer model loaded successfully.") # Optional: For debugging
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
        # Use the random user agent function
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
        # Use the random user agent function
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

# --------------------------------------------------
# NEW: Load BERT-based NER pipeline (using dslim/bert-base-NER)
# --------------------------------------------------
@st.cache_resource
def load_bert_ner_pipeline():
    # The aggregation_strategy="simple" groups tokens into complete entities.
    return pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

# --------------------------------------------------
# UPDATED: Use BERT for Named Entity Recognition
# --------------------------------------------------
def identify_entities(text, nlp_model):
    """
    Extracts named entities from the input text using a BERT-based NER pipeline.
    The 'nlp_model' parameter is retained for compatibility with the rest of the code,
    but is not used for entity extraction.
    Returns a list of tuples (entity_text, entity_label).
    """
    ner_pipe = load_bert_ner_pipeline()
    bert_entities = ner_pipe(text)
    entities = []
    for ent in bert_entities:
        # ent is a dict with keys like 'word' and 'entity_group'
        entity_text = ent["word"].strip()
        entity_label = ent["entity_group"]
        entities.append((entity_text, entity_label))
    return entities

# ORIGINAL count_entities (for unique counts per source)
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

# New function to count every occurrence
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

# This function is no longer used by GSC Analyzer, but might be used elsewhere
nltk.download('stopwords') #Ensure stopwords are downloaded if not already
stop_words_nltk = set(nltk.corpus.stopwords.words('english')) # Renamed to avoid clash

def generate_topic_label(queries_in_topic):
    words = []
    for query in queries_in_topic:
        tokens = query.lower().split()
        filtered = [t for t in tokens if t not in stop_words_nltk] # Use renamed variable
        words.extend(filtered)
    if words:
        freq = collections.Counter(words)
        common = freq.most_common(2) # Top 2 most common words
        label = ", ".join([word for word, count in common])
        return label.capitalize()
    else:
        return "N/A"

# ------------------------------------
# Cosine Similarity Functions
# ------------------------------------
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

# ------------------------------------
# NEW: Helper function using SPARQLWrapper to get Wikidata link for an entity
# ------------------------------------
@st.cache_data(ttl=86400)
def get_wikidata_link(entity_name: str) -> str:
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    # Escape quotes in the entity name
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

# ------------------------------------
# Streamlit UI Functions
# ------------------------------------
def url_analysis_dashboard_page():
    st.header("URL Analysis Dashboard")
    st.markdown("Analyze multiple URLs and gather key SEO metrics.")

    urls_input = st.text_area("Enter URLs (one per line):", key="dashboard_urls", value="")
    urls = [url.strip() for url in urls_input.splitlines() if url.strip()]

    search_term = st.text_input("Enter Search Term (for Cosine Similarity):", key="dashboard_search_term", value="")

    if st.button("Analyze URLs", key="dashboard_button"):
        if not urls:
            st.warning("Please enter at least one URL.")
            return

        with st.spinner("Analyzing URLs..."):
            nlp_model = load_spacy_model()
            model = initialize_sentence_transformer()
            data = []
            similarity_results = calculate_overall_similarity(urls, search_term, model)

            for i, url in enumerate(urls):
                try:
                    enforce_rate_limit() # Enforce rate limit before making request
                    chrome_options = Options()
                    chrome_options.add_argument("--headless")
                    chrome_options.add_argument("--no-sandbox")
                    chrome_options.add_argument("--disable-dev-shm-usage")
                    user_agent = get_random_user_agent()
                    chrome_options.add_argument(f"user-agent={user_agent}")
                    driver = webdriver.Chrome(options=chrome_options)
                    driver.get(url)
                    wait = WebDriverWait(driver, 20) # Added Wait
                    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "body"))) # Wait for body
                    page_source = driver.page_source
                    meta_title = driver.title  # Get meta title using Selenium
                    driver.quit()

                    soup = BeautifulSoup(page_source, "html.parser") # Use full soup for some elements

                    # Extract content text after removing header/footer
                    content_soup = BeautifulSoup(page_source, "html.parser")
                    total_text = ""
                    if content_soup.find("body"):
                         body = content_soup.find("body")
                         for tag in body.find_all(["header", "footer", "nav", "aside", "script", "style"]): # More rigorous removal
                             tag.decompose()
                         total_text = body.get_text(separator="\n", strip=True)
                    else:
                        total_text = "" # Handle case where body is not found

                    total_word_count = len(total_text.split())

                    # Extract "custom" content from specific tags for content word count
                    custom_words = []
                    if body: # Check if body exists before finding elements
                        custom_elements = body.find_all(["p", "li", "h1", "h2", "h3", "h4", "h5", "h6"])
                        for el in custom_elements:
                            custom_words.extend(el.get_text().split())
                        # Also include tables
                        for table in body.find_all("table"):
                            for row in table.find_all("tr"):
                                for cell in row.find_all(["td", "th"]):
                                    custom_words.extend(cell.get_text().split())
                    else:
                        # If no body, custom word count is 0
                        custom_words = []


                    # Ensure custom_word_count is never greater than total_word_count
                    custom_word_count = min(len(custom_words), total_word_count)

                    h1_tag = soup.find("h1").get_text(strip=True) if soup.find("h1") else "None"

                    # Find header/footer links from the original soup
                    header = soup.find("header")
                    footer = soup.find("footer")
                    header_links = len(header.find_all("a", href=True)) if header else 0
                    footer_links = len(footer.find_all("a", href=True)) if footer else 0
                    total_nav_links = header_links + footer_links
                    total_links = len(soup.find_all("a", href=True))

                    # Schema Markup (using original soup)
                    schema_types = set()
                    ld_json_scripts = soup.find_all("script", type="application/ld+json")
                    for script in ld_json_scripts:
                        try:
                            script_content = script.string
                            if script_content:
                                data_json = json.loads(script_content)
                                if isinstance(data_json, list):
                                    for item in data_json:
                                        if isinstance(item, dict) and "@type" in item:
                                            schema_types.add(item["@type"])
                                elif isinstance(data_json, dict):
                                    if "@type" in data_json:
                                        schema_types.add(data_json["@type"])
                        except Exception:
                            continue  # Handle JSON parsing errors

                    schema_markup = ", ".join(schema_types) if schema_types else "None"

                    # Lists/Tables (check within the filtered body if available)
                    lists_tables = (
                        f"OL: {'Yes' if body and body.find('ol') else 'No'} | "
                        f"UL: {'Yes' if body and body.find('ul') else 'No'} | "
                        f"Table: {'Yes' if body and body.find('table') else 'No'}"
                    )
                    num_images = len(soup.find_all("img")) # Count images from original soup
                    num_videos = count_videos(soup) # Count videos from original soup
                    similarity_val = similarity_results[i][1] if i < len(similarity_results) and similarity_results[i][1] is not None else np.nan
                    entities = identify_entities(total_text, nlp_model) if total_text and nlp_model else []
                    unique_entity_count = len(set([ent[0].lower().strip() for ent in entities])) # Lowercase for uniqueness
                    flesch_kincaid = textstat.flesch_kincaid_grade(total_text) if total_text else None

                    data.append([
                        url,
                        meta_title,
                        h1_tag,
                        total_word_count,
                        custom_word_count,
                        similarity_val,
                        unique_entity_count,
                        total_nav_links,
                        total_links,
                        schema_markup,
                        lists_tables,
                        num_images,
                        num_videos,
                        flesch_kincaid
                    ])

                except WebDriverException as e:
                     st.error(f"Selenium error processing URL {url}: {e}. Skipping.")
                     data.append([url] + ["Selenium Error"] * 13) # Append error placeholders
                except Exception as e:
                    st.error(f"Unexpected error processing URL {url}: {e}. Skipping.")
                    data.append([url] + ["Error"] * 13)  # Append error placeholders

            if not data:
                 st.warning("No data could be processed from the provided URLs.")
                 return

            df = pd.DataFrame(data, columns=[
                "URL",
                "Meta Title",
                "H1 Tag",
                "Total Word Count",
                "Content Word Count", # Renamed for clarity
                "Overall Cosine Similarity Score",
                "# of Unique Entities",
                "# of Header & Footer Links",
                "Total # of Links",
                "Schema Markup Types",
                "Lists/Tables Present",
                "# of Images",
                "# of Videos",
                "Flesch-Kincaid Grade Level"
            ])

            # Reorder for better presentation
            df = df[[
                "URL",
                "Meta Title",
                "H1 Tag",
                "Total Word Count",
                "Content Word Count", # Keep new name
                "Overall Cosine Similarity Score",
                "Flesch-Kincaid Grade Level",
                "# of Unique Entities",
                "# of Header & Footer Links",
                "Total # of Links",
                "Schema Markup Types",
                "Lists/Tables Present",
                "# of Images",
                "# of Videos",
            ]]

            # Rename columns for display
            df.columns = [
                "URL",
                "Meta Title",
                "H1",
                "Total Words", # Renamed
                "Content Words", # Renamed
                "Cosine Sim.", # Renamed
                "Grade Level",
                "# Entities", # Renamed
                "Nav Links",
                "Total Links",
                "Schema Types",
                "Lists/Tables",
                "Images",
                "Videos",
            ]

            # Convert columns to numeric where applicable, coercing errors
            numeric_cols_dash = ["Total Words", "Content Words", "Cosine Sim.", "Grade Level", "# Entities",
                                 "Nav Links", "Total Links", "Images", "Videos"]
            for col in numeric_cols_dash:
                 if col in df.columns:
                      df[col] = pd.to_numeric(df[col], errors='coerce')

            # Define format dictionary for styling
            format_dict_dash = {
                 "Cosine Sim.": "{:.3f}",
                 "Grade Level": "{:.1f}",
                 "Total Words": "{:,.0f}",
                 "Content Words": "{:,.0f}",
                 "# Entities": "{:,.0f}",
                 "Nav Links": "{:,.0f}",
                 "Total Links": "{:,.0f}",
                 "Images": "{:,.0f}",
                 "Videos": "{:,.0f}",
            }

            # Apply formatting and display
            st.dataframe(df.style.format(format_dict_dash, na_rep="N/A"))


def cosine_similarity_competitor_analysis_page():
    st.title("Cosine Similarity Competitor Analysis")
    st.markdown("By: [The SEO Consultant.ai](https://theseoconsultant.ai)")
    search_term = st.text_input("Enter Search Term:", "")
    source_option = st.radio("Select content source for competitors:", options=["Extract from URL", "Paste Content"], index=0)
    if source_option == "Extract from URL":
        urls_input = st.text_area("Enter Competitor URLs (one per line):", "")
        competitor_urls = [url.strip() for url in urls_input.splitlines() if url.strip()]
        competitor_contents = [] # Initialize empty list
    else:
        st.markdown("Paste the competitor content below. If you have multiple competitors, separate each content block with `---`.")
        pasted_content = st.text_area("Enter Competitor Content:", height=200)
        competitor_contents = [content.strip() for content in pasted_content.split('---') if content.strip()]
        competitor_urls = [] # Initialize empty list

    if st.button("Calculate Similarity"):
        model = initialize_sentence_transformer()
        if not search_term:
             st.warning("Please enter a search term.")
             return

        similarity_scores = []
        content_lengths = []
        competitor_labels = [] # To store URL or "Competitor X"

        if source_option == "Extract from URL":
            if not competitor_urls:
                st.warning("Please enter at least one URL.")
                return
            with st.spinner("Calculating similarities from URLs..."):
                # Use the existing function which handles errors
                similarity_results = calculate_overall_similarity(competitor_urls, search_term, model)

                for url, similarity in similarity_results:
                    competitor_labels.append(url)
                    if similarity is not None:
                        similarity_scores.append(similarity)
                        # Re-extract text to get length (or cache it if performance is an issue)
                        text = extract_text_from_url(url)
                        content_lengths.append(len(text.split()) if text else 0)
                    else:
                        similarity_scores.append(np.nan) # Append NaN if similarity failed
                        content_lengths.append(0)

        else:  # Paste Content
            if not competitor_contents:
                st.warning("Please paste at least one content block.")
                return
            with st.spinner("Calculating similarities from pasted content..."):
                search_embedding = get_embedding(search_term, model) # Embed search term once
                for idx, content in enumerate(competitor_contents):
                    label = f"Pasted Content {idx+1}"
                    competitor_labels.append(label)
                    if content:
                        text_embedding = get_embedding(content, model)
                        similarity = cosine_similarity([text_embedding], [search_embedding])[0][0]
                        similarity_scores.append(similarity)
                        content_lengths.append(len(content.split()))
                    else:
                        similarity_scores.append(np.nan)
                        content_lengths.append(0)

        # Create DataFrame
        df = pd.DataFrame({
            'Competitor': competitor_labels,
            'Cosine Similarity': similarity_scores,
            'Content Length (Words)': content_lengths
        })

        # Drop rows where similarity calculation failed
        df.dropna(subset=['Cosine Similarity'], inplace=True)

        if df.empty:
             st.warning("Could not calculate similarity for any competitors.")
             return

        # --- Option 1: 2D Scatter Plot ---
        st.subheader("Scatter Plot: Similarity vs. Content Length")
        fig_scatter = px.scatter(df, x='Cosine Similarity', y='Content Length (Words)',
                         title='Competitor Analysis: Similarity vs. Content Length',
                         hover_data=['Competitor', 'Cosine Similarity', 'Content Length (Words)'],
                         color='Cosine Similarity',
                         color_continuous_scale=px.colors.sequential.Viridis,
                         text='Competitor')
        fig_scatter.update_traces(textposition='top center')
        fig_scatter.update_layout(
            xaxis_title="Cosine Similarity (Higher = More Relevant)",
            yaxis_title="Content Length (Words)",
            width=800,
            height=600
        )
        st.plotly_chart(fig_scatter)

        # --- Option 2: Bar Chart with Secondary Y-Axis ---
        st.subheader("Bar Chart: Similarity and Content Length")
        df_sorted_bar = df.sort_values('Cosine Similarity', ascending=False)
        fig_bar = go.Figure(data=[
            go.Bar(name='Cosine Similarity', x=df_sorted_bar['Competitor'], y=df_sorted_bar['Cosine Similarity'],
                   marker_color=df_sorted_bar['Cosine Similarity'],
                   marker_colorscale='Viridis',
                   text=df_sorted_bar['Competitor'],
                   textposition='outside'),
            go.Scatter(name='Content Length', x=df_sorted_bar['Competitor'], y=df_sorted_bar['Content Length (Words)'], yaxis='y2',
                       mode='lines+markers', marker=dict(color='red'))
        ])
        fig_bar.update_traces(textfont_size=10) # Adjust text size
        fig_bar.update_layout(
            title='Competitor Analysis: Similarity and Content Length',
            xaxis_title="Competitor",
            yaxis_title="Cosine Similarity (Higher = More Relevant)",
            yaxis2=dict(title='Content Length (Words)', overlaying='y', side='right'),
            width=800,
            height=600,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis={'categoryorder':'array', 'categoryarray': df_sorted_bar['Competitor']} # Ensure bar order
        )
        st.plotly_chart(fig_bar)

        # --- Option 3: Bubble Chart ---
        st.subheader("Bubble Chart: Similarity vs. Content Length")
        fig_bubble = px.scatter(df, x='Cosine Similarity', y='Content Length (Words)',
                         size=[10] * len(df), # Constant bubble size or use another metric
                         title='Competitor Analysis: Similarity & Content Length (Bubble Chart)',
                         hover_data=['Competitor', 'Cosine Similarity', 'Content Length (Words)'],
                         color='Cosine Similarity',
                         color_continuous_scale=px.colors.sequential.Viridis,
                         text='Competitor')
        fig_bubble.update_traces(textposition='top center')
        fig_bubble.update_layout(
            xaxis_title="Cosine Similarity (Higher = More Relevant)",
            yaxis_title="Content Length (Words)",
            width=800,
            height=600
        )
        st.plotly_chart(fig_bubble)

        # --- Show Data Table ---
        st.subheader("Data Table")
        st.dataframe(df.style.format({"Cosine Similarity": "{:.4f}", "Content Length (Words)": "{:,.0f}"}))

def cosine_similarity_every_embedding_page():
    st.header("Cosine Similarity Score - Every Embedding")
    st.markdown("Calculates the cosine similarity score for each sentence in your input.")
    url = st.text_input("Enter URL (Optional):", key="every_embed_url", value="")
    use_url = st.checkbox("Use URL for Text Input", key="every_embed_use_url")
    text = st.text_area("Enter Text:", key="every_embed_text", value="", disabled=use_url)
    search_term = st.text_input("Enter Search Term:", key="every_embed_search", value="")
    if st.button("Calculate Similarity", key="every_embed_button"):
        if not search_term:
            st.warning("Please enter a search term.")
            return

        input_text_source = ""
        if use_url:
            if url:
                with st.spinner(f"Extracting text from {url}..."):
                    input_text_source = extract_text_from_url(url)
                    if not input_text_source:
                        st.error(f"Could not extract text from {url}. Please check the URL.")
                        return
            else:
                st.warning("Please enter a URL to extract the text.")
                return
        else: # Use pasted text
             input_text_source = text
             if not input_text_source:
                st.warning("Please enter text or provide a URL.")
                return

        model = initialize_sentence_transformer()
        with st.spinner("Calculating Similarities..."):
            sentences, similarities = calculate_similarity(input_text_source, search_term, model)

        if not sentences:
             st.warning("No sentences found in the input text.")
             return

        st.subheader("Similarity Scores:")
        results_df = pd.DataFrame({
            "Sentence": sentences,
            "Similarity": similarities
        })
        results_df = results_df.sort_values(by="Similarity", ascending=False).reset_index(drop=True)
        results_df.index = results_df.index + 1 # Start index at 1
        st.dataframe(results_df.style.format({"Similarity": "{:.4f}"}))


def cosine_similarity_content_heatmap_page():
    st.header("Cosine Similarity Content Heatmap")
    st.markdown("Green text is the most relevant to the search query. Red is the least relevant.")
    url = st.text_input("Enter URL (Optional):", key="heatmap_url", value="")
    use_url = st.checkbox("Use URL for Text Input", key="heatmap_use_url")
    input_text = st.text_area("Enter your text:", key="heatmap_input", height=300, value="", disabled=use_url)
    search_term = st.text_input("Enter your search term:", key="heatmap_search", value="")
    if st.button("Highlight", key="heatmap_button"):
        if not search_term:
            st.warning("Please enter a search term.")
            return

        text_to_highlight = ""
        if use_url:
            if url:
                with st.spinner(f"Extracting text from {url}..."):
                    text_to_highlight = extract_text_from_url(url)
                    if not text_to_highlight:
                        st.error(f"Could not extract text from {url}. Please check the URL.")
                        return
            else:
                st.warning("Please enter a URL to extract the text.")
                return
        else: # Use pasted text
            text_to_highlight = input_text
            if not text_to_highlight:
                st.error("Please enter text or provide a URL.")
                return

        with st.spinner("Generating highlighted text..."):
            highlighted_html = highlight_text(text_to_highlight, search_term) # Function now returns HTML

        if not highlighted_html:
             st.warning("Could not generate highlighted text.")
             return

        st.markdown("### Highlighted Content")
        st.markdown(highlighted_html, unsafe_allow_html=True)


def top_bottom_embeddings_page():
    st.header("Top & Bottom Embeddings")
    st.markdown("Identify the most and least semantically relevant sentences to your search query.")
    url = st.text_input("Enter URL (Optional):", key="tb_url", value="")
    use_url = st.checkbox("Use URL for Text Input", key="tb_use_url")
    text = st.text_area("Enter your text:", key="top_bottom_text", height=300, value="", disabled=use_url)
    search_term = st.text_input("Enter your search term:", key="top_bottom_search", value="")
    top_n = st.slider("Number of results (Top N & Bottom N):", min_value=1, max_value=25, value=10, key="top_bottom_slider")
    if st.button("Find Top/Bottom Sentences", key="top_bottom_button"):
        if not search_term:
            st.warning("Please enter a search term.")
            return

        input_text_source = ""
        if use_url:
            if url:
                with st.spinner(f"Extracting text from {url}..."):
                    input_text_source = extract_text_from_url(url)
                    if not input_text_source:
                        st.error(f"Could not extract text from {url}. Please check the URL.")
                        return
            else:
                st.warning("Please enter a URL or uncheck the box.")
                return
        else: # Use pasted text
            input_text_source = text
            if not input_text_source:
                st.error("Please enter text or provide a URL.")
                return

        # Note: rank_sections_by_similarity_bert already uses the SBERT model internally
        with st.spinner(f"Finding Top {top_n} and Bottom {top_n} sentences..."):
            top_sections, bottom_sections = rank_sections_by_similarity_bert(input_text_source, search_term, top_n)

        if not top_sections and not bottom_sections:
            st.warning("No sentences could be processed or ranked.")
            return

        st.subheader(f"Top {len(top_sections)} Sections (Highest Cosine Similarity):")
        if top_sections:
            df_top = pd.DataFrame(top_sections, columns=["Sentence", "Similarity"])
            df_top.index = df_top.index + 1
            st.dataframe(df_top.style.format({"Similarity": "{:.4f}"}))
        else:
            st.write("No top sections found.")

        st.subheader(f"Bottom {len(bottom_sections)} Sections (Lowest Cosine Similarity):")
        if bottom_sections:
             # Reverse bottom sections so the absolute lowest is first in the display list
            df_bottom = pd.DataFrame(reversed(bottom_sections), columns=["Sentence", "Similarity"])
            df_bottom.index = df_bottom.index + 1
            st.dataframe(df_bottom.style.format({"Similarity": "{:.4f}"}))
        else:
             st.write("No bottom sections found.")


def entity_analysis_page():
    st.header("Entity Topic Gap Analysis")
    st.markdown("Analyze multiple sources to identify common entities missing on your site, *and* unique entities on your site.")

    # Get competitor content
    st.markdown("#### Competitor Content")
    competitor_source_option = st.radio(
        "Select competitor content source:",
        options=["Extract from URL", "Paste Content"],
        index=0,
        key="entity_comp_source"
    )
    if competitor_source_option == "Extract from URL":
        competitor_input = st.text_area("Enter Competitor URLs (one per line):", key="entity_urls", value="")
        competitor_list = [url.strip() for url in competitor_input.splitlines() if url.strip()]
        competitor_content_blocks = [] # Initialize
    else:
        st.markdown("Paste competitor content below. For multiple sources, separate each with `---`.")
        competitor_input = st.text_area("Enter Competitor Content:", key="entity_competitor_text", value="", height=200)
        competitor_content_blocks = [content.strip() for content in competitor_input.split('---') if content.strip()]
        competitor_list = [] # Initialize

    # Get target content
    st.markdown("#### Target Site Content")
    target_option = st.radio(
        "Select target content source:",
        options=["Extract from URL", "Paste Content"],
        index=0,
        key="target_source_entity"
    )
    target_input = ""
    if target_option == "Extract from URL":
        target_input = st.text_input("Enter Target URL:", key="target_url_entity", value="")
    else:
        target_input = st.text_area("Paste target content:", key="target_text_entity", value="", height=100)

    # Exclude content
    st.markdown("#### Exclude Entities (Optional)")
    exclude_input = st.text_area("Paste text containing entities to exclude (e.g., brand name):", key="exclude_text_entity", value="", height=100)
    exclude_types = st.multiselect(
        "Select entity types to universally exclude:",
        options=["CARDINAL", "DATE", "TIME", "PERCENT", "MONEY",
                 "QUANTITY", "ORDINAL", "GPE", "ORG", "PERSON", "NORP",
                 "FAC", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART",
                 "LAW", "LANGUAGE", "MISC"],
        default=["CARDINAL", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL"]
    )

    if st.button("Analyze Entities", key="entity_button"):
        if not competitor_list and not competitor_content_blocks:
            st.warning("Please provide at least one competitor URL or content block.")
            return
        if not target_input:
            st.warning("Please provide target content or URL.")
            return

        with st.spinner("Analyzing entities..."):
            nlp_model = load_spacy_model()
            if not nlp_model:
                 st.error("Could not load spaCy model.")
                 return

            # --- Get Target Text ---
            target_text = ""
            if target_option == "Extract from URL":
                 if target_input:
                     with st.spinner(f"Extracting target text from {target_input}..."):
                         target_text = extract_text_from_url(target_input)
                 else:
                     st.warning("Target URL is empty.")
                     return
            else:
                 target_text = target_input

            if not target_text:
                 st.warning("Could not get target text.")
                 return

            # --- Get Excluded Entities ---
            excluded_entities_set = set()
            if exclude_input:
                 exclude_doc = nlp_model(exclude_input)
                 # Extract lemmas of excluded entities for broader matching
                 for ent in exclude_doc.ents:
                     lemma = " ".join([token.lemma_.lower() for token in nlp_model(ent.text)])
                     excluded_entities_set.add(lemma)
                 st.write(f"Excluding {len(excluded_entities_set)} entity lemmas based on provided text.")

            # --- Process Target Entities ---
            target_entities_raw = identify_entities(target_text, nlp_model)
            target_entity_counts = Counter()
            target_entities_set = set() # Store (lemma, label) tuples
            for entity, label in target_entities_raw:
                 if label not in exclude_types:
                     doc = nlp_model(entity)
                     lemma = " ".join([token.lemma_.lower() for token in doc])
                     # Check against excluded lemmas
                     if lemma not in excluded_entities_set:
                          target_entity_counts[(entity, label)] += 1 # Count original form
                          target_entities_set.add((lemma, label)) # Store lemma for comparison

            # --- Process Competitor Content ---
            competitor_texts = {} # Store text by source identifier
            if competitor_source_option == "Extract from URL":
                 with st.spinner("Extracting competitor texts from URLs..."):
                     for url in competitor_list:
                          text = extract_text_from_url(url)
                          if text:
                               competitor_texts[url] = text
                          else:
                               st.warning(f"Could not extract text from competitor URL: {url}")
            else:
                 for i, content in enumerate(competitor_content_blocks):
                      competitor_texts[f"Pasted Content {i+1}"] = content

            if not competitor_texts:
                 st.error("No competitor content could be processed.")
                 return

            # --- Analyze Competitor Entities & Find Gaps ---
            gap_entities_agg = Counter() # Aggregated count across all competitors
            gap_entities_site_count = Counter() # Count of sites mentioning the gap entity
            entities_per_competitor = {} # Store counts per competitor

            with st.spinner("Analyzing competitor entities and identifying gaps..."):
                 for source_id, text in competitor_texts.items():
                      if not text: continue
                      comp_entities_raw = identify_entities(text, nlp_model)
                      comp_entity_counts = Counter()
                      comp_entities_lemmas = set()

                      for entity, label in comp_entities_raw:
                           if label not in exclude_types:
                                doc = nlp_model(entity)
                                lemma = " ".join([token.lemma_.lower() for token in doc])
                                if lemma not in excluded_entities_set:
                                     comp_entity_counts[(entity, label)] += 1 # Count original form
                                     comp_entities_lemmas.add((lemma, label)) # Store lemma

                      entities_per_competitor[source_id] = comp_entity_counts # Store competitor's entities

                      # Check for gaps against target lemmas
                      for lemma_label in comp_entities_lemmas:
                           if lemma_label not in target_entities_set:
                                # Find original entity forms matching this lemma/label for counting
                                original_forms = [orig for orig, lbl in comp_entity_counts if lbl == lemma_label[1] and " ".join([tok.lemma_.lower() for tok in nlp_model(orig)]) == lemma_label[0]]
                                if original_forms:
                                    # Use the most common original form for aggregation key if multiple exist
                                    agg_key = max(original_forms, key=lambda x: comp_entity_counts.get((x, lemma_label[1]), 0))
                                    gap_key = (agg_key, lemma_label[1])
                                    # Sum counts of all original forms matching the lemma
                                    total_count_for_lemma = sum(comp_entity_counts.get((orig, lemma_label[1]), 0) for orig in original_forms)
                                    gap_entities_agg[gap_key] += total_count_for_lemma
                                    gap_entities_site_count[gap_key] += 1 # Increment site count only once per lemma per site


            # --- Display Gap Analysis ---
            st.markdown("### Entity Gap Analysis (Present in Competitors, Missing on Target)")
            if gap_entities_agg:
                # Display Bar Chart of Total Occurrences
                st.markdown("#### Gap Entity Frequency (Total Occurrences Across Competitors)")
                display_entity_barchart(gap_entities_agg, top_n=30) # Show total counts

                # Build table with Site Count and Wikidata links
                st.markdown("#### Gap Entities Details (# Sites Present & Wikidata Link)")
                gap_table_data = []
                # Sort by site count first, then total frequency
                sorted_gap_entities = sorted(gap_entities_site_count.items(), key=lambda item: (item[1], gap_entities_agg.get(item[0], 0)), reverse=True)

                with st.spinner("Fetching Wikidata links for gap entities..."):
                    for (entity, label), site_count in sorted_gap_entities:
                        total_freq = gap_entities_agg.get((entity, label), 0)
                        wikidata_url = get_wikidata_link(entity) # Fetch link
                        gap_table_data.append({
                            "Entity": entity,
                            "Label": label,
                            "# Sites": site_count,
                            "Total Freq.": total_freq,
                            "Wikidata URL": wikidata_url if wikidata_url else "Not Found"
                        })

                if gap_table_data:
                    df_aggregated_gap = pd.DataFrame(gap_table_data)
                    st.dataframe(
                        df_aggregated_gap,
                        column_config={
                            "Wikidata URL": st.column_config.LinkColumn("Wikidata URL"),
                            "# Sites": st.column_config.NumberColumn(format="%d"),
                            "Total Freq.": st.column_config.NumberColumn(format="%d")
                        }
                    )
                else:
                    st.write("No gap entities found for table.")
            else:
                st.info("No significant entity gaps identified.")


            # --- Display Unique Target Entities ---
            st.markdown("### Entities Unique to Target Site")
            unique_target_entities = Counter()
            for (entity, label), count in target_entity_counts.items():
                 doc = nlp_model(entity)
                 lemma = " ".join([token.lemma_.lower() for token in doc])
                 # Check if the *lemma* of this target entity exists in the gap entities' lemmas
                 is_in_gap = any(lemma == gap_lemma for (gap_lemma, gap_label) in gap_entities_agg.keys())
                 if not is_in_gap:
                      unique_target_entities[(entity, label)] = count

            if unique_target_entities:
                 st.markdown("#### Unique Target Entity Frequency")
                 display_entity_barchart(unique_target_entities, top_n=30)
                 st.markdown("#### List of Unique Target Entities (Top 50)")
                 for (entity, label), count in unique_target_entities.most_common(50):
                    st.write(f"- {entity} ({label}): {count}")
            else:
                st.write("No unique entities found on the target site (compared to competitors).")

            # --- Display Entities Per Competitor ---
            st.markdown("### Entities Per Competitor Source (Top 50)")
            for source, entity_counts_local in entities_per_competitor.items():
                st.markdown(f"#### Source: {source}")
                if entity_counts_local:
                    # Filter out excluded types and lemmas again for display consistency
                    filtered_local_counts = Counter({
                        (ent, lbl): ct for (ent, lbl), ct in entity_counts_local.items()
                        if lbl not in exclude_types and " ".join([tok.lemma_.lower() for tok in nlp_model(ent)]) not in excluded_entities_set
                    })
                    if filtered_local_counts:
                        for (entity, label), count in filtered_local_counts.most_common(50):
                            st.write(f"- {entity} ({label}): {count}")
                    else:
                         st.write("No relevant entities found after filtering.")
                else:
                    st.write("No entities extracted from this source.")


def displacy_visualization_page():
    st.header("Entity Visualizer")
    st.markdown("Visualize named entities within your content using spaCy's displaCy.")
    url = st.text_input("Enter a URL (Optional):", key="displacy_url", value="")
    use_url = st.checkbox("Use URL for Text Input", key="displacy_use_url")
    text = st.text_area("Enter Text:", key="displacy_text", value="", height=300, disabled=use_url)
    if st.button("Visualize Entities", key="displacy_button"):
        input_text_source = ""
        if use_url:
            if url:
                with st.spinner("Extracting text from URL..."):
                    input_text_source = extract_text_from_url(url)
                    if not input_text_source:
                        st.error("Could not extract text from the URL.")
                        return
            else:
                st.warning("Please enter a URL or uncheck 'Use URL for Text Input'.")
                return
        else: # Use pasted text
            input_text_source = text
            if not input_text_source:
                st.warning("Please enter text or provide a URL.")
                return

        nlp_model = load_spacy_model()
        if not nlp_model:
            st.error("Could not load spaCy model.")
            return

        with st.spinner("Rendering visualization..."):
            # Process smaller chunks if text is very long to avoid browser issues
            max_chars = 100000 # Adjust if needed
            if len(input_text_source) > max_chars:
                st.warning(f"Text is very long ({len(input_text_source)} chars). Visualizing the first {max_chars} characters.")
                input_text_source = input_text_source[:max_chars]

            doc = nlp_model(input_text_source)
            try:
                # Generate HTML using displaCy
                html = spacy.displacy.render(doc, style="ent", jupyter=False, page=False) # page=False for embedding

                # Wrap in a scrollable div
                st.markdown("### Entity Visualization")
                st.components.v1.html(f'<div style="border: 1px solid #e6e6e6; padding: 10px; border-radius: 5px; max-height: 600px; overflow-y: auto;">{html}</div>', height=620, scrolling=False) # Use scrolling=False on component, rely on div

            except Exception as e:
                st.error(f"Error rendering visualization: {e}")


def named_entity_barchart_page():
    st.header("Entity Frequency Charts")
    st.markdown("Visualize the most frequent named entities across one or multiple sources. This tool counts the *total* number of occurrences for each entity.")
    input_method = st.radio(
        "Select content input method:",
        options=["Extract from URL", "Paste Content"],
        key="entity_barchart_input"
    )

    all_text = ""
    entity_texts_by_source: Dict[str, str] = {} # Store text per source (URL or "Pasted Block X")

    if input_method == "Paste Content":
        st.markdown("Please paste your content. For multiple sources, separate each block with `---`.")
        text_input = st.text_area("Enter Text:", key="barchart_text", height=300, value="")
        if text_input:
             content_blocks = [block.strip() for block in text_input.split('---') if block.strip()]
             for i, block in enumerate(content_blocks):
                  source_id = f"Pasted Block {i+1}"
                  entity_texts_by_source[source_id] = block
                  all_text += block + "\n" # Combine for overall analysis
    else: # Extract from URL
        st.markdown("Enter one or more URLs (one per line). The app will fetch and combine the text from each URL.")
        urls_input = st.text_area("Enter URLs (one per line):", key="barchart_url", value="")
        urls = [url.strip() for url in urls_input.splitlines() if url.strip()]
        if not urls and st.button("Generate Visualizations", key="barchart_button"): # Only check if button clicked
            st.warning("Please enter at least one URL.")
            # No return here, let the button click handle it

    if st.button("Generate Visualizations", key="barchart_button_confirm"): # Renamed button key
        if not all_text and input_method == "Paste Content":
             st.warning("Please paste content to analyze.")
             return
        if not urls and input_method == "Extract from URL":
             st.warning("Please enter at least one URL.")
             return

        # Extract text if URLs were provided
        if input_method == "Extract from URL":
            with st.spinner("Extracting text from URLs..."):
                for url in urls:
                    extracted_text = extract_text_from_url(url)
                    if extracted_text:
                        entity_texts_by_source[url] = extracted_text
                        all_text += extracted_text + "\n"
                    else:
                        st.warning(f"Couldn't grab the text from {url}...")

        if not all_text:
            st.error("No text could be extracted or provided for analysis.")
            return

        with st.spinner("Analyzing entities and generating visualizations..."):
            nlp_model = load_spacy_model()
            if not nlp_model:
                st.error("Could not load spaCy model. Aborting.")
                return

            # Use the BERT NER pipeline for identification
            entities_raw = identify_entities(all_text, nlp_model) # Use BERT NER

            # Filter common non-semantic types before counting
            # Use more specific types based on BERT model's output ('ORG', 'PER', 'LOC', 'MISC')
            allowed_labels = {'ORG', 'PER', 'LOC', 'MISC'} # Adjust as needed based on dslim/bert-base-NER labels
            filtered_entities = [(entity, label) for entity, label in entities_raw if label in allowed_labels]

            # Use the function that counts *total* occurrences
            entity_counts = count_entities_total(filtered_entities, nlp_model) # Counts total occurrences

            if entity_counts:
                st.subheader("Overall Entity Frequency (All Sources Combined)")
                # Display visualizations for combined text
                display_entity_barchart(entity_counts, top_n=30)
                st.subheader("Overall Entity Wordcloud")
                display_entity_wordcloud(entity_counts)

                # Display per-source breakdown
                st.subheader("Entity Counts Per Source (Top 20)")
                for source_id, text_from_source in entity_texts_by_source.items():
                    st.markdown(f"#### Source: {source_id}")
                    if text_from_source:
                        source_entities_raw = identify_entities(text_from_source, nlp_model)
                        source_filtered_entities = [(entity, label) for entity, label in source_entities_raw if label in allowed_labels]
                        source_entity_counts = count_entities_total(source_filtered_entities, nlp_model)
                        if source_entity_counts:
                            for (entity, label), count in source_entity_counts.most_common(20):
                                st.write(f"- {entity} ({label}): {count}")
                        else:
                            st.write("No relevant entities found in this source.")
                    else:
                        st.write(f"No text processed for {source_id}")
            else:
                st.warning("No relevant entities found in the provided content after filtering.")


# ------------------------------------
# Semantic Gap Analyzer (without clustering)
# ------------------------------------
def ngram_tfidf_analysis_page():
    st.header("Semantic Gap Analyzer")
    st.markdown("""
        Uncover hidden opportunities by comparing your website's content to your top competitors.
        Identify key phrases and topics they're covering that you might be missing, and prioritize your content creation.
    """)
    # Competitor input
    st.subheader("Competitors")
    competitor_source_option = st.radio(
        "Select competitor content source:",
        options=["Extract from URL", "Paste Content"],
        index=0,
        key="competitor_source_gap" # Unique key
    )
    competitor_list = []
    competitor_content_blocks = []
    if competitor_source_option == "Extract from URL":
        competitor_input = st.text_area("Enter Competitor URLs (one per line):", key="competitor_urls_gap", value="")
        competitor_list = [url.strip() for url in competitor_input.splitlines() if url.strip()]
    else:
        st.markdown("Paste competitor content below. Separate each competitor content block with `---`.")
        competitor_input = st.text_area("Enter Competitor Content:", key="competitor_text_gap", value="", height=200)
        competitor_content_blocks = [content.strip() for content in competitor_input.split('---') if content.strip()]

    # Target input
    st.subheader("Your Site")
    target_source_option = st.radio(
        "Select target content source:",
        options=["Extract from URL", "Paste Content"],
        index=0,
        key="target_source_gap" # Unique key
    )
    target_url = ""
    target_text = ""
    if target_source_option == "Extract from URL":
        target_url = st.text_input("Enter Your Target URL:", key="target_url_gap", value="")
    else:
        target_text = st.text_area("Paste your target content:", key="target_text_gap", value="", height=200)

    # Word options
    st.subheader("N-gram & TF-IDF Settings")
    n_value = st.selectbox("Select N (words per phrase):", options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=1, key="ngram_n_gap")
    min_df = st.number_input("Minimum Document Frequency (TF-IDF):", value=1, min_value=1, step=1, key="min_df_gap_tfidf")
    max_df = st.number_input("Maximum Document Frequency (TF-IDF, 0.0-1.0):", value=1.0, min_value=0.0, max_value=1.0, step=0.05, key="max_df_gap_tfidf")
    top_n_results = st.slider("Number of top gap results to display:", min_value=10, max_value=100, value=30, key="top_n_results_gap")
    # LDA is removed from this specific tool as requested by user in other contexts, focus on TF-IDF + SBERT gap
    # num_topics = st.slider("Number of topics for LDA:", min_value=2, max_value=15, value=5, key="lda_topics") # Slider for LDA topics

    if st.button("Analyze Content Gaps", key="content_gap_button"):
        if competitor_source_option == "Extract from URL" and not competitor_list:
            st.warning("Please enter at least one competitor URL.")
            return
        if competitor_source_option == "Paste Content" and not competitor_content_blocks:
            st.warning("Please paste competitor content.")
            return
        if target_source_option == "Extract from URL" and not target_url:
            st.warning("Please enter your target URL.")
            return
        if target_source_option == "Paste Content" and not target_text:
            st.warning("Please paste your target content.")
            return

        # Extract competitor content
        competitor_texts_map = {} # Use map {source_id: text}
        valid_competitor_sources = []
        with st.spinner("Extracting competitor content..."):
            if competitor_source_option == "Extract from URL":
                for url in competitor_list:
                    text = extract_relevant_text_from_url(url) # Use relevant text extractor
                    if text:
                        competitor_texts_map[url] = text
                        valid_competitor_sources.append(url)
                    else:
                        st.warning(f"Could not extract content from competitor URL: {url}")
            else: # Pasted content
                for i, content in enumerate(competitor_content_blocks):
                    source_id = f"Pasted Competitor {i+1}"
                    competitor_texts_map[source_id] = content
                    valid_competitor_sources.append(source_id)

        if not competitor_texts_map:
             st.error("No competitor content could be processed.")
             return

        # Extract target content
        target_content_processed = ""
        if target_source_option == "Extract from URL":
            target_content_processed = extract_relevant_text_from_url(target_url) # Use relevant text extractor
            if not target_content_processed:
                st.error(f"Could not extract content from target URL: {target_url}")
                return
        else:
            target_content_processed = target_text

        nlp_model = load_spacy_model()
        if not nlp_model:
             st.error("Could not load spaCy model.")
             return

        # Preprocess texts
        with st.spinner("Preprocessing text..."):
            processed_competitor_texts = [preprocess_text(text, nlp_model) for text in competitor_texts_map.values()]
            processed_target_content = preprocess_text(target_content_processed, nlp_model)

        # Calculate TF-IDF scores for competitors
        with st.spinner("Calculating TF-IDF scores for competitors..."):
            # Ensure ENGLISH_STOP_WORDS is available
            stop_words_list = list(ENGLISH_STOP_WORDS)
            vectorizer = TfidfVectorizer(
                ngram_range=(n_value, n_value),
                min_df=min_df,
                max_df=max_df,
                stop_words=stop_words_list # Pass the list explicitly
            )
            try:
                tfidf_matrix = vectorizer.fit_transform(processed_competitor_texts)
                feature_names = vectorizer.get_feature_names_out()
                if len(feature_names) == 0:
                     st.error("TF-IDF resulted in zero features. Check your text and TF-IDF settings (especially min/max df).")
                     return
                df_tfidf_competitors = pd.DataFrame(tfidf_matrix.toarray(), index=valid_competitor_sources, columns=feature_names)
            except ValueError as e:
                 st.error(f"TF-IDF Error (likely due to min/max df settings): {e}")
                 return

        # --- LDA Topic Modeling Removed ---

        # Calculate TF-IDF for target
        with st.spinner("Calculating TF-IDF scores for target content..."):
            target_tfidf_vector = vectorizer.transform([processed_target_content])
            # Ensure target DF columns match competitor columns exactly
            df_tfidf_target = pd.DataFrame(target_tfidf_vector.toarray(), index=["Target Content"], columns=feature_names)


        # Get SentenceTransformer Embeddings
        model = initialize_sentence_transformer()
        with st.spinner("Calculating SentenceTransformer embeddings..."):
            target_embedding = get_embedding(processed_target_content, model)
            competitor_embeddings = [get_embedding(text, model) for text in processed_competitor_texts]

        # Compute candidate gap scores
        candidate_scores = []
        with st.spinner("Identifying potential gap n-grams..."):
            # Consider ALL n-grams found in competitors, not just top N per competitor initially
            all_competitor_ngrams = set(feature_names)

            for idx, source in enumerate(valid_competitor_sources):
                for ngram in all_competitor_ngrams:
                     # Check if ngram exists in this specific competitor's TF-IDF (it might be zero)
                     if ngram in df_tfidf_competitors.columns:
                          competitor_tfidf = df_tfidf_competitors.loc[source, ngram]
                     else:
                          competitor_tfidf = 0.0 # Should not happen if using feature_names, but safety check

                     # If competitor TF-IDF is zero or very low, skip for this competitor
                     if competitor_tfidf < 1e-6:
                          continue

                     # Get target TF-IDF score (will be 0 if ngram not in target vocab)
                     target_tfidf = df_tfidf_target.loc["Target Content", ngram] if ngram in df_tfidf_target.columns else 0.0

                     tfidf_diff = competitor_tfidf - target_tfidf

                     # Only consider if competitor score is higher
                     if tfidf_diff > 1e-6: # Use small threshold > 0
                          # Calculate semantic similarity difference
                          ngram_embedding = get_embedding(ngram, model)
                          competitor_similarity = cosine_similarity([ngram_embedding], [competitor_embeddings[idx]])[0][0]
                          target_similarity = cosine_similarity([ngram_embedding], [target_embedding])[0][0]
                          bert_diff = competitor_similarity - target_similarity

                          # Store competitor source along with scores
                          candidate_scores.append({'source': source, 'ngram': ngram, 'tfidf_diff': tfidf_diff, 'bert_diff': bert_diff})

        if not candidate_scores:
            st.error("No gap n-grams were identified where competitor TF-IDF > target TF-IDF. Consider adjusting TF-IDF parameters or checking content.")
            return

        # Normalize and combine scores
        df_candidates = pd.DataFrame(candidate_scores)
        tfidf_vals = df_candidates['tfidf_diff']
        bert_vals = df_candidates['bert_diff']

        # Normalize using min-max scaling (handle cases where max == min)
        min_tfidf, max_tfidf = tfidf_vals.min(), tfidf_vals.max()
        min_bert, max_bert = bert_vals.min(), bert_vals.max()
        epsilon = 1e-8 # Avoid division by zero

        df_candidates['norm_tfidf'] = (tfidf_vals - min_tfidf) / (max_tfidf - min_tfidf + epsilon) if (max_tfidf - min_tfidf) > epsilon else 0.0
        df_candidates['norm_bert'] = (bert_diff - min_bert) / (max_bert - min_bert + epsilon) if (max_bert - min_bert) > epsilon else 0.0

        # Combine scores (e.g., weighted average - adjust weights if needed)
        tfidf_weight = 0.4
        bert_weight = 1.0 - tfidf_weight
        df_candidates['gap_score'] = (tfidf_weight * df_candidates['norm_tfidf'] + bert_weight * df_candidates['norm_bert'])

        # Filter out zero scores and sort
        df_candidates = df_candidates[df_candidates['gap_score'] > epsilon].sort_values(by='gap_score', ascending=False)

        # Aggregate results - Find the *highest* gap score for each unique n-gram across all competitors
        df_consolidated = df_candidates.loc[df_candidates.groupby('ngram')['gap_score'].idxmax()]
        df_consolidated = df_consolidated.sort_values(by='gap_score', ascending=False).reset_index(drop=True)


        # Display consolidated gap analysis table
        st.markdown("### Consolidated Semantic Gap Analysis (Top N-grams)")
        st.markdown("Showing n-grams with the highest gap score across all competitors.")
        st.dataframe(
             df_consolidated[['ngram', 'gap_score', 'source', 'tfidf_diff', 'bert_diff']].head(top_n_results).style.format({
                  'gap_score': '{:.4f}',
                  'tfidf_diff': '{:.4f}',
                  'bert_diff': '{:.4f}'
             })
        )

        # Display per-competitor gap analysis tables
        st.markdown("### Per-Competitor Semantic Gap Analysis (Top N-grams)")
        for source in valid_competitor_sources:
            df_source = df_candidates[df_candidates['source'] == source].sort_values(by='gap_score', ascending=False)
            if not df_source.empty:
                 st.markdown(f"#### Competitor: {source}")
                 st.dataframe(
                      df_source[['ngram', 'gap_score', 'tfidf_diff', 'bert_diff']].head(top_n_results // len(valid_competitor_sources) + 1).style.format({ # Show proportional slice
                            'gap_score': '{:.4f}',
                            'tfidf_diff': '{:.4f}',
                            'bert_diff': '{:.4f}'
                      })
                 )


        # --- Word Cloud Visualization ---
        st.subheader("Combined Semantic Gap Wordcloud")
        # Use the consolidated scores (highest score per ngram) for the combined cloud
        gap_scores_for_cloud = df_consolidated.set_index('ngram')['gap_score'].to_dict()
        if gap_scores_for_cloud:
            # Scale scores slightly for better visualization if needed
            max_score_cloud = max(gap_scores_for_cloud.values())
            scaled_scores = {k: v / max_score_cloud * 100 for k, v in gap_scores_for_cloud.items()} # Scale 0-100
            display_entity_wordcloud(scaled_scores) # Use the existing wordcloud function
        else:
            st.write("No combined gap n-grams to create a wordcloud.")

        # Display a wordcloud for each competitor using their specific gap score weights
        st.subheader("Per-Competitor Gap Wordclouds")
        for source in valid_competitor_sources:
            comp_gap_scores = df_candidates[df_candidates['source'] == source].set_index('ngram')['gap_score'].to_dict()
            if comp_gap_scores:
                st.markdown(f"**Wordcloud for Competitor: {source}**")
                # Scale scores for this competitor
                max_comp_score = max(comp_gap_scores.values()) if comp_gap_scores else 1.0
                scaled_comp_scores = {k: v / max_comp_score * 100 for k, v in comp_gap_scores.items()}
                display_entity_wordcloud(scaled_comp_scores)
            else:
                st.write(f"No gap n-grams for competitor: {source}")


def keyword_clustering_from_gap_page():
    st.header("Keyword Clusters from Semantic Gap")
    st.markdown(
        """
        This tool combines semantic gap analysis with keyword clustering.
        1. It identifies key phrases (n-grams) where competitors have higher relevance (TF-IDF & Semantic Similarity) than your target content.
        2. It then clusters these identified "gap" n-grams based on their semantic similarity using SentenceTransformer embeddings.
        3. Finally, it visualizes these clusters to reveal thematic opportunity areas.
        """
    )
    st.subheader("Competitors")
    competitor_source_option = st.radio("Select competitor content source:", options=["Extract from URL", "Paste Content"], index=0, key="comp_source_cluster")
    competitor_list = []
    competitor_content_blocks = []
    if competitor_source_option == "Extract from URL":
        competitor_input = st.text_area("Enter Competitor URLs (one per line):", key="comp_urls_cluster", value="")
        competitor_list = [url.strip() for url in competitor_input.splitlines() if url.strip()]
    else:
        st.markdown("Paste competitor content below. Separate each block with `---`.")
        competitor_input = st.text_area("Enter Competitor Content:", key="competitor_content_cluster", value="", height=200)
        competitor_content_blocks = [content.strip() for content in competitor_input.split('---') if content.strip()]

    st.subheader("Your Site")
    target_source_option = st.radio("Select target content source:", options=["Extract from URL", "Paste Content"], index=0, key="target_source_cluster")
    target_url = ""
    target_text = ""
    if target_source_option == "Extract from URL":
        target_url = st.text_input("Enter Your URL:", key="target_url_cluster", value="")
    else:
        target_text = st.text_area("Paste your content:", key="target_content_cluster", value="", height=200)

    st.subheader("N-gram & TF-IDF Settings")
    n_value = st.selectbox("Select N (words per phrase):", options=[1,2,3,4,5], index=1, key="ngram_n_cluster")
    min_df = st.number_input("Minimum Document Frequency (TF-IDF):", value=1, min_value=1, step=1, key="min_df_cluster")
    max_df = st.number_input("Maximum Document Frequency (TF-IDF, 0.0-1.0):", value=1.0, min_value=0.0, max_value=1.0, step=0.05, key="max_df_cluster")
    # top_n per competitor is implicitly handled by considering all ngrams now

    st.subheader("Clustering Settings")
    algorithm = st.selectbox("Select Clustering Algorithm:", options=["KMeans", "Agglomerative"], index=0, key="clustering_algo_gap")
    n_clusters = st.number_input("Number of Clusters:", min_value=2, max_value=30, value=5, step=1, key="clusters_num_gap")
    # Dimension reduction for plot
    dim_reduction = st.selectbox("Dimension Reduction for Plot:", options=["PCA", "UMAP"], index=1) # Default to UMAP

    if st.button("Analyze & Cluster Gaps", key="gap_cluster_button"):
        # --- 1. Input Validation ---
        if competitor_source_option == "Extract from URL" and not competitor_list:
            st.warning("Please enter at least one competitor URL.")
            return
        if competitor_source_option == "Paste Content" and not competitor_content_blocks:
             st.warning("Please paste competitor content.")
             return
        if target_source_option == "Extract from URL" and not target_url:
            st.warning("Please enter your target URL.")
            return
        if target_source_option == "Paste Content" and not target_text:
             st.warning("Please paste your target content.")
             return

        # --- 2. Content Extraction ---
        competitor_texts_map = {}
        valid_competitor_sources = []
        with st.spinner("Extracting competitor content..."):
             # (Same extraction logic as Semantic Gap Analyzer)
            if competitor_source_option == "Extract from URL":
                for url in competitor_list:
                    text = extract_relevant_text_from_url(url)
                    if text:
                        competitor_texts_map[url] = text
                        valid_competitor_sources.append(url)
                    else:
                        st.warning(f"Could not extract content from competitor URL: {url}")
            else: # Pasted content
                for i, content in enumerate(competitor_content_blocks):
                    source_id = f"Pasted Competitor {i+1}"
                    competitor_texts_map[source_id] = content
                    valid_competitor_sources.append(source_id)

        if not competitor_texts_map:
             st.error("No competitor content could be processed.")
             return

        target_content_processed = ""
        if target_source_option == "Extract from URL":
            target_content_processed = extract_relevant_text_from_url(target_url)
            if not target_content_processed:
                st.error(f"Could not extract content from target URL: {target_url}")
                return
        else:
            target_content_processed = target_text

        # --- 3. Preprocessing ---
        nlp_model = load_spacy_model()
        if not nlp_model:
             st.error("Could not load spaCy model.")
             return
        with st.spinner("Preprocessing text..."):
             processed_competitor_texts = [preprocess_text(text, nlp_model) for text in competitor_texts_map.values()]
             processed_target_content = preprocess_text(target_content_processed, nlp_model)

        # --- 4. TF-IDF Calculation ---
        with st.spinner("Calculating TF-IDF scores..."):
             stop_words_list = list(ENGLISH_STOP_WORDS)
             vectorizer = TfidfVectorizer(ngram_range=(n_value, n_value), min_df=min_df, max_df=max_df, stop_words=stop_words_list)
             try:
                 tfidf_matrix = vectorizer.fit_transform(processed_competitor_texts)
                 feature_names = vectorizer.get_feature_names_out()
                 if len(feature_names) == 0:
                      st.error("TF-IDF resulted in zero features. Check settings.")
                      return
                 df_tfidf_competitors = pd.DataFrame(tfidf_matrix.toarray(), index=valid_competitor_sources, columns=feature_names)
                 target_tfidf_vector = vectorizer.transform([processed_target_content])
                 df_tfidf_target = pd.DataFrame(target_tfidf_vector.toarray(), index=["Target Content"], columns=feature_names)
             except ValueError as e:
                  st.error(f"TF-IDF Error: {e}")
                  return

        # --- 5. Embedding Calculation ---
        model = initialize_sentence_transformer()
        with st.spinner("Calculating SentenceTransformer embeddings..."):
            target_embedding = get_embedding(processed_target_content, model)
            competitor_embeddings = [get_embedding(text, model) for text in processed_competitor_texts]

        # --- 6. Gap Score Calculation ---
        candidate_scores = []
        with st.spinner("Identifying potential gap n-grams..."):
             # (Same gap calculation logic as Semantic Gap Analyzer)
            all_competitor_ngrams = set(feature_names)
            for idx, source in enumerate(valid_competitor_sources):
                for ngram in all_competitor_ngrams:
                    if ngram in df_tfidf_competitors.columns:
                         competitor_tfidf = df_tfidf_competitors.loc[source, ngram]
                    else: continue # Skip if not in this comp's features (shouldn't happen)

                    if competitor_tfidf < 1e-6: continue

                    target_tfidf = df_tfidf_target.loc["Target Content", ngram] if ngram in df_tfidf_target.columns else 0.0
                    tfidf_diff = competitor_tfidf - target_tfidf

                    if tfidf_diff > 1e-6:
                         ngram_embedding = get_embedding(ngram, model)
                         competitor_similarity = cosine_similarity([ngram_embedding], [competitor_embeddings[idx]])[0][0]
                         target_similarity = cosine_similarity([ngram_embedding], [target_embedding])[0][0]
                         bert_diff = competitor_similarity - target_similarity
                         candidate_scores.append({'source': source, 'ngram': ngram, 'tfidf_diff': tfidf_diff, 'bert_diff': bert_diff})

        if not candidate_scores:
            st.error("No gap n-grams identified. Cannot perform clustering.")
            return

        # --- 7. Normalize & Combine Gap Scores ---
        df_candidates = pd.DataFrame(candidate_scores)
        # (Same normalization logic as Semantic Gap Analyzer)
        tfidf_vals = df_candidates['tfidf_diff']
        bert_vals = df_candidates['bert_diff']
        min_tfidf, max_tfidf = tfidf_vals.min(), tfidf_vals.max()
        min_bert, max_bert = bert_vals.min(), bert_vals.max()
        epsilon = 1e-8
        df_candidates['norm_tfidf'] = (tfidf_vals - min_tfidf) / (max_tfidf - min_tfidf + epsilon) if (max_tfidf - min_tfidf) > epsilon else 0.0
        df_candidates['norm_bert'] = (bert_vals - min_bert) / (max_bert - min_bert + epsilon) if (max_bert - min_bert) > epsilon else 0.0
        tfidf_weight = 0.4
        bert_weight = 1.0 - tfidf_weight
        df_candidates['gap_score'] = (tfidf_weight * df_candidates['norm_tfidf'] + bert_weight * df_candidates['norm_bert'])

        # Get unique gap ngrams with their highest score
        df_gap_ngrams = df_candidates.loc[df_candidates.groupby('ngram')['gap_score'].idxmax()]
        df_gap_ngrams = df_gap_ngrams[df_gap_ngrams['gap_score'] > epsilon].sort_values(by='gap_score', ascending=False)

        gap_ngrams_list = df_gap_ngrams['ngram'].tolist()
        if not gap_ngrams_list:
             st.error("No valid gap n-grams found after scoring. Cannot cluster.")
             return

        # --- 8. Compute Embeddings for Gap N-grams ---
        gap_embeddings = []
        valid_gap_ngrams = [] # Keep track of ngrams for which embedding worked
        with st.spinner("Computing SentenceTransformer embeddings for gap n-grams..."):
            embeddings_batch = model.encode(gap_ngrams_list, show_progress_bar=False)
            # Assume all embeddings are valid for now unless encode raises error or returns None
            gap_embeddings = np.array(embeddings_batch)
            valid_gap_ngrams = gap_ngrams_list # If encode succeeds, all are valid

        if len(valid_gap_ngrams) < n_clusters:
            st.error(f"Number of valid gap n-grams ({len(valid_gap_ngrams)}) is less than the desired number of clusters ({n_clusters}). Please reduce the number of clusters or check input content.")
            return

        # --- 9. Clustering ---
        cluster_labels = None
        rep_keywords = {}
        with st.spinner(f"Performing {algorithm} clustering with {n_clusters} clusters..."):
            if algorithm == "KMeans":
                 # Adjust n_clusters if it's more than the number of samples
                 actual_n_clusters = min(n_clusters, gap_embeddings.shape[0])
                 if actual_n_clusters != n_clusters:
                     st.warning(f"Reduced number of clusters to {actual_n_clusters} as it cannot exceed the number of data points.")
                     n_clusters = actual_n_clusters

                 clustering_model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
                 cluster_labels = clustering_model.fit_predict(gap_embeddings)
                 centers = clustering_model.cluster_centers_

                 # Find representative keyword (closest to centroid)
                 for i in range(n_clusters):
                     cluster_indices = np.where(cluster_labels == i)[0]
                     if len(cluster_indices) == 0: continue
                     cluster_grams = [valid_gap_ngrams[idx] for idx in cluster_indices]
                     cluster_embeddings_subset = gap_embeddings[cluster_indices]
                     distances = np.linalg.norm(cluster_embeddings_subset - centers[i], axis=1)
                     rep_keyword_index = cluster_indices[np.argmin(distances)]
                     rep_keywords[i] = valid_gap_ngrams[rep_keyword_index]

            elif algorithm == "Agglomerative":
                 # Adjust n_clusters if it's more than the number of samples
                 actual_n_clusters = min(n_clusters, gap_embeddings.shape[0])
                 if actual_n_clusters != n_clusters:
                     st.warning(f"Reduced number of clusters to {actual_n_clusters} as it cannot exceed the number of data points.")
                     n_clusters = actual_n_clusters
                 # n_clusters cannot be 1 for Agglomerative
                 if n_clusters < 2:
                     st.error("Agglomerative clustering requires at least 2 clusters.")
                     return

                 clustering_model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward') # Ward linkage is common
                 cluster_labels = clustering_model.fit_predict(gap_embeddings)

                 # Find representative keyword (most central by average similarity)
                 for i in range(n_clusters):
                      cluster_indices = np.where(cluster_labels == i)[0]
                      if len(cluster_indices) == 0: continue
                      cluster_grams = [valid_gap_ngrams[idx] for idx in cluster_indices]
                      cluster_embeddings_subset = gap_embeddings[cluster_indices]
                      if len(cluster_indices) == 1:
                           rep_keywords[i] = cluster_grams[0]
                      else:
                           sim_matrix_subset = cosine_similarity(cluster_embeddings_subset)
                           avg_similarity = np.mean(sim_matrix_subset, axis=1)
                           rep_keyword_index = cluster_indices[np.argmax(avg_similarity)]
                           rep_keywords[i] = valid_gap_ngrams[rep_keyword_index]

        if cluster_labels is None:
            st.error("Clustering failed.")
            return

        # --- 10. Dimensionality Reduction for Plotting ---
        embeddings_2d = None
        with st.spinner(f"Reducing dimensions using {dim_reduction}..."):
             n_neighbors_dim = min(15, gap_embeddings.shape[0] - 1) # UMAP constraint
             min_dist_dim = 0.1
             if n_neighbors_dim < 2: # Need at least 2 neighbors
                 st.warning("Too few data points for reliable UMAP/PCA. Visualization might be less meaningful.")
                 n_neighbors_dim = max(2, n_neighbors_dim) # Force at least 2 if possible

             if dim_reduction == "UMAP":
                 reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors_dim, min_dist=min_dist_dim, metric='cosine', random_state=42)
                 embeddings_2d = reducer.fit_transform(gap_embeddings)
             else: # PCA
                 reducer = PCA(n_components=2, random_state=42)
                 embeddings_2d = reducer.fit_transform(gap_embeddings)

        # --- 11. Visualization & Output ---
        st.markdown(f"### Interactive Cluster Visualization ({dim_reduction})")
        df_plot = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'N-gram': valid_gap_ngrams,
            'Cluster ID': cluster_labels,
            'Cluster Name': [f"Cluster {l}: {rep_keywords.get(l, 'N/A')}" for l in cluster_labels] # Add rep keyword to hover
        })
        df_plot['Cluster Label'] = df_plot['Cluster ID'].apply(lambda l: f"Cluster {l}") # For coloring

        fig = px.scatter(df_plot, x='x', y='y', color='Cluster Label', text='N-gram',
                         hover_data={'N-gram': True, 'Cluster Name': True, 'x': False, 'y': False}, # Customize hover
                         title=f"Semantic Opportunity Clusters ({dim_reduction} Projection)")
        fig.update_traces(textposition='top center', marker=dict(size=5), textfont_size=8) # Smaller markers/text
        fig.update_layout(
            xaxis_title=f"{dim_reduction} Dimension 1",
            yaxis_title=f"{dim_reduction} Dimension 2",
            height=700,
            legend_title_text='Clusters'
        )
        st.plotly_chart(fig)

        st.markdown("### Keyword Clusters & Representatives")
        clusters_dict = {}
        for gram, label in zip(valid_gap_ngrams, cluster_labels):
            clusters_dict.setdefault(label, []).append(gram)

        # Create DataFrame for cluster details
        cluster_data_list = []
        for label, gram_list in sorted(clusters_dict.items()):
            rep = rep_keywords.get(label, "N/A")
            cluster_data_list.append({
                "Cluster ID": label,
                "Representative Keyword": rep,
                "N-grams in Cluster": ", ".join(gram_list[:10]) + ('...' if len(gram_list) > 10 else ''), # Show sample
                "Count": len(gram_list)
            })
        df_clusters = pd.DataFrame(cluster_data_list)
        st.dataframe(df_clusters)

        # Optionally display full lists per cluster
        with st.expander("Show all N-grams per Cluster"):
            for label, gram_list in sorted(clusters_dict.items()):
                rep = rep_keywords.get(label, "N/A")
                st.markdown(f"**Cluster {label}** (Representative: *{rep}*) - {len(gram_list)} N-grams")
                st.markdown(f"`{', '.join(gram_list)}`")


def paa_extraction_clustering_page():
    st.header("People Also Asked Recommendations")
    st.markdown(
        """
        This tool scrapes Google for "People Also Asked" (PAA), Autocomplete suggestions, and Related Searches for a given query.
        It then filters these based on semantic similarity to the original query and visualizes the relationships using hierarchical clustering.
        Use this to identify related sub-topics and questions to build out content clusters or inform content briefs.
        **Note:** Scraping Google may be slow and is subject to Google's blocking mechanisms. Use responsibly.
        """
    )

    search_query = st.text_input("Enter Search Query:", "")
    max_paa_depth = st.slider("Maximum PAA click depth (higher = more questions, slower):", 1, 5, 2, key="paa_depth") # Limit depth

    if st.button("Analyze Related Queries", key="paa_button"):
        if not search_query:
            st.warning("Please enter a search query.")
            return

        all_related_phrases = set() # Use a set to avoid duplicates early on
        paa_questions = set()
        autocomplete_suggestions = []
        related_searches = []

        # --- PAA Extraction ---
        st.write("🔍 Scraping People Also Asked...")
        paa_driver = None
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            user_agent = get_random_user_agent()
            chrome_options.add_argument(f"user-agent={user_agent}")
            paa_driver = webdriver.Chrome(options=chrome_options)
            paa_driver.get("https://www.google.com/search?q=" + search_query.replace(" ", "+")) # Encode query
            time.sleep(random.uniform(2, 4)) # Random sleep

            # Function to find and click PAA elements safely
            def find_and_process_paa(current_depth, max_depth):
                if current_depth > max_depth:
                    return
                try:
                    # Try multiple selectors, prioritize more specific ones
                    selectors = [
                        "div[jsname='Cpkphb'] span.CSkcDe", # More specific span for question text
                        "div.related-question-pair", # Older structure
                        "div[jsname='F79BRe']" # Another potential container
                    ]
                    questions_found_this_round = set()

                    for selector in selectors:
                        elements = paa_driver.find_elements(By.CSS_SELECTOR, selector)
                        if elements:
                             # st.write(f"Found {len(elements)} elements with selector '{selector}' at depth {current_depth}") # Debugging
                             for el in elements:
                                 try:
                                     question_text = el.text.strip()
                                     # Basic filtering for valid questions
                                     if question_text and len(question_text) > 5 and question_text.endswith('?') and question_text not in paa_questions:
                                         paa_questions.add(question_text)
                                         questions_found_this_round.add(question_text)
                                         # Try to click the parent container to expand
                                         parent_clickable = el.find_element(By.XPATH, "./ancestor::div[contains(@class, 'related-question-pair') or contains(@jsname, 'Cpkphb') or contains(@jsname, 'F79BRe')][1]")
                                         if parent_clickable:
                                             paa_driver.execute_script("arguments[0].click();", parent_clickable)
                                             time.sleep(random.uniform(1.5, 2.5)) # Wait for expansion
                                             # Recursive call *after* clicking
                                             find_and_process_paa(current_depth + 1, max_depth)
                                         else:
                                             # st.write(f"Could not find clickable parent for: {question_text}") # Debugging
                                             pass
                                 except NoSuchElementException:
                                      # st.write(f"NoSuchElement while processing an element with selector '{selector}'") # Debugging
                                      continue # Ignore if element structure changes during interaction
                                 except Exception as click_err:
                                     st.warning(f"Minor error clicking/processing PAA element: {click_err}")
                                     continue
                             break # Stop trying selectors if one works

                    # if not questions_found_this_round:
                         # st.write(f"No new PAA questions found at depth {current_depth}") # Debugging

                except Exception as e:
                    st.error(f"Error during PAA extraction at depth {current_depth}: {e}")

            find_and_process_paa(1, max_paa_depth)

        except WebDriverException as e:
             st.error(f"Selenium WebDriver error during PAA scraping: {e}")
        except Exception as e:
             st.error(f"Unexpected error during PAA scraping: {e}")
        finally:
            if paa_driver:
                paa_driver.quit()
        st.write(f"✅ Found {len(paa_questions)} PAA questions.")
        all_related_phrases.update(paa_questions)

        # --- Autocomplete Suggestions ---
        st.write("🔍 Fetching Autocomplete Suggestions...")
        import requests
        autocomplete_url = "http://suggestqueries.google.com/complete/search"
        params = {"client": "firefox", "q": search_query} # Use 'firefox' client for potentially different results
        try:
            response = requests.get(autocomplete_url, params=params, timeout=10)
            response.raise_for_status() # Raise error for bad status codes
            suggestions_json = response.json()
            if len(suggestions_json) > 1 and isinstance(suggestions_json[1], list):
                 autocomplete_suggestions = [s for s in suggestions_json[1] if isinstance(s, str)]
                 st.write(f"✅ Found {len(autocomplete_suggestions)} autocomplete suggestions.")
                 all_related_phrases.update(autocomplete_suggestions)
            else:
                 st.write("Autocomplete suggestions format unexpected.")
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching autocomplete suggestions: {e}")
        except Exception as e:
            st.error(f"Error processing autocomplete suggestions: {e}")


        # --- Related Searches ---
        st.write("🔍 Scraping Related Searches...")
        related_driver = None
        try:
            chrome_options_rel = Options()
            chrome_options_rel.add_argument("--headless")
            chrome_options_rel.add_argument("--no-sandbox")
            chrome_options_rel.add_argument("--disable-dev-shm-usage")
            user_agent_rel = get_random_user_agent()
            chrome_options_rel.add_argument(f"user-agent={user_agent_rel}")
            related_driver = webdriver.Chrome(options=chrome_options_rel)
            related_driver.get("https://www.google.com/search?q=" + search_query.replace(" ", "+"))
            time.sleep(random.uniform(2, 4)) # Random sleep

            # Try multiple selectors for related searches container/items
            related_selectors = [
                "div[jsname='j4xjEf'] a span", # Example selector, might change
                "p.nVcaUb", # Older selector
                "div.s75CSd.OhScic.AB4Wff", # Another potential container
                "a.k8XOCe" # Links often contain the text
            ]
            found_related = False
            for selector in related_selectors:
                 elements = related_driver.find_elements(By.CSS_SELECTOR, selector)
                 if elements:
                      # st.write(f"Found {len(elements)} related search elements with selector '{selector}'") # Debugging
                      for el in elements:
                          try:
                              text = el.text.strip()
                              if text and len(text) > 1: # Basic filter
                                  related_searches.append(text)
                          except:
                              continue # Ignore elements that disappear or cause errors
                      found_related = True
                      break # Stop if a selector works

            if related_searches:
                 related_searches = list(dict.fromkeys(related_searches)) # Remove duplicates preserving order
                 st.write(f"✅ Found {len(related_searches)} related searches.")
                 all_related_phrases.update(related_searches)
            # elif not found_related:
                 # st.write("Could not find related search elements with known selectors.") # Debugging


        except WebDriverException as e:
             st.error(f"Selenium WebDriver error during Related Searches scraping: {e}")
        except Exception as e:
            st.error(f"Unexpected error during Related Searches scraping: {e}")
        finally:
            if related_driver:
                related_driver.quit()


        # --- Similarity Analysis ---
        st.write("🧠 Analyzing similarity...")
        if not all_related_phrases:
            st.warning("No related phrases (PAA, Autocomplete, Related Searches) were collected.")
            return

        model = initialize_sentence_transformer()
        query_embedding = get_embedding(search_query, model)
        phrase_similarities = []
        valid_phrases_for_clustering = []
        valid_embeddings_for_clustering = []

        with st.spinner("Calculating similarity scores for collected phrases..."):
             phrase_list = list(all_related_phrases)
             phrase_embeddings = model.encode(phrase_list, show_progress_bar=False)

             similarities = cosine_similarity(phrase_embeddings, [query_embedding]).flatten()

             for phrase, emb, sim in zip(phrase_list, phrase_embeddings, similarities):
                  phrase_similarities.append({'Phrase': phrase, 'Similarity': sim})
                  valid_phrases_for_clustering.append(phrase)
                  valid_embeddings_for_clustering.append(emb)

        if not phrase_similarities:
            st.warning("Could not calculate similarities for any phrases.")
            return

        df_similarities = pd.DataFrame(phrase_similarities).sort_values(by="Similarity", ascending=False).reset_index(drop=True)

        # --- Filtering & Dendrogram ---
        similarity_threshold = st.slider("Similarity Threshold (Keep phrases above this score):", 0.0, 1.0, 0.5, 0.05, key="paa_sim_threshold")
        recommended_df = df_similarities[df_similarities['Similarity'] >= similarity_threshold]

        st.subheader("Topic Dendrogram (Hierarchical Clustering)")
        if not recommended_df.empty:
             # Filter embeddings based on the threshold
             indices_to_keep = recommended_df.index # Get indices from the filtered df *before* resetting
             filtered_embeddings = [valid_embeddings_for_clustering[i] for i in indices_to_keep if i < len(valid_embeddings_for_clustering)]
             filtered_phrases = recommended_df['Phrase'].tolist()

             if len(filtered_embeddings) >= 2: # Dendrogram needs at least 2 points
                 st.write(f"Generating dendrogram for {len(filtered_phrases)} phrases above threshold {similarity_threshold:.2f}...")
                 with st.spinner("Creating dendrogram..."):
                     # Use Plotly Figure Factory for dendrogram
                     fig_dendro = ff.create_dendrogram(
                          np.array(filtered_embeddings), # Must be numpy array
                          orientation='left',
                          labels=filtered_phrases,
                          linkagefun=lambda x: scipy.cluster.hierarchy.linkage(x, method='ward') # Use scipy linkage
                     )
                     fig_dendro.update_layout(
                          width=800,
                          height=max(600, len(filtered_phrases) * 20), # Adjust height based on number of labels
                          title=f"Semantic Clustering of Related Phrases (Similarity > {similarity_threshold:.2f})"
                          )
                     st.plotly_chart(fig_dendro)
             else:
                  st.info(f"Not enough recommended phrases ({len(filtered_embeddings)}) above the threshold to create a dendrogram.")
        else:
            st.info(f"No related phrases found above the similarity threshold of {similarity_threshold:.2f}.")

        # --- Display Results ---
        st.subheader(f"Related Phrases Above {similarity_threshold:.2f} Similarity")
        if not recommended_df.empty:
             recommended_df.index = recommended_df.index + 1
             st.dataframe(recommended_df.style.format({"Similarity": "{:.4f}"}))
        else:
             st.write("None.")

        with st.expander("Show All Collected Phrases & Similarities"):
             df_similarities.index = df_similarities.index + 1
             st.dataframe(df_similarities.style.format({"Similarity": "{:.4f}"}))

        # Add download button for the results
        csv_all = df_similarities.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download All Phrases & Similarities (CSV)",
            data=csv_all,
            file_name=f"related_phrases_{search_query.replace(' ','_')}_all.csv",
            mime='text/csv',
            key='download_all_paa'
        )
        if not recommended_df.empty:
            csv_rec = recommended_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"Download Recommended Phrases (>{similarity_threshold:.2f} Sim) (CSV)",
                data=csv_rec,
                file_name=f"related_phrases_{search_query.replace(' ','_')}_recommended.csv",
                mime='text/csv',
                key='download_rec_paa'
            )


# ------------------------------------
# NEW TOOL: Google Ads Search Term Analyzer (with Classifier)
# ------------------------------------
def google_ads_search_term_analyzer_page():
    st.header("Google Ads Search Term Analyzer")
    st.markdown(
        """
        Upload an Excel file (.xlsx) from your Google Ads search terms report.
        This tool extracts n-grams (or skip-grams) and analyzes their performance (Clicks, Cost, Conversions, etc.).
        Use this to identify high/low performing word combinations for campaign optimization or negative keyword mining.
        """
    )

    uploaded_file = st.file_uploader("Upload Google Ads Search Terms Excel File", type=["xlsx"], key="gads_upload")

    if uploaded_file is not None:
        try:
            # Attempt to read, find header row automatically
            # Common headers to look for to identify the start of the table
            common_headers = ["Search term", "Match type", "Clicks", "Impr.", "Cost", "Conversions"]
            df = None
            # Try reading without skipping first, check for headers
            try:
                 df_test = pd.read_excel(uploaded_file)
                 header_row_index = -1
                 for i, row in df_test.head(10).iterrows(): # Check first 10 rows
                     row_values = [str(v).strip().lower() for v in row.values]
                     # Check if a significant number of common headers are present
                     if sum(h.lower() in row_values for h in common_headers) >= 3:
                          header_row_index = i
                          break
                 if header_row_index != -1:
                     df = pd.read_excel(uploaded_file, header=header_row_index)
                 else:
                     # Fallback: Assume header is on row 3 (index 2) as per original code
                     df = pd.read_excel(uploaded_file, header=2)
                     st.info("Could not automatically detect header row, assuming it starts at row 3.")

            except Exception as read_err:
                 st.error(f"Error reading Excel file. Ensure it's a valid .xlsx file. Error: {read_err}")
                 return

            # --- Column Renaming & Validation ---
            # Standardize potential column names (lowercase, strip spaces)
            df.columns = [str(col).strip().lower() for col in df.columns]

            # Define mapping from potential input names to standard names
            col_mapping = {
                "search term": "Search term",
                "match type": "Match type",
                "added/excluded": "Added/Excluded", # Keep original case if needed later
                "campaign": "Campaign",
                "ad group": "Ad group",
                "clicks": "Clicks",
                "impr.": "Impressions",
                "impressions": "Impressions", # Handle variation
                "currency code": "Currency code",
                "cost": "Cost",
                "avg. cpc": "Avg. CPC",
                "conv. rate": "Conversion Rate",
                "conversions": "Conversions",
                "cost / conv.": "Cost per Conversion",
                "cost/conv.": "Cost per Conversion", # Handle variation
            }

            # Apply renaming based on mapping
            df = df.rename(columns=col_mapping, errors='ignore') # Ignore if a key isn't found

            # Validate required columns (using standard names)
            required_columns = ["Search term", "Clicks", "Impressions", "Cost", "Conversions"]
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                st.error(f"The following required columns are missing after standardization: {', '.join(missing_cols)}. Found columns: {', '.join(df.columns)}")
                return

            # Convert numeric columns, handling errors.
            for col in required_columns[1:]: # Skip "Search term"
                 try:
                     # Handle potential non-numeric characters like '--' or ','
                     df[col] = df[col].astype(str).str.replace(',', '', regex=False).replace('--', np.nan, regex=False)
                     df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                 except Exception as e:
                     st.error(f"Error converting column '{col}' to numeric: {e}. Please check the data format.")
                     return

            st.subheader("N-gram Performance Analysis")

            # --- N-gram Settings ---
            extraction_method = st.radio("N-gram Extraction Method:", options=["Contiguous n-grams", "Skip-grams"], index=0, key="gads_extract_method")
            n_value = st.selectbox("Select N (words per phrase):", options=[1, 2, 3, 4, 5], index=1, key="gads_n_value") # Increased max N
            min_frequency = st.number_input("Minimum N-gram Frequency:", value=2, min_value=1, step=1, key="gads_min_freq")
            ignore_added_excluded = st.checkbox("Ignore search terms already Added/Excluded?", value=True, key="gads_ignore")

            # Filter based on Added/Excluded status if required
            df_filtered = df.copy()
            if ignore_added_excluded and "Added/Excluded" in df.columns:
                # Keep rows where Added/Excluded is 'None' or NaN/empty
                df_filtered = df_filtered[df_filtered["Added/Excluded"].isin(['None', None, np.nan, ''])]
                st.info(f"Ignoring {len(df) - len(df_filtered)} rows marked as Added/Excluded.")
            elif ignore_added_excluded and "Added/Excluded" not in df.columns:
                 st.warning("Cannot ignore Added/Excluded terms - column not found.")

            if df_filtered.empty:
                 st.warning("No search terms remaining after filtering.")
                 return

            # --- N-gram Extraction ---
            lemmatizer = WordNetLemmatizer()
            stop_words_ads = set(stopwords.words('english')) # Use a specific variable

            def extract_ngrams(text, n):
                text = str(text).lower()
                tokens = word_tokenize(text)
                # Keep numbers, allow specific symbols if needed, basic stopword/lemma
                tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalnum() and t not in stop_words_ads]
                ngrams_list = list(nltk.ngrams(tokens, n))
                return [" ".join(gram) for gram in ngrams_list]

            def extract_skipgrams(text, n, k): # k = max skip distance
                import itertools
                text = str(text).lower()
                tokens = word_tokenize(text)
                tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalnum() and t not in stop_words_ads]
                if len(tokens) < n: return []
                skipgrams_list = []
                # Generate combinations and check distance implicitly by combinations logic
                # This simple combinations approach doesn't enforce max skip distance k easily.
                # Using nltk.skipgrams is better if k is needed.
                # For simplicity here, we use combinations which can have large skips.
                for combo in itertools.combinations(tokens, n):
                    skipgram = " ".join(combo)
                    skipgrams_list.append(skipgram)
                # Alternative using nltk (handles k):
                # skipgrams_list = list(nltk.skipgrams(tokens, n=n, k=k))
                # return [" ".join(gram) for gram in skipgrams_list]
                return list(dict.fromkeys(skipgrams_list)) # Remove duplicates from combinations


            all_extracted_items = [] # Can be ngrams or skipgrams
            search_term_to_items = {} # Map search term to its extracted items

            with st.spinner("Extracting items (n-grams/skip-grams)..."):
                for term in df_filtered["Search term"]:
                    extracted = []
                    if extraction_method == "Contiguous n-grams":
                        extracted = extract_ngrams(term, n_value)
                    else: # Skip-grams
                        # Using simple combinations method, k is not explicitly used here
                        extracted = extract_skipgrams(term, n_value, k=n_value*2) # Pass dummy k
                    all_extracted_items.extend(extracted)
                    search_term_to_items[term] = extracted

            item_counts = Counter(all_extracted_items)
            filtered_items = {item: count for item, count in item_counts.items() if count >= min_frequency}

            if not filtered_items:
                st.warning("No n-grams/skip-grams found meeting the minimum frequency requirement.")
                return

            # --- Performance Aggregation ---
            item_performance = {}
            with st.spinner("Aggregating performance metrics..."):
                for index, row in df_filtered.iterrows():
                    search_term_text = row["Search term"]
                    # Use the pre-calculated items for this search term
                    items_in_term = search_term_to_items.get(search_term_text, [])
                    for item in items_in_term:
                        if item in filtered_items: # Check if item meets frequency threshold
                            if item not in item_performance:
                                item_performance[item] = {
                                    "Frequency": 0, # Add frequency count
                                    "Clicks": 0,
                                    "Impressions": 0,
                                    "Cost": 0.0,
                                    "Conversions": 0.0
                                }
                            # We only increment frequency once per search term containing the item
                            # But aggregate metrics for every occurrence within the filtered data
                            item_performance[item]["Clicks"] += row["Clicks"]
                            item_performance[item]["Impressions"] += row["Impressions"]
                            item_performance[item]["Cost"] += row["Cost"]
                            item_performance[item]["Conversions"] += row["Conversions"]

                # Add frequency count after iterating through rows
                for item in item_performance:
                     item_performance[item]["Frequency"] = filtered_items[item]


            if not item_performance:
                 st.warning("Could not aggregate performance. Check data.")
                 return

            df_item_performance = pd.DataFrame.from_dict(item_performance, orient='index')
            df_item_performance.index.name = "N-gram / Skip-gram"
            df_item_performance = df_item_performance.reset_index()

            # --- Calculate KPIs ---
            with st.spinner("Calculating KPIs..."):
                 # CTR
                 df_item_performance["CTR (%)"] = (
                      df_item_performance["Clicks"].astype(float) / df_item_performance["Impressions"].replace(0, np.nan) # Avoid DivByZero
                 ) * 100
                 # Conversion Rate
                 df_item_performance["Conv. Rate (%)"] = (
                      df_item_performance["Conversions"].astype(float) / df_item_performance["Clicks"].replace(0, np.nan) # Avoid DivByZero
                 ) * 100
                 # Cost per Conversion
                 df_item_performance["Cost / Conv."] = (
                      df_item_performance["Cost"].astype(float) / df_item_performance["Conversions"].replace(0, np.nan) # Avoid DivByZero
                 )
                 # Avg CPC
                 df_item_performance["Avg. CPC"] = (
                     df_item_performance["Cost"].astype(float) / df_item_performance["Clicks"].replace(0, np.nan) # Avoid DivByZero
                 )


            # --- Sorting and Display ---
            st.markdown("### Performance Table")
            sort_column = st.selectbox(
                 "Sort table by:",
                 options=df_item_performance.columns,
                 index=df_item_performance.columns.get_loc("Conversions") if "Conversions" in df_item_performance.columns else 0, # Default sort by Conversions
                 key="gads_sort_col"
                 )
            sort_ascending = st.checkbox("Sort Ascending", value=False, key="gads_sort_asc")

            df_sorted = df_item_performance.sort_values(by=sort_column, ascending=sort_ascending, na_position='last')


            # --- Formatting ---
            format_dict_gads = {
                "Clicks": "{:,.0f}",
                "Impressions": "{:,.0f}",
                "Frequency": "{:,.0f}",
                "Conversions": "{:,.1f}", # Allow decimals for conversions
                "Cost": "${:,.2f}",
                "Avg. CPC": "${:,.2f}",
                "Cost / Conv.": "${:,.2f}",
                "CTR (%)": "{:,.2f}%",
                "Conv. Rate (%)": "{:,.2f}%",
            }

            # Apply formatting and display
            st.dataframe(df_sorted.style.format(format_dict_gads, na_rep="N/A"))

            # --- Download Button ---
            csv_gads = df_sorted.to_csv(index=False).encode('utf-8')
            st.download_button(
                 label="Download N-gram Performance Data (CSV)",
                 data=csv_gads,
                 file_name=f"gads_ngram_performance_{n_value}gram.csv",
                 mime='text/csv',
                 key='download_gads_ngram'
            )

        except Exception as e:
            st.error(f"An error occurred while processing the Google Ads data: {e}")
            import traceback
            st.error(traceback.format_exc()) # Show full traceback

# ------------------------------------
# NEW HELPER FUNCTIONS FOR GSC ANALYZER (KMEANS/GPT UPDATE)
# ------------------------------------
@st.cache_resource
def get_openai_client():
    """Safely initializes and returns the OpenAI client using secrets."""
    api_key = None
    if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
        api_key = st.secrets["openai"]["api_key"]
    else:
        # Don't show error here, let the main function handle it if labeling is needed
        print("OpenAI API key not found in Streamlit Secrets.")
        return None

    if api_key:
        try:
            client = OpenAI(api_key=api_key)
            # Optional: Test connection? Be mindful of cost/rate limits.
            # client.models.list()
            print("OpenAI client initialized successfully.")
            return client
        except Exception as e:
            st.error(f"Failed to initialize OpenAI client: {e}")
            return None
    else:
        return None

@st.cache_data(ttl=3600) # Cache GPT labels for an hour
def get_gpt_cluster_label(_client, queries_in_cluster: list, cluster_id: int) -> str:
    """Generates a topic label for a cluster using OpenAI API."""
    # Ensure client is valid before proceeding
    if not isinstance(_client, OpenAI):
        print(f"Invalid OpenAI client passed to get_gpt_cluster_label for cluster {cluster_id}")
        return f"Cluster {cluster_id + 1} (No API Key)" # Return default if client failed or is wrong type

    # Take a sample of queries (max 15)
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
            model="gpt-3.5-turbo", # Or "gpt-4" if preferred/available
            messages=[
                {"role": "system", "content": "You are an expert SEO assistant. Summarize search query clusters into concise 3–5 word topic labels, using Title Case."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4, # Slightly lower temperature for more deterministic labels
            max_tokens=20, # Increased slightly, ensure enough for 5 words
            timeout=20.0, # Add timeout
        )
        topic = response.choices[0].message.content.strip().replace('"', '') # Remove quotes
        # Basic cleanup
        topic = re.sub(r"^(Topic|Label|Cluster) ?\d*: ?", "", topic, flags=re.IGNORECASE).strip() # Remove prefixes
        return topic if topic else f"Cluster {cluster_id + 1} (GPT Failed)"
    except Exception as e:
        st.warning(f"❌ OpenAI API Error labeling cluster {cluster_id + 1}: {e}")
        return f"Cluster {cluster_id + 1} (GPT Error)"

# ------------------------------------
# MODIFIED GSC Analyzer Function
# ------------------------------------
# MODIFIED Google Search Console Analysis Page (Fixes for Aggregation Totals + Formatting)
def google_search_console_analysis_page():
    st.header("Google Search Console Data Analysis")
    st.markdown(
        """
        Compare GSC query data from two periods to identify performance changes.
        This tool now uses **KMeans clustering** on query embeddings (SentenceTransformer) and **GPT-based labeling** to group queries into topics.
        Upload CSV files (one for the 'Before' period and one for the 'After' period), and the tool will:
        - Calculate overall performance changes (based on full input data).
        - **Merge data using an outer join** to preserve all queries.
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
        openai_client = get_openai_client()
        progress_bar = st.progress(0)
        status_text = st.empty()
        try:
            # Step 1: Read CSVs
            status_text.text("Reading CSV files...")
            df_before = pd.read_csv(uploaded_file_before)
            df_after = pd.read_csv(uploaded_file_after)
            progress_bar.progress(5)

            # Step 2: Validate columns & Standardize Names
            status_text.text("Validating columns...")
            # ... (find_col_name and column identification logic remains the same) ...
            required_query_col = "Top queries"
            required_pos_col = "Position"

            def find_col_name(df, potential_names):
                for name in potential_names:
                    for col in df.columns:
                        if col.strip().lower() == name.strip().lower():
                            return col
                return None

            query_col_before = find_col_name(df_before, [required_query_col, "Query"])
            pos_col_before = find_col_name(df_before, [required_pos_col, "Average position", "Position"])
            query_col_after = find_col_name(df_after, [required_query_col, "Query"])
            pos_col_after = find_col_name(df_after, [required_pos_col, "Average position", "Position"])

            clicks_col_before = find_col_name(df_before, ["Clicks"])
            impressions_col_before = find_col_name(df_before, ["Impressions", "Impr."])
            ctr_col_before = find_col_name(df_before, ["CTR"])

            clicks_col_after = find_col_name(df_after, ["Clicks"])
            impressions_col_after = find_col_name(df_after, ["Impressions", "Impr."])
            ctr_col_after = find_col_name(df_after, ["CTR"])

            if not query_col_before or not pos_col_before:
                 st.error(f"The 'Before' CSV must contain columns recognizable as '{required_query_col}' and '{required_pos_col}'. Found: {df_before.columns}")
                 return
            if not query_col_after or not pos_col_after:
                 st.error(f"The 'After' CSV must contain columns recognizable as '{required_query_col}' and '{required_pos_col}'. Found: {df_after.columns}")
                 return

            metrics_to_check = [
                (clicks_col_before, clicks_col_after, "Clicks"),
                (impressions_col_before, impressions_col_after, "Impressions"),
                (ctr_col_before, ctr_col_after, "CTR"),
            ]
            for before_col, after_col, name in metrics_to_check:
                 if (before_col and not after_col) or (not before_col and after_col):
                     st.warning(f"Metric '{name}' column found in only one file. Calculations involving this metric might be incomplete or fail.")

            rename_map_before = {query_col_before: "Query", pos_col_before: "Average Position"}
            rename_map_after = {query_col_after: "Query", pos_col_after: "Average Position"}
            if clicks_col_before: rename_map_before[clicks_col_before] = "Clicks"
            if impressions_col_before: rename_map_before[impressions_col_before] = "Impressions"
            if ctr_col_before: rename_map_before[ctr_col_before] = "CTR"
            if clicks_col_after: rename_map_after[clicks_col_after] = "Clicks"
            if impressions_col_after: rename_map_after[impressions_col_after] = "Impressions"
            if ctr_col_after: rename_map_after[ctr_col_after] = "CTR"

            df_before = df_before.rename(columns=rename_map_before)
            df_after = df_after.rename(columns=rename_map_after)


            # Step 3: Data Cleaning & Type Conversion
            status_text.text("Cleaning data...")
            def clean_metric(series):
                if pd.api.types.is_numeric_dtype(series): return series
                series_str = series.astype(str)
                cleaned = series_str.str.replace('%', '', regex=False)
                cleaned = cleaned.str.replace('<', '', regex=False)
                cleaned = cleaned.str.replace('>', '', regex=False)
                cleaned = cleaned.str.replace(',', '', regex=False)
                cleaned = cleaned.str.strip()
                cleaned = cleaned.replace('', np.nan).replace('N/A', np.nan).replace('--', np.nan)
                return pd.to_numeric(cleaned, errors='coerce')

            potential_metrics = ["Average Position", "Clicks", "Impressions", "CTR"]
            df_before_cleaned = df_before.copy()
            df_after_cleaned = df_after.copy()
            for df in [df_before_cleaned, df_after_cleaned]:
                for col in potential_metrics:
                    if col in df.columns:
                        df[col] = clean_metric(df[col])

            # Step 4: Dashboard Summary (using CLEANED dataframes BEFORE merge)
            st.markdown("## Dashboard Summary")
            cols = st.columns(4)
            # ... (Dashboard calculation logic using df_before_cleaned, df_after_cleaned - REMAINS THE SAME as previous version) ...
            # --- Calculate Weighted Averages Safely ---
            def calculate_weighted_average(values, weights):
                if values is None or weights is None: return np.nan
                valid_indices = values.notna() & weights.notna() & (weights > 0)
                if not valid_indices.any(): return values.mean() if values.notna().any() else np.nan # Fallback
                try:
                    avg = np.average(values[valid_indices], weights=weights[valid_indices])
                    return avg
                except ZeroDivisionError:
                    return values.mean() if values.notna().any() else np.nan # Fallback

            # Clicks
            if "Clicks" in df_before_cleaned.columns and "Clicks" in df_after_cleaned.columns:
                total_clicks_before = df_before_cleaned["Clicks"].sum()
                total_clicks_after = df_after_cleaned["Clicks"].sum()
                overall_clicks_change = total_clicks_after - total_clicks_before
                overall_clicks_change_pct = (overall_clicks_change / total_clicks_before * 100) if pd.notna(total_clicks_before) and total_clicks_before != 0 else 0
                cols[0].metric(label="Clicks Change", value=f"{overall_clicks_change:,.0f}", delta=f"{overall_clicks_change_pct:.1f}%")
            else: cols[0].metric(label="Clicks Change", value="N/A")

            # Impressions
            if "Impressions" in df_before_cleaned.columns and "Impressions" in df_after_cleaned.columns:
                total_impressions_before = df_before_cleaned["Impressions"].sum()
                total_impressions_after = df_after_cleaned["Impressions"].sum()
                overall_impressions_change = total_impressions_after - total_impressions_before
                overall_impressions_change_pct = (overall_impressions_change / total_impressions_before * 100) if pd.notna(total_impressions_before) and total_impressions_before != 0 else 0
                cols[1].metric(label="Impressions Change", value=f"{overall_impressions_change:,.0f}", delta=f"{overall_impressions_change_pct:.1f}%")
            else: cols[1].metric(label="Impressions Change", value="N/A")

            # Position
            overall_avg_position_before = np.nan
            if "Average Position" in df_before_cleaned.columns and "Impressions" in df_before_cleaned.columns:
                overall_avg_position_before = calculate_weighted_average(df_before_cleaned["Average Position"], df_before_cleaned["Impressions"])

            overall_avg_position_after = np.nan
            if "Average Position" in df_after_cleaned.columns and "Impressions" in df_after_cleaned.columns:
                overall_avg_position_after = calculate_weighted_average(df_after_cleaned["Average Position"], df_after_cleaned["Impressions"])

            if pd.notna(overall_avg_position_before) and pd.notna(overall_avg_position_after):
                 overall_position_change = overall_avg_position_before - overall_avg_position_after
                 overall_position_change_pct = (overall_position_change / overall_avg_position_before * 100) if overall_avg_position_before != 0 else 0
                 cols[2].metric(label="Avg. Position Change", value=f"{overall_position_change:.1f}", delta=f"{overall_position_change_pct:.1f}%", delta_color="inverse")
            else:
                 cols[2].metric(label="Avg. Position Change", value="N/A")

            # CTR
            overall_ctr_before = np.nan
            if "CTR" in df_before_cleaned.columns and "Impressions" in df_before_cleaned.columns:
                 overall_ctr_before = calculate_weighted_average(df_before_cleaned["CTR"], df_before_cleaned["Impressions"])

            overall_ctr_after = np.nan
            if "CTR" in df_after_cleaned.columns and "Impressions" in df_after_cleaned.columns:
                 overall_ctr_after = calculate_weighted_average(df_after_cleaned["CTR"], df_after_cleaned["Impressions"])

            if pd.notna(overall_ctr_before) and pd.notna(overall_ctr_after):
                 overall_ctr_change = overall_ctr_after - overall_ctr_before
                 overall_ctr_change_pct = (overall_ctr_change / overall_ctr_before * 100) if pd.notna(overall_ctr_before) and overall_ctr_before != 0 else 0
                 cols[3].metric(label="Avg. CTR Change", value=f"{overall_ctr_change:.2f}% pts", delta=f"{overall_ctr_change_pct:.1f}%")
            else: cols[3].metric(label="Avg. CTR Change", value="N/A")
            progress_bar.progress(10)

            # Step 5: Merge Data using OUTER JOIN
            status_text.text("Merging data (Outer Join)...")
            cols_to_keep_before = ["Query"] + [col for col in potential_metrics if col in df_before_cleaned.columns]
            cols_to_keep_after = ["Query"] + [col for col in potential_metrics if col in df_after_cleaned.columns]

            merged_df = pd.merge(
                df_before_cleaned[cols_to_keep_before],
                df_after_cleaned[cols_to_keep_after],
                on="Query",
                suffixes=("_before", "_after"),
                how='outer' # <<< Use OUTER join here >>>
            )
            if merged_df.empty:
                st.error("Merge resulted in an empty dataframe. Check input files.")
                return
            # NOTE: merged_df now contains NaNs where queries were unique to one period.

            progress_bar.progress(15)

            # Step 6: Calculate YOY changes (handling NaNs from outer join)
            status_text.text("Calculating YOY changes...")

            def calculate_yoy_change(before, after):
                if pd.notna(after) and pd.notna(before): return after - before
                elif pd.notna(after): return after # Treat missing before as 0
                elif pd.notna(before): return -before # Treat missing after as 0
                else: return np.nan # Both missing

            def calculate_yoy_pct_change(yoy_abs, before):
                 if pd.notna(yoy_abs) and pd.notna(before) and before != 0:
                      return (yoy_abs / before) * 100
                 # Handle cases where 'before' is 0 or NaN - % change is undefined/infinite
                 elif pd.notna(yoy_abs) and yoy_abs != 0 and (pd.isna(before) or before == 0):
                      return np.inf # Or np.nan, or a large number like 9999
                 else:
                      return np.nan # Change is 0 or NaN, or before is NaN

            # Position YOY (Lower is better, so calculate Before - After)
            if "Average Position_before" in merged_df.columns and "Average Position_after" in merged_df.columns:
                merged_df["Position_YOY"] = merged_df.apply(lambda row: calculate_yoy_change(row["Average Position_after"], row["Average Position_before"]), axis=1) # Note order swap for "improvement" direction
                merged_df["Position_YOY_pct"] = merged_df.apply(lambda row: calculate_yoy_pct_change(row["Position_YOY"], row["Average Position_before"]), axis=1)

            # Clicks YOY
            if "Clicks_before" in merged_df.columns and "Clicks_after" in merged_df.columns:
                merged_df["Clicks_YOY"] = merged_df.apply(lambda row: calculate_yoy_change(row["Clicks_before"], row["Clicks_after"]), axis=1)
                merged_df["Clicks_YOY_pct"] = merged_df.apply(lambda row: calculate_yoy_pct_change(row["Clicks_YOY"], row["Clicks_before"]), axis=1)

            # Impressions YOY
            if "Impressions_before" in merged_df.columns and "Impressions_after" in merged_df.columns:
                merged_df["Impressions_YOY"] = merged_df.apply(lambda row: calculate_yoy_change(row["Impressions_before"], row["Impressions_after"]), axis=1)
                merged_df["Impressions_YOY_pct"] = merged_df.apply(lambda row: calculate_yoy_pct_change(row["Impressions_YOY"], row["Impressions_before"]), axis=1)

            # CTR YOY
            if "CTR_before" in merged_df.columns and "CTR_after" in merged_df.columns:
                merged_df["CTR_YOY"] = merged_df.apply(lambda row: calculate_yoy_change(row["CTR_before"], row["CTR_after"]), axis=1)
                merged_df["CTR_YOY_pct"] = merged_df.apply(lambda row: calculate_yoy_pct_change(row["CTR_YOY"], row["CTR_before"]), axis=1)

            progress_bar.progress(20)

            # --- STEP 7: Compute Query Embeddings ---
            status_text.text("Computing query embeddings...")
            model = initialize_sentence_transformer()
            if model is None:
                 st.error("Sentence Transformer model failed to load. Cannot proceed with clustering.")
                 return
            # Use queries from the merged_df (includes unique from both periods)
            queries = merged_df["Query"].astype(str).unique().tolist()
            if not queries:
                st.error("No queries found in the merged data.")
                return

            query_embeddings_unique = None
            with st.spinner(f"Generating embeddings for {len(queries)} unique queries..."):
                 try:
                      query_embeddings_unique = model.encode(queries, show_progress_bar=True)
                 except Exception as encode_err:
                      st.error(f"Error during sentence embedding generation: {encode_err}")
                      return

            if query_embeddings_unique is None or len(query_embeddings_unique) != len(queries):
                 st.error("Embedding generation failed or returned unexpected results.")
                 return

            query_to_embedding = {query: emb for query, emb in zip(queries, query_embeddings_unique)}
            merged_df['query_embedding'] = merged_df['Query'].map(query_to_embedding)
            # Keep rows even if embedding failed? Or drop? For outer join, maybe keep?
            # merged_df.dropna(subset=['query_embedding'], inplace=True)
            # If keeping, clustering needs to handle missing embeddings
            valid_embedding_mask = merged_df['query_embedding'].notna()
            if not valid_embedding_mask.any():
                 st.error("Failed to generate embeddings for any query. Cannot proceed.")
                 return

            embeddings_matrix = np.vstack(merged_df.loc[valid_embedding_mask, 'query_embedding'].values)
            progress_bar.progress(35)


            # --- STEP 8: KMEANS CLUSTERING (only on rows with valid embeddings) ---
            status_text.text("Performing KMeans clustering...")
            num_to_cluster = embeddings_matrix.shape[0]
            max_k = min(30, num_to_cluster - 1) if num_to_cluster > 1 else 1
            min_k = 3

            # ... (Silhouette score calculation and optimal K selection - REMAINS THE SAME) ...
            optimal_k = min(max(min_k, max_k // 2), max_k) if max_k >= min_k else max_k # Default heuristic
            if max_k < min_k:
                 st.warning(f"Not enough unique queries with embeddings ({num_to_cluster}) for robust clustering (Min K={min_k}). Setting K={max_k if max_k > 0 else 1}.")
                 optimal_k = max_k if max_k > 0 else 1
            else:
                silhouette_scores = {}
                k_range = range(min_k, max_k + 1)
                status_text.text(f"Calculating silhouette scores for K={min_k} to K={max_k}...")
                n_samples_for_silhouette = min(5000, embeddings_matrix.shape[0])
                if n_samples_for_silhouette < embeddings_matrix.shape[0]:
                     indices = np.random.choice(embeddings_matrix.shape[0], n_samples_for_silhouette, replace=False)
                     embeddings_sample = embeddings_matrix[indices]
                else:
                     embeddings_sample = embeddings_matrix

                if embeddings_sample.shape[0] < min_k:
                    st.warning(f"Sample size ({embeddings_sample.shape[0]}) too small for min cluster count ({min_k}). Skipping silhouette calculation.")
                    optimal_k = min_k # Or max_k if smaller
                else:
                    sil_prog_bar = st.progress(0)
                    for i, k in enumerate(k_range):
                        if embeddings_sample.shape[0] < k:
                             silhouette_scores[k] = -1
                             continue
                        try:
                            km_temp = KMeans(n_clusters=k, random_state=42, n_init=5)
                            temp_labels = km_temp.fit_predict(embeddings_sample)
                            if len(set(temp_labels)) > 1:
                                 sil = silhouette_score(embeddings_sample, temp_labels)
                                 silhouette_scores[k] = sil
                            else: silhouette_scores[k] = -1
                        except Exception as e:
                            print(f"Error calculating silhouette for k={k}: {e}")
                            silhouette_scores[k] = -1
                        sil_prog_bar.progress((i + 1) / len(k_range))
                    sil_prog_bar.empty()

                    if silhouette_scores and max(silhouette_scores.values()) > -1:
                        optimal_k = max(silhouette_scores, key=silhouette_scores.get)
                        st.write(f"Optimal number of clusters suggested by Silhouette Score: {optimal_k}")
                    else:
                        optimal_k = max(min_k, math.ceil(num_to_cluster / 50)) # Default based on data size
                        optimal_k = min(optimal_k, max_k) # Ensure it doesn't exceed max_k
                        st.warning(f"Could not determine optimal K via silhouette score. Defaulting to K={optimal_k}")

            optimal_k = max(1, optimal_k)
            slider_min = max(1, min_k if num_to_cluster >= min_k else 1)
            slider_max = max(1, max_k)
            slider_default = max(slider_min, min(int(optimal_k), slider_max))

            n_clusters_selected = st.slider(
                "Select number of query clusters (K):",
                min_value=slider_min,
                max_value=slider_max,
                value=slider_default,
                key="kmeans_clusters_gsc"
            )

            status_text.text(f"Running KMeans with K={n_clusters_selected}...")
            kmeans = KMeans(n_clusters=n_clusters_selected, random_state=42, n_init='auto')
            # Fit only on valid embeddings
            cluster_labels = kmeans.fit_predict(embeddings_matrix)
            # Assign labels back to the original dataframe, using NaN for rows without embeddings
            merged_df["Cluster_ID"] = np.nan # Initialize column
            merged_df.loc[valid_embedding_mask, "Cluster_ID"] = cluster_labels
            merged_df["Cluster_ID"] = merged_df["Cluster_ID"].astype('Int64') # Use nullable integer type

            progress_bar.progress(55)


            # --- STEP 9: GPT Topic Labeling ---
            status_text.text("Generating topic labels with GPT...")
            cluster_topics = {}
            # Get unique valid cluster IDs (ignore NaN)
            valid_cluster_ids = merged_df["Cluster_ID"].dropna().unique()
            if openai_client:
                gpt_prog_bar = st.progress(0)
                status_text.text(f"Requesting topic labels from OpenAI for {len(valid_cluster_ids)} clusters...")
                for i, cluster_id in enumerate(sorted(valid_cluster_ids)):
                    # Get queries for this specific cluster_id (where it's not NaN)
                    queries_in_cluster = merged_df[merged_df["Cluster_ID"] == cluster_id]["Query"].unique().tolist()
                    if queries_in_cluster:
                        cluster_topics[cluster_id] = get_gpt_cluster_label(openai_client, queries_in_cluster, cluster_id)
                    else:
                        cluster_topics[cluster_id] = f"Cluster {cluster_id + 1} (Empty)"
                    time.sleep(0.1)
                    gpt_prog_bar.progress((i + 1) / len(valid_cluster_ids))
                gpt_prog_bar.empty()
            else:
                 st.warning("OpenAI client not initialized. Using default labels.")
                 for cluster_id in sorted(valid_cluster_ids):
                      cluster_topics[cluster_id] = f"Cluster {cluster_id + 1}"

            # Add a label for unclustered items
            cluster_topics[pd.NA] = "Unclustered / No Embedding" # Use pd.NA as key for NaN

            # Map topic labels (handles NaN in Cluster_ID correctly)
            merged_df["Query_Topic"] = merged_df["Cluster_ID"].map(cluster_topics)
            # Fill NaN topics if any slipped through (shouldn't with pd.NA key)
            merged_df["Query_Topic"].fillna("Unclustered", inplace=True)

            progress_bar.progress(70)

            # --- Display Merged Data Table ---
            st.markdown("### Combined Data with Topic Labels")
            st.markdown("Merged data (outer join) with cluster ID and GPT-generated topic labels.")
            # ... (Display order and formatting dict definition - REMAINS THE SAME as previous fix) ...
            display_order = ["Query", "Cluster_ID", "Query_Topic"]
            metrics_ordered = ["Average Position", "Clicks", "Impressions", "CTR"]
            for metric in metrics_ordered:
                 for suffix in ["_before", "_after", "_YOY", "_YOY_pct"]:
                      col = f"{metric}{suffix}"
                      if col in merged_df.columns: display_order.append(col)
            merged_df_display = merged_df[[col for col in display_order if col in merged_df.columns]]

            format_dict_merged = {}
            def add_format(col_name, fmt_str):
                 if col_name in merged_df_display.columns: format_dict_merged[col_name] = fmt_str
            add_format("Cluster_ID", "{:.0f}") # Format nullable int
            add_format("Average Position_before", "{:.1f}")
            add_format("Average Position_after", "{:.1f}")
            add_format("Position_YOY", "{:+.1f}")
            add_format("Position_YOY_pct", "{:+.1f}%")
            add_format("Clicks_before", "{:,.0f}")
            add_format("Clicks_after", "{:,.0f}")
            add_format("Clicks_YOY", "{:+,d}")
            add_format("Clicks_YOY_pct", "{:+.1f}%")
            add_format("Impressions_before", "{:,.0f}")
            add_format("Impressions_after", "{:,.0f}")
            add_format("Impressions_YOY", "{:+,d}")
            add_format("Impressions_YOY_pct", "{:+.1f}%")
            add_format("CTR_before", "{:.2f}%")
            add_format("CTR_after", "{:.2f}%")
            add_format("CTR_YOY", "{:+.2f}%")
            add_format("CTR_YOY_pct", "{:+.1f}%")

            st.dataframe(merged_df_display.style.format(format_dict_merged, na_rep="N/A"))

            # --- Step 10: Aggregated Metrics by Topic ---
            status_text.text("Aggregating metrics by topic...")
            st.markdown("### Aggregated Metrics by Topic")
            agg_dict = {}

            # Aggregation functions need to handle potential NaNs introduced by outer join
            # Use np.nansum for sums, np.nanmean for means if not using weighted avg
            if "Average Position_before" in merged_df.columns and "Impressions_before" in merged_df.columns:
                 agg_dict["Average Position_before"] = lambda x: calculate_weighted_average(x, merged_df.loc[x.index, "Impressions_before"])
            elif "Average Position_before" in merged_df.columns: agg_dict["Average Position_before"] = "mean" # nanmean handles NaN

            if "Average Position_after" in merged_df.columns and "Impressions_after" in merged_df.columns:
                 agg_dict["Average Position_after"] = lambda x: calculate_weighted_average(x, merged_df.loc[x.index, "Impressions_after"])
            elif "Average Position_after" in merged_df.columns: agg_dict["Average Position_after"] = "mean"

            if "Clicks_before" in merged_df.columns: agg_dict["Clicks_before"] = "sum" # sum ignores NaN
            if "Clicks_after" in merged_df.columns: agg_dict["Clicks_after"] = "sum"
            if "Impressions_before" in merged_df.columns: agg_dict["Impressions_before"] = "sum"
            if "Impressions_after" in merged_df.columns: agg_dict["Impressions_after"] = "sum"

            if "CTR_before" in merged_df.columns and "Impressions_before" in merged_df.columns:
                 agg_dict["CTR_before"] = lambda x: calculate_weighted_average(x, merged_df.loc[x.index, "Impressions_before"])
            elif "CTR_before" in merged_df.columns: agg_dict["CTR_before"] = "mean"

            if "CTR_after" in merged_df.columns and "Impressions_after" in merged_df.columns:
                 agg_dict["CTR_after"] = lambda x: calculate_weighted_average(x, merged_df.loc[x.index, "Impressions_after"])
            elif "CTR_after" in merged_df.columns: agg_dict["CTR_after"] = "mean"

            # Perform aggregation (groupby will exclude NaN Query_Topic if any remain)
            aggregated = merged_df.groupby("Query_Topic").agg(agg_dict).reset_index()
            aggregated.rename(columns={"Query_Topic": "Topic"}, inplace=True)

            # Calculate aggregated YOY changes *after* aggregation (using helper functions)
            if "Average Position_before" in aggregated.columns and "Average Position_after" in aggregated.columns:
                 aggregated["Position_YOY"] = aggregated.apply(lambda row: calculate_yoy_change(row["Average Position_after"], row["Average Position_before"]), axis=1) # Swapped order
                 aggregated["Position_YOY_pct"] = aggregated.apply(lambda row: calculate_yoy_pct_change(row["Position_YOY"], row["Average Position_before"]), axis=1)

            if "Clicks_before" in aggregated.columns and "Clicks_after" in aggregated.columns:
                 aggregated["Clicks_YOY"] = aggregated.apply(lambda row: calculate_yoy_change(row["Clicks_before"], row["Clicks_after"]), axis=1)
                 aggregated["Clicks_YOY_pct"] = aggregated.apply(lambda row: calculate_yoy_pct_change(row["Clicks_YOY"], row["Clicks_before"]), axis=1)

            if "Impressions_before" in aggregated.columns and "Impressions_after" in aggregated.columns:
                 aggregated["Impressions_YOY"] = aggregated.apply(lambda row: calculate_yoy_change(row["Impressions_before"], row["Impressions_after"]), axis=1)
                 aggregated["Impressions_YOY_pct"] = aggregated.apply(lambda row: calculate_yoy_pct_change(row["Impressions_YOY"], row["Impressions_before"]), axis=1)

            if "CTR_before" in aggregated.columns and "CTR_after" in aggregated.columns:
                 aggregated["CTR_YOY"] = aggregated.apply(lambda row: calculate_yoy_change(row["CTR_before"], row["CTR_after"]), axis=1)
                 aggregated["CTR_YOY_pct"] = aggregated.apply(lambda row: calculate_yoy_pct_change(row["CTR_YOY"], row["CTR_before"]), axis=1)


            progress_bar.progress(85)

            # Reorder columns for the aggregated table
            new_order_agg = ["Topic"]
            agg_yoy_cols_ordered = []
            for metric in metrics_ordered:
                 before_col = f"{metric}_before"
                 after_col = f"{metric}_after"
                 yoy_col = f"{metric}_YOY"
                 yoy_pct_col = f"{metric}_YOY_pct"
                 if before_col in aggregated.columns: new_order_agg.append(before_col)
                 if after_col in aggregated.columns: new_order_agg.append(after_col)
                 if yoy_col in aggregated.columns: new_order_agg.append(yoy_col)
                 if yoy_pct_col in aggregated.columns:
                      new_order_agg.append(yoy_pct_col)
                      agg_yoy_cols_ordered.append((yoy_pct_col, metric))

            aggregated = aggregated[[col for col in new_order_agg if col in aggregated.columns]]

            # Define formatting for aggregated metrics display
            format_dict_agg = {}
            def add_agg_format(col_name, fmt_str):
                 if col_name in aggregated.columns: format_dict_agg[col_name] = fmt_str
            # ... (format_dict_agg definition REMAINS THE SAME as previous fix) ...
            add_agg_format("Average Position_before", "{:.1f}")
            add_agg_format("Average Position_after", "{:.1f}")
            add_agg_format("Position_YOY", "{:+.1f}")
            add_agg_format("Position_YOY_pct", "{:+.1f}%")
            add_agg_format("Clicks_before", "{:,.0f}")
            add_agg_format("Clicks_after", "{:,.0f}")
            add_agg_format("Clicks_YOY", "{:+,d}")
            add_agg_format("Clicks_YOY_pct", "{:+.1f}%")
            add_agg_format("Impressions_before", "{:,.0f}")
            add_agg_format("Impressions_after", "{:,.0f}")
            add_agg_format("Impressions_YOY", "{:+,d}")
            add_agg_format("Impressions_YOY_pct", "{:+.1f}%")
            add_agg_format("CTR_before", "{:.2f}%")
            add_agg_format("CTR_after", "{:.2f}%")
            add_agg_format("CTR_YOY", "{:+.2f}%")
            add_agg_format("CTR_YOY_pct", "{:+.1f}%")


            display_count = st.number_input("Number of aggregated topics to display:", min_value=1, value=min(aggregated.shape[0], 50), max_value=aggregated.shape[0])
            # Exclude 'Unclustered' topic from default sort order if desired, or sort by a metric
            sort_metric_agg = "Impressions_after" if "Impressions_after" in aggregated.columns else "Topic"
            aggregated_sorted = aggregated.sort_values(by=sort_metric_agg, ascending=False, na_position='last')
            st.dataframe(aggregated_sorted.head(display_count).style.format(format_dict_agg, na_rep="N/A"))
            progress_bar.progress(90)

            # Step 11: Visualization
            status_text.text("Generating visualizations...")
            st.markdown("### YOY % Change by Topic for Each Metric")

            # Exclude 'Unclustered' from default selection for plotting maybe?
            default_topics = [t for t in aggregated["Topic"].unique() if t != "Unclustered / No Embedding"]
            available_topics = aggregated["Topic"].unique().tolist()
            selected_topics = st.multiselect("Select topics to display on the chart:", options=available_topics, default=default_topics)

            vis_data = []
            # Use aggregated_sorted to reflect sorting in plot data potentially
            for idx, row in aggregated_sorted.iterrows(): # Iterate sorted df
                topic = row["Topic"]
                if topic not in selected_topics: continue
                for yoy_pct_col, metric_name in agg_yoy_cols_ordered:
                     if yoy_pct_col in row and pd.notna(row[yoy_pct_col]) and np.isfinite(row[yoy_pct_col]): # Exclude NaN and Inf
                         vis_data.append({"Topic": topic, "Metric": metric_name, "YOY % Change": row[yoy_pct_col]})

            if vis_data:
                 vis_df = pd.DataFrame(vis_data)
                 # Use category orders based on the sorted aggregated data (or selected topics)
                 topic_order_plot = [t for t in aggregated_sorted['Topic'] if t in selected_topics]
                 fig = px.bar(vis_df, x="Topic", y="YOY % Change", color="Metric", barmode="group",
                              title="YOY % Change by Topic for Each Metric",
                              labels={"YOY % Change": "YOY % Change (%)", "Topic": "GPT-Generated Topic"},
                              category_orders={"Topic": topic_order_plot}) # Use sorted order
                 fig.update_layout(height=600)
                 st.plotly_chart(fig, use_container_width=True)
            else:
                 st.info("No finite YOY % change data available to plot (might be NaN, Inf or missing).")

            progress_bar.progress(100)
            status_text.text("Analysis Complete!")

        except FileNotFoundError:
            st.error("Error: One or both CSV files not found. Please ensure they are uploaded correctly.")
        except pd.errors.EmptyDataError:
            st.error("Error: One or both CSV files appear to be empty.")
        except KeyError as e:
             st.error(f"Error: A required column is missing or named incorrectly after standardization: {e}. Please check the CSV files and column names.")
             import traceback
             st.error(f"Full Traceback: {traceback.format_exc()}")
        except Exception as e:
            st.error(f"An unexpected error occurred during analysis: {e}")
            import traceback
            st.error(f"Full Traceback: {traceback.format_exc()}")
        finally:
             progress_bar.empty()
             status_text.empty()

    else:
        st.info("Please upload both GSC CSV files to start the analysis.")


# ------------------------------------
# NEW TOOL: Vector Embeddings Scatterplot
# ------------------------------------
# Cache the SentenceTransformer model so it loads only once
@st.cache_resource
def load_sentence_transformer_semantic(model_name='all-MiniLM-L6-v2'): # Renamed to avoid conflict if used elsewhere
    print(f"Loading Sentence Transformer for Semantic Clustering: {model_name}")
    return SentenceTransformer(model_name)

@st.cache_data # Cache data loading
def load_semantic_data(uploaded_file):
    """Loads Screaming Frog crawl data from an uploaded CSV file for semantic clustering."""
    df = pd.read_csv(uploaded_file)
    # Look for common variations of URL and Content column names
    url_col = None
    content_col = None
    possible_url_cols = ['Address', 'URL', 'url']
    possible_content_cols = ['Content', 'content', 'Text Content', 'Body'] # Add more if needed

    for col in possible_url_cols:
        if col in df.columns:
            url_col = col
            break
    for col in possible_content_cols:
        if col in df.columns:
            content_col = col
            break

    if not url_col:
        raise ValueError(f"CSV must contain a URL column (e.g., {possible_url_cols}). Found: {df.columns}")
    if not content_col:
        raise ValueError(f"CSV must contain a Content column (e.g., {possible_content_cols}). Found: {df.columns}")

    # Rename to standard names for consistency
    df = df.rename(columns={url_col: 'URL', content_col: 'Content'})

    # Drop rows with missing content, as they cannot be vectorized
    df.dropna(subset=['Content'], inplace=True)
    if df.empty:
        raise ValueError("No rows with content found after removing missing values.")

    return df[['URL', 'Content']]

# Use _model suffix to avoid potential conflicts
@st.cache_data
def vectorize_pages(_contents, _model):
    """Converts page content into vector embeddings using a transformer model."""
    print(f"Vectorizing {len(_contents)} pages...")
    # Ensure contents are strings
    contents_str = [str(c) for c in _contents]
    embeddings = _model.encode(contents_str, convert_to_numpy=True, show_progress_bar=True)
    print("Vectorization complete.")
    return embeddings

@st.cache_data
def reduce_dimensions_semantic(_embeddings, n_components=2, method='UMAP', n_neighbors=15, min_dist=0.1):
    """Reduces vector dimensionality using UMAP or PCA."""
    print(f"Reducing dimensions using {method}...")
    if method == 'UMAP':
        # Ensure n_neighbors is less than the number of samples
        actual_n_neighbors = min(n_neighbors, _embeddings.shape[0] - 1)
        if actual_n_neighbors < 2:
             st.warning("Too few samples for reliable UMAP, using n_neighbors=2.")
             actual_n_neighbors = 2
        if actual_n_neighbors != n_neighbors:
             print(f"Adjusted UMAP n_neighbors to {actual_n_neighbors}")

        reducer = umap.UMAP(n_components=n_components, n_neighbors=actual_n_neighbors, min_dist=min_dist, metric='cosine', random_state=42)
    elif method == 'PCA':
        reducer = PCA(n_components=n_components, random_state=42)
    else:
        raise ValueError("Invalid reduction method. Choose 'UMAP' or 'PCA'.")

    reduced_embeddings = reducer.fit_transform(_embeddings)
    print("Dimension reduction complete.")
    return reduced_embeddings

@st.cache_data
def cluster_embeddings_semantic(_embeddings, n_clusters=5):
    """Clusters embeddings using KMeans."""
    print(f"Clustering into {n_clusters} clusters using KMeans...")
    # Ensure n_clusters is not more than the number of samples
    actual_n_clusters = min(n_clusters, _embeddings.shape[0])
    if actual_n_clusters != n_clusters:
         st.warning(f"Reduced number of clusters to {actual_n_clusters} as it cannot exceed the number of data points.")
         n_clusters = actual_n_clusters
    if n_clusters < 1:
         st.error("Number of clusters must be at least 1.")
         return None # Indicate failure

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(_embeddings)
    print("Clustering complete.")
    return labels

def plot_embeddings_interactive(embeddings_2d, labels, urls):
    """
    Creates an interactive UMAP/PCA scatter plot using Plotly Express.
    Hovering over a point displays the corresponding URL.
    """
    if embeddings_2d is None or labels is None:
        st.error("Cannot plot - missing reduced embeddings or labels.")
        return None

    # Create a DataFrame with the coordinates, cluster labels, and URLs
    df_plot = pd.DataFrame({
        "x": embeddings_2d[:, 0],
        "y": embeddings_2d[:, 1],
        "Cluster ID": labels,
        "URL": urls
    })
    df_plot["Cluster"] = df_plot["Cluster ID"].apply(lambda l: f"Cluster {l}") # String label for coloring/legend

    # Create an interactive scatter plot
    fig = px.scatter(
        df_plot,
        x="x",
        y="y",
        color="Cluster", # Use string label for color legend
        hover_data=["URL"], # Show URL on hover
        title="Interactive Scatterplot of Website Pages by Semantic Content"
    )
    fig.update_traces(marker=dict(size=5)) # Smaller markers
    fig.update_layout(
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        legend_title_text='Clusters',
        height=700
    )
    return fig

def semantic_clustering_page():
    st.header("Site Focus Visualizer (Semantic Clustering)")
    st.markdown(
        """
        Upload a CSV file (e.g., from Screaming Frog) containing URLs and their text content.
        This tool visualizes the semantic relationships between pages based on their content similarity.
        It helps identify the main topical clusters within your website.
        The CSV must include columns recognizable as **URL** (e.g., 'Address', 'URL') and **Content** (e.g., 'Content', 'Text Content').
        """
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], key="semantic_cluster_upload")

    if uploaded_file is not None:
        data = None
        try:
            with st.spinner("Loading and validating data..."):
                 data = load_semantic_data(uploaded_file)
                 st.write(f"Data loaded: {len(data)} pages with content.")
                 st.dataframe(data.head())
        except ValueError as ve:
             st.error(f"Data Loading Error: {ve}")
             return # Stop execution if data loading fails
        except Exception as e:
            st.error(f"An unexpected error occurred during data loading: {e}")
            return

        # Extract the URLs and content for processing
        urls = data['URL'].tolist()
        contents = data['Content'].tolist()

        # Load the transformer model (cached)
        model = load_sentence_transformer_semantic()

        embeddings = None
        with st.spinner(f"Vectorizing {len(contents)} page contents... This may take time."):
            embeddings = vectorize_pages(contents, model)

        if embeddings is None or embeddings.shape[0] == 0:
             st.error("Vectorization failed or resulted in no embeddings.")
             return

        # Dimension Reduction Options
        st.sidebar.subheader("Visualization Settings")
        dim_reduction_method = st.sidebar.selectbox("Dimension Reduction Method:", options=["UMAP", "PCA"], index=0, key="sem_dim_red")
        n_components = 2 # Keep 2D for scatter plot

        # Clustering Options
        max_clusters = max(2, embeddings.shape[0] // 2) # Heuristic for max clusters
        n_clusters = st.sidebar.number_input("Select number of clusters:", min_value=2, max_value=max_clusters, value=min(5, max_clusters), step=1, key="sem_n_cluster")

        reduced_embeddings = None
        with st.spinner(f"Reducing dimensions using {dim_reduction_method}..."):
             reduced_embeddings = reduce_dimensions_semantic(embeddings, n_components=n_components, method=dim_reduction_method)

        if reduced_embeddings is None:
             st.error("Dimension reduction failed.")
             return

        labels = None
        with st.spinner(f"Clustering embeddings into {n_clusters} clusters..."):
            # Use the reduced embeddings for clustering (often faster and can be effective)
            labels = cluster_embeddings_semantic(reduced_embeddings, n_clusters)

        if labels is None:
            st.error("Clustering failed.")
            return

        st.success("Processing complete! Generating plot...")

        # Use the interactive Plotly function
        fig = plot_embeddings_interactive(reduced_embeddings, labels, urls)
        if fig:
             st.plotly_chart(fig, use_container_width=True)
        else:
             st.error("Failed to generate plot.")


# ------------------------------------
# Entity Relationship Graph Generator
# ------------------------------------

# --- Helper Functions (OUTSIDE entity_relationship_graph_page) ---

def extract_entities_and_relationships(sentences, nlp):
    """Extracts entities and relationships - (Modified from original)."""
    entities = [] # List of (text, label) tuples
    relationships = [] # List of (entity1_text, entity2_text) tuples
    entity_counts = Counter()

    relevant_labels = {"PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE"} # Define relevant types

    for sentence in sentences:
        doc = nlp(sentence)
        sentence_entities_raw = [(ent.text.strip(), ent.label_) for ent in doc.ents if ent.label_ in relevant_labels and len(ent.text.strip()) > 1]

        # Add unique entities found in this sentence to overall list and count
        current_sentence_unique_entities = set()
        for entity_text, entity_label in sentence_entities_raw:
             # Normalize text slightly (e.g., lowercasing) for counting and relationship consistency? Optional.
             normalized_text = entity_text # Keep original case for now
             if (normalized_text, entity_label) not in current_sentence_unique_entities:
                 entities.append((normalized_text, entity_label))
                 entity_counts[normalized_text] += 1
                 current_sentence_unique_entities.add((normalized_text, entity_label))

        # Co-occurrence within sentences (using original case text)
        sentence_entity_texts = [ent[0] for ent in sentence_entities_raw] # Get just the text
        # Create unique pairs within the sentence
        for i in range(len(sentence_entity_texts)):
            for j in range(i + 1, len(sentence_entity_texts)):
                 ent1, ent2 = sorted((sentence_entity_texts[i], sentence_entity_texts[j])) # Sort to make edge unique (A,B) == (B,A)
                 if ent1 != ent2: # Avoid self-loops if the same entity text appears twice
                      relationships.append((ent1, ent2))

    # Deduplicate entities list keeping the first label encountered (or most common?) - simple deduplication for now
    unique_entities = list({ent[0]: ent for ent in entities}.values())

    return unique_entities, relationships, entity_counts

def create_entity_graph(unique_entities, relationships, entity_counts):
    """Creates a NetworkX graph (Modified from original)."""
    G = nx.Graph()

    # Add nodes with attributes (type and count)
    for entity_text, entity_type in unique_entities:
        # Use the count from the counter based on the text
        count = entity_counts.get(entity_text, 1) # Default to 1 if somehow missing
        G.add_node(entity_text, type=entity_type, count=count)

    # Add edges with weights based on co-occurrence frequency
    relationship_counts = Counter(relationships) # Count tuples like (ent1, ent2)
    for (entity1, entity2), count in relationship_counts.items():
        # Ensure both nodes exist in the graph before adding edge
        if G.has_node(entity1) and G.has_node(entity2):
            G.add_edge(entity1, entity2, weight=count)

    # Remove isolated nodes (nodes with no edges) - Optional
    # isolated_nodes = list(nx.isolates(G))
    # G.remove_nodes_from(isolated_nodes)
    # print(f"Removed {len(isolated_nodes)} isolated nodes.")

    return G

def visualize_graph(G, source_identifier): # Changed name from website_url
    """Visualizes the ERG using Matplotlib."""
    if not G.nodes():
         st.warning("Graph has no nodes to visualize.")
         return

    plt.style.use('seaborn-v0_8-whitegrid') # Use a clean style
    plt.figure(figsize=(18, 14)) # Make figure larger

    # Use a layout that spaces out nodes better, increase iterations
    pos = nx.spring_layout(G, seed=42, k=0.6, iterations=100)

    # Node sizing based on count (log scale can help if counts vary wildly)
    min_node_size = 100
    max_node_size = 5000
    node_sizes = [G.nodes[node]['count'] for node in G.nodes()]
    # Optional: Log scaling or MinMax scaling for size
    if node_sizes:
        # Simple scaling: map range of counts to range of sizes
        min_count = min(node_sizes) if node_sizes else 1
        max_count = max(node_sizes) if node_sizes else 1
        if max_count > min_count:
             node_sizes = [min_node_size + (count - min_count) * (max_node_size - min_node_size) / (max_count - min_count) for count in node_sizes]
        else:
             node_sizes = [min_node_size] * len(node_sizes) # All same size if counts are identical
    else:
        node_sizes = [min_node_size] * G.number_of_nodes()


    # Node coloring based on entity type
    color_map = {
        'ORG': 'skyblue', 'GPE': 'lightgreen', 'LOC': 'lightcoral',
        'PERSON': 'gold', 'PRODUCT': 'lightsalmon', 'EVENT': 'plum',
        'WORK_OF_ART': 'lightpink', 'NORP': 'khaki', 'FAC': 'wheat',
        'LAW': 'lightgray', 'LANGUAGE': 'lightblue',
        # Add more colors as needed
    }
    default_color = 'silver'
    node_colors = [color_map.get(G.nodes[node]['type'], default_color) for node in G.nodes()]

    # Edge width based on weight (co-occurrence count)
    edge_weights = [G.edges[edge]['weight'] for edge in G.edges()]
    # Scale edge weights if needed
    max_weight = max(edge_weights) if edge_weights else 1
    scaled_widths = [(w / max_weight * 5.0) + 0.5 for w in edge_weights] # Scale 0.5 to 5.5

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=scaled_widths, edge_color='grey', alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif", font_weight='bold')

    # Add title and remove axes
    title = f"Entity Relationship Graph for: {source_identifier}"
    plt.title(title, fontsize=18)
    plt.axis("off")
    st.pyplot(plt) # Display in Streamlit


def entity_relationship_graph_page():
    st.header("Entity Relationship Graph Generator")
    st.markdown("""
    Enter a single URL to scrape its content and visualize the relationships between the named entities (people, organizations, locations, etc.) found on the page.
    Nodes represent entities, sized by frequency. Edges connect entities that appear together in the same sentence, with thickness indicating co-occurrence frequency.
    """)
    url = st.text_input("Enter a website URL:", key="erg_url")

    if url: # Simplified URL handling
        if st.button("Generate Graph", key="erg_button"):
            sentences = []
            text = ""
            with st.spinner(f"Scraping content from {url}..."):
                text = extract_text_from_url(url) # Use original function
                if text:
                    # Sentence splitting (ensure robust splitting)
                    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s', text) # Improved splitter
                    sentences = [s.strip() for s in sentences if s.strip() and len(s.split()) > 2] # Basic filter
                    st.write(f"Found {len(sentences)} sentences.")
                else:
                    st.error(f"Could not retrieve or process content from {url}.")
                    return # Exit if no content

            if sentences:
                with st.spinner("Extracting entities and relationships..."):
                    nlp_model = load_spacy_model()
                    if not nlp_model:
                         st.error("Could not load spaCy model.")
                         return
                    # Use the helper function to get entities, relationships, and counts
                    unique_entities, relationships, entity_counts = extract_entities_and_relationships(sentences, nlp_model)
                    st.write(f"Found {len(unique_entities)} unique entities and {len(relationships)} co-occurrences.")

                    if not unique_entities:
                         st.warning("No relevant entities found on the page to build a graph.")
                         return

                    # Create the graph using the helper function
                    graph = create_entity_graph(unique_entities, relationships, entity_counts)

                with st.spinner("Visualizing graph..."):
                    # Pass the URL as the identifier for the title
                    visualize_graph(graph, url)
            else:
                st.warning("No sentences extracted from the URL content.")
    else:
        st.info("Please enter a URL to generate the graph.")


# ------------------------------------
# SEMRush Organic Pages Sub-Directories
# ------------------------------------

def semrush_organic_pages_by_subdirectory_page():
    st.header("SEMRush Organic Pages by Top Subdirectory")
    st.markdown("""
    Upload your SEMRush Organic Pages report (Excel format) to see data aggregated by **top-level subdirectory**.
    The file should contain a 'URL' column plus numeric metric columns (e.g. 'Traffic', 'Number of Keywords', etc.).
    The tool sums the metrics for all pages within the same initial subdirectory (e.g., all pages under `/blog/`).
    """)

    uploaded_file = st.file_uploader(
        "Upload SEMRush Organic Pages Excel file",
        type=["xlsx"],
        key="semrush_file_top" # Unique key
    )

    if uploaded_file is not None:
        try:
            # Read the Excel file into a DataFrame
            df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error reading the Excel file: {e}")
            return

        # Find URL column (case-insensitive)
        url_col = None
        for col in ['URL', 'Url', 'Address']:
            if col in df.columns:
                url_col = col
                break
        if not url_col:
            st.error("No 'URL' column found (checked for 'URL', 'Url', 'Address').")
            return
        # Rename to standard 'URL'
        if url_col != 'URL':
             df = df.rename(columns={url_col: 'URL'})

        # Identify and convert numeric columns
        numeric_cols = []
        for col in df.columns:
            if col == "URL": continue # Skip URL column
            try:
                # Attempt conversion, coercing errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Check if the column has at least one non-NaN value after conversion
                if df[col].notna().any():
                     numeric_cols.append(col)
                     # Fill NaNs with 0 after identifying as numeric
                     df[col] = df[col].fillna(0)
            except Exception:
                 # Ignore columns that fail conversion completely
                 continue

        if not numeric_cols:
             st.warning("No numeric metric columns found to aggregate.")
             # Continue to show subdirectory extraction if needed, but aggregation won't work

        # Helper function to extract the first subdirectory from a URL
        def get_subdirectory(url):
            try:
                # Handle potential non-string URLs
                if not isinstance(url, str):
                     return "Invalid URL format"
                parsed = urlparse(url)
                # Split path, remove empty segments (e.g., leading/trailing slashes)
                path_segments = [seg for seg in parsed.path.split('/') if seg]
                # Return first segment or 'Root' if no path segments
                return "/" + path_segments[0] if path_segments else "/" # Represent root as '/'
            except Exception:
                return "URL Parsing Error"

        # Create a new column for the first subdirectory
        df["Subdirectory"] = df["URL"].apply(get_subdirectory)

        st.markdown("### Data Preview with Subdirectory")
        st.dataframe(df[['URL', 'Subdirectory'] + numeric_cols].head())

        st.markdown("### Aggregated Metrics by Top-Level Subdirectory")
        if numeric_cols:
             # Build an aggregation dictionary for numeric columns (sum them)
             agg_dict = {col: "sum" for col in numeric_cols}
             # Add a count of URLs per subdirectory
             agg_dict['URL'] = pd.NamedAgg(column='URL', aggfunc='count')

             # Group by Subdirectory and apply the aggregator
             subdir_agg = df.groupby("Subdirectory").agg(agg_dict).reset_index()
             subdir_agg = subdir_agg.rename(columns={'URL': 'Page Count'}) # Rename URL count column

             # Sort by a prominent metric if available (e.g., Traffic), else by Page Count
             sort_col_agg = None
             if "Traffic" in subdir_agg.columns:
                 sort_col_agg = "Traffic"
             elif "Page Count" in subdir_agg.columns:
                  sort_col_agg = "Page Count"

             if sort_col_agg:
                  subdir_agg = subdir_agg.sort_values(by=sort_col_agg, ascending=False)


             # Display the aggregated data - add formatting
             format_dict_sem = {col: "{:,.0f}" for col in numeric_cols} # Default format for numeric
             format_dict_sem["Page Count"] = "{:,.0f}"
             if "Traffic" in format_dict_sem: format_dict_sem["Traffic"] = "{:,.0f}" # Ensure Traffic is integer
             if "Number of Keywords" in format_dict_sem: format_dict_sem["Number of Keywords"] = "{:,.0f}"

             st.dataframe(subdir_agg.style.format(format_dict_sem, na_rep="N/A"))

             # Example: Show a bar chart for Traffic by Subdirectory if "Traffic" exists
             plot_metric = None
             if "Traffic" in subdir_agg.columns:
                 plot_metric = "Traffic"
             elif "Page Count" in subdir_agg.columns:
                  plot_metric = "Page Count" # Fallback to page count

             if plot_metric:
                 st.markdown(f"### {plot_metric} by Subdirectory Chart")
                 # Limit to top N subdirectories for clarity
                 top_n_plot = 25
                 df_plot = subdir_agg.head(top_n_plot)
                 fig = px.bar(
                     df_plot,
                     x="Subdirectory",
                     y=plot_metric,
                     title=f"Top {top_n_plot} Subdirectories by {plot_metric}",
                     labels={"Subdirectory": "Top-Level Subdirectory", plot_metric: plot_metric}
                 )
                 fig.update_layout(xaxis={'categoryorder':'total descending'})
                 st.plotly_chart(fig, use_container_width=True)

        else:
             st.warning("No numeric columns identified to aggregate. Showing subdirectory list:")
             st.dataframe(df[["URL", "Subdirectory"]].drop_duplicates())

    else:
        st.info("Please upload a SEMRush Organic Pages Excel file to begin the analysis.")



# ------------------------------------
# SEMRush Organic Pages Hierarchical Sub-Directories
# ------------------------------------

def semrush_hierarchical_subdirectories_minimal_no_leaf_with_intent_filter():
    st.header("SEMRush Hierarchical Aggregation (No Leaf Nodes)")
    st.markdown("""
    Aggregates SEMRush metrics ('Number of Keywords', 'Traffic', and User Intent Traffic) across **all levels** of the site's subdirectory structure, excluding the final pages/directories (leaf nodes).

    **Process:**
    1. Keeps **URL**, **Number of Keywords**, **Traffic**, and any detected User Intent Traffic columns (e.g., 'Traffic with informational intents in top 20').
    2. Expands each URL into its full hierarchical path segments (e.g., `/folder/`, `/folder/subfolder/`).
    3. Identifies and removes rows corresponding to the final level (leaf nodes).
    4. Aggregates (sums) the metrics for each remaining non-leaf hierarchical path.
    5. Provides an interactive chart of the results, filterable by User Intent.
    """)

    uploaded_file = st.file_uploader(
        "Upload SEMRush Excel with 'URL', 'Number of Keywords', 'Traffic' and optional User Intent Traffic columns",
        type=["xlsx"],
        key="semrush_hierarchical_upload" # Unique key
    )

    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error reading Excel file: {e}")
            return

        # --- Column Identification & Selection ---
        # Find URL column (case-insensitive)
        url_col = None
        for col in ['URL', 'Url', 'Address']:
            if col in df.columns:
                url_col = col
                break
        if not url_col:
            st.error("No 'URL' column found (checked for 'URL', 'Url', 'Address').")
            return
        # Rename to standard 'URL'
        if url_col != 'URL':
             df = df.rename(columns={url_col: 'URL'})

        # Find standard metric columns (case-insensitive)
        keywords_col = None
        for col in ['Number of Keywords', 'Keywords', 'Keyword Count']:
             if col in df.columns:
                  keywords_col = col
                  break
        traffic_col = None
        for col in ['Traffic', 'Estimated Traffic', 'traffic']:
            if col in df.columns:
                 traffic_col = col
                 break

        # Check required metric columns
        if not keywords_col: st.warning("Column 'Number of Keywords' not found.")
        if not traffic_col: st.warning("Column 'Traffic' not found.")

        # User intent traffic columns (case-sensitive as per SEMrush export usually)
        user_intent_options = [
            "Traffic with commercial intents in top 20",
            "Traffic with informational intents in top 20",
            "Traffic with navigational intents in top 20",
            "Traffic with transactional intents in top 20",
            "Traffic with unknown intents in top 20" # Less common, but include
        ]
        available_intent_cols = [col for col in user_intent_options if col in df.columns]

        # Select columns to keep
        cols_to_keep = ['URL']
        if keywords_col: cols_to_keep.append(keywords_col)
        if traffic_col: cols_to_keep.append(traffic_col)
        cols_to_keep.extend(available_intent_cols)

        df_selected = df[cols_to_keep].copy() # Work on a copy

        # Rename found metric columns to standard names
        rename_map = {}
        if keywords_col: rename_map[keywords_col] = "Number of Keywords"
        if traffic_col: rename_map[traffic_col] = "Traffic"
        df_selected = df_selected.rename(columns=rename_map)


        # Convert non-URL columns to numeric
        numeric_cols_hier = [col for col in df_selected.columns if col != 'URL']
        for col in numeric_cols_hier:
            df_selected[col] = pd.to_numeric(df_selected[col], errors='coerce').fillna(0)

        # --- Hierarchical Path Expansion ---
        # Function to get all hierarchical subdirectory levels from a URL
        def get_subdirectory_levels(url):
            try:
                if not isinstance(url, str): return ["/"] # Handle non-strings
                parsed = urlparse(url)
                segments = [seg for seg in parsed.path.strip("/").split("/") if seg]
                if not segments:
                    return ["/"] # Represent root
                paths = ["/"] # Always include root
                for i in range(len(segments)):
                    # Create path like /seg1, /seg1/seg2
                    path = "/" + "/".join(segments[:i+1])
                    paths.append(path)
                return paths
            except Exception:
                 return ["/"] # Default to root on error

        # Explode each URL into its hierarchical subdirectory levels
        exploded_rows = []
        with st.spinner("Expanding URLs into hierarchical paths..."):
            for _, row in df_selected.iterrows():
                url_levels = get_subdirectory_levels(row["URL"])
                for level in url_levels:
                    new_row = row.drop("URL").copy() # Drop original URL, keep metrics
                    new_row["Hierarchical_Path"] = level
                    exploded_rows.append(new_row)

        if not exploded_rows:
             st.error("Failed to expand URLs into paths.")
             return

        df_exploded = pd.DataFrame(exploded_rows)

        # --- Identify and Remove Leaf Nodes ---
        with st.spinner("Identifying and removing leaf nodes..."):
            all_paths = set(df_exploded["Hierarchical_Path"].unique())
            # A path is a leaf if no *other* path starts with (path + "/")
            # Need to handle the root "/" case carefully
            def is_leaf(path):
                if path == "/": # Root is never a leaf unless it's the *only* path
                     return len(all_paths) == 1
                prefix = path.rstrip("/") + "/"
                # Check if any other path starts with this prefix
                return not any(p.startswith(prefix) for p in all_paths if p != path)

            df_exploded["IsLeaf"] = df_exploded["Hierarchical_Path"].apply(is_leaf)
            df_filtered = df_exploded[~df_exploded["IsLeaf"]].copy() # Keep non-leaf nodes
            # Drop the helper column
            df_filtered.drop(columns=["IsLeaf"], inplace=True)

        st.markdown("### Expanded Data Preview (After Removing Leaf Nodes)")
        st.dataframe(df_filtered.head())

        # --- Aggregate Metrics ---
        st.markdown("### Aggregated Data by Hierarchical Subdirectory (No Leaf Nodes)")
        if numeric_cols_hier and not df_filtered.empty:
             # Aggregate all available numeric columns by hierarchical subdirectory
             df_agg = df_filtered.groupby("Hierarchical_Path")[numeric_cols_hier].sum().reset_index()

             # Sort by Traffic or Keywords if available
             sort_col_hier = None
             if "Traffic" in df_agg.columns: sort_col_hier = "Traffic"
             elif "Number of Keywords" in df_agg.columns: sort_col_hier = "Number of Keywords"

             if sort_col_hier:
                  df_agg = df_agg.sort_values(by=sort_col_hier, ascending=False)

             # Formatting for display
             format_dict_hier = {col: "{:,.0f}" for col in numeric_cols_hier}
             st.dataframe(df_agg.style.format(format_dict_hier, na_rep="N/A"))

             # --- Plotly Chart Options ---
             st.markdown("### Plotly Chart")
             st.write("Visualize aggregated metrics. Filter by User Intent if available.")

             plot_options = []
             if "Traffic" in df_agg.columns: plot_options.append("Overall Traffic")
             if "Number of Keywords" in df_agg.columns: plot_options.append("Number of Keywords")
             if available_intent_cols: plot_options.append("User Intent Traffic (Grouped)")

             if not plot_options:
                  st.warning("No metrics available for plotting.")
             else:
                  chart_type = st.selectbox("Select Chart to Display:", options=plot_options)

                  fig = None
                  # Limit paths shown in chart for readability
                  top_n_paths_chart = 50
                  df_plot_data = df_agg.head(top_n_paths_chart)

                  if chart_type == "User Intent Traffic (Grouped)" and available_intent_cols:
                       # Melt the DataFrame for grouped bar chart by intent
                       df_melt = df_plot_data.melt(
                            id_vars=["Hierarchical_Path"],
                            value_vars=available_intent_cols, # Only melt intent columns
                            var_name="Intent Type",
                            value_name="Intent Traffic"
                       )
                       # Filter out zero values for cleaner plot
                       df_melt = df_melt[df_melt["Intent Traffic"] > 0]

                       if not df_melt.empty:
                            fig = px.bar(
                                df_melt,
                                x="Hierarchical_Path",
                                y="Intent Traffic",
                                color="Intent Type",
                                barmode="group",
                                title=f"Top {top_n_paths_chart} Hierarchical Subdirectories by User Intent Traffic (No Leaf Nodes)",
                                labels={"Hierarchical_Path": "Subdirectory Path", "Intent Traffic": "Aggregated Traffic"}
                            )
                       else: st.warning("No non-zero User Intent Traffic data to plot.")

                  elif chart_type == "Overall Traffic" and "Traffic" in df_plot_data.columns:
                       fig = px.bar(
                            df_plot_data,
                            x="Hierarchical_Path",
                            y="Traffic",
                            title=f"Top {top_n_paths_chart} Hierarchical Subdirectories by Overall Traffic (No Leaf Nodes)",
                            labels={"Hierarchical_Path": "Subdirectory Path", "Traffic": "Aggregated Traffic"}
                       )
                  elif chart_type == "Number of Keywords" and "Number of Keywords" in df_plot_data.columns:
                       fig = px.bar(
                            df_plot_data,
                            x="Hierarchical_Path",
                            y="Number of Keywords",
                            title=f"Top {top_n_paths_chart} Hierarchical Subdirectories by Number of Keywords (No Leaf Nodes)",
                            labels={"Hierarchical_Path": "Subdirectory Path", "Number of Keywords": "Aggregated Keywords"}
                       )

                  if fig:
                       fig.update_layout(height=800, xaxis={'categoryorder':'total descending'})
                       st.plotly_chart(fig, use_container_width=True)
                  elif chart_type == "User Intent Traffic (Grouped)":
                       # Handle case where melt was empty
                       st.info("No data to display for User Intent Traffic chart (might be all zeros).")


        elif df_filtered.empty:
             st.warning("No non-leaf subdirectory paths found after processing.")
        else: # No numeric columns
             st.warning("No numeric columns found to aggregate or plot.")
             st.dataframe(df_filtered[['Hierarchical_Path']].drop_duplicates())

    else:
        st.info("Please upload an Excel file to begin the analysis.")


# ------------------------------------
# Main Streamlit App
# ------------------------------------
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
        footer {visibility: hidden !important;} /* Hide Streamlit footer */
        /* Hide GitHub icon */
        .stApp > header > div:nth-child(3) { display: none !important; }
        /* Hide Streamlit menu button */
        button[title="View fullscreen"] { display: none !important; }
        div[data-testid="stToolbar"] { display: none !important; } /* More robust toolbar hiding */

        /* Attempt to hide Streamlit decoration bar */
        div[data-testid="stDecoration"] { display: none !important; }

        /* Attempt to hide Streamlit Cloud connection/login elements */
        a[href*='streamlit.io/cloud'],
        div[class*='_profileContainer'], /* Target class containing login info */
        div[class*='stActionButton'] > button[title*='Manage app'], /* Target Manage App button */
        div[data-testid="stStatusWidget"] /* Target status widget (connecting, etc.) */
        {
            display: none !important;
            visibility: hidden !important;
        }

        /* Reduce top padding */
        div.block-container {padding-top: 1rem;}
        </style>
        """
    st.markdown(hide_streamlit_elements, unsafe_allow_html=True)
    create_navigation_menu(logo_url)
    st.sidebar.header("Semantic Search SEO Analysis Tools")
    tool = st.sidebar.selectbox("Select Tool:", [
        "URL Analysis Dashboard",
        "Cosine Similarity - Competitor Analysis",
        "Cosine Similarity - Every Embedding",
        "Cosine Similarity - Content Heatmap",
        "Top/Bottom Embeddings", # Renamed for brevity
        "Entity Topic Gap Analysis",
        "Entity Visualizer",
        "Entity Frequency Charts",
        "Semantic Gap Analyzer",
        "Keyword Clustering from Semantic Gap", # Renamed for clarity
        "People Also Asked Analyzer", # Renamed
        "Google Ads Search Term Analyzer",
        "Google Search Console Analyzer",
        "Site Focus Visualizer (Semantic Clustering)", # Renamed
        "Entity Relationship Graph",
        "SEMRush - Top Sub-Directory Aggregation", # Renamed
        "SEMRush - Hierarchical Aggregation (No Leaf Nodes)" # Renamed

    ])
    if tool == "URL Analysis Dashboard":
        url_analysis_dashboard_page()
    elif tool == "Cosine Similarity - Competitor Analysis":
        cosine_similarity_competitor_analysis_page()
    elif tool == "Cosine Similarity - Every Embedding":
        cosine_similarity_every_embedding_page()
    elif tool == "Cosine Similarity - Content Heatmap":
        cosine_similarity_content_heatmap_page()
    elif tool == "Top/Bottom Embeddings":
        top_bottom_embeddings_page()
    elif tool == "Entity Topic Gap Analysis":
        entity_analysis_page()
    elif tool == "Entity Visualizer":
        displacy_visualization_page()
    elif tool == "Entity Frequency Charts":
        named_entity_barchart_page()
    elif tool == "Semantic Gap Analyzer":
        ngram_tfidf_analysis_page()
    elif tool == "Keyword Clustering from Semantic Gap":
        keyword_clustering_from_gap_page()
    elif tool == "People Also Asked Analyzer":
        paa_extraction_clustering_page()
    elif tool == "Google Ads Search Term Analyzer":
        google_ads_search_term_analyzer_page()
    elif tool == "Google Search Console Analyzer":
        google_search_console_analysis_page() # This is the updated function
    elif tool == "Site Focus Visualizer (Semantic Clustering)":
        semantic_clustering_page()
    elif tool == "Entity Relationship Graph":
        entity_relationship_graph_page()
    elif tool == "SEMRush - Top Sub-Directory Aggregation":
        semrush_organic_pages_by_subdirectory_page()
    elif tool == "SEMRush - Hierarchical Aggregation (No Leaf Nodes)":
        semrush_hierarchical_subdirectories_minimal_no_leaf_with_intent_filter()

    # Footer link
    st.markdown("---")
    st.markdown("<div style='text-align: center; margin-top: 20px;'>Powered by <a href='https://theseoconsultant.ai' target='_blank'>The SEO Consultant.ai</a></div>", unsafe_allow_html=True)
    # Add an empty div to push footer down slightly if needed
    # st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
    pass # Keep pass for structure

# <<< REPLACE THE OLD BLOCK BELOW WITH THE NEW ONE >>>
# The entry point - THIS IS THE PART TO REPLACE
if __name__ == "__main__":
    # Download necessary NLTK data (This is okay here as it uses print/nltk, not st)
    try:
        # Check if already downloaded
        nltk.data.find('tokenizers/punkt')
    except LookupError: # Correct exception type
        print("Downloading NLTK 'punkt' resource...")
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError: # Correct exception type
        print("Downloading NLTK 'stopwords' resource...")
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError: # Correct exception type
        print("Downloading NLTK 'wordnet' resource...")
        nltk.download('wordnet')

    # --- REMOVED MODEL PRE-LOADING CALLS ---
    # The @st.cache_resource decorator handles loading when needed inside main()
    # print("Pre-loading models...")
    # load_spacy_model()                 # REMOVED
    # initialize_sentence_transformer()  # REMOVED
    # load_bert_ner_pipeline()           # REMOVED
    # print("Model pre-loading complete.") # REMOVED
    print("NLTK checks complete. Starting Streamlit app...")

    # Run the main Streamlit app function
    # st.set_page_config() will be the first st command inside main()
    main()

# <<< NO MORE CODE AFTER THIS BLOCK >>>
