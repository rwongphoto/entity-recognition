import streamlit as st
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
from sklearn.cluster import KMeans, AgglomerativeClustering

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, WebDriverException, NoSuchElementException

# Import SentenceTransformer from sentence_transformers
from sentence_transformers import SentenceTransformer

# NEW IMPORTS for existing tools
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

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

# NEW IMPORT for Gemini Tool
import google.generativeai as genai

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
            nlp = spacy.load("en_core_web_md")
            print("spaCy model loaded successfully")
        except OSError:
            print("Downloading en_core_web_md model...")
            try:
                spacy.cli.download("en_core_web_md")
                nlp = spacy.load("en_core_web_md")
                print("en_core_web_md downloaded and loaded")
            except Exception as download_e:
                 st.error(f"Failed to download spaCy model: {download_e}")
                 st.error("Please try installing it manually: python -m spacy download en_core_web_md")
                 return None # Return None if download fails
        except Exception as e:
            st.error(f"Failed to load spaCy model: {e}")
            return None
    return nlp

@st.cache_resource # Cache SentenceTransformer model
def initialize_sentence_transformer():
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        st.error(f"Error loading SentenceTransformer model: {e}")
        st.error("Please ensure the 'sentence-transformers' library is installed correctly.")
        return None

def get_embedding(text, model):
    if model is None:
        st.error("SentenceTransformer model not loaded. Cannot generate embeddings.")
        return None # Or handle appropriately
    return model.encode(text)

@st.cache_data(ttl=86400) # Cache scraping results for a day
def extract_text_from_url(url):
    # Check if URL is valid format before proceeding
    if not url or not url.startswith(('http://', 'https://')):
        st.warning(f"Invalid or missing URL scheme: {url}. Skipping.")
        return None
    try:
        enforce_rate_limit()
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        # Use the random user agent function
        user_agent = get_random_user_agent()
        chrome_options.add_argument(f"user-agent={user_agent}")

        # Handle potential driver setup issues
        driver = None # Initialize driver to None
        try:
            driver = webdriver.Chrome(options=chrome_options)
        except WebDriverException as driver_err:
            st.error(f"Failed to start WebDriver: {driver_err}")
            st.error("Ensure ChromeDriver is installed and accessible in your system's PATH, or specify its path.")
            return None

        driver.get(url)
        # Increased wait time slightly
        wait = WebDriverWait(driver, 25)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "body")))
        page_source = driver.page_source
        driver.quit()

        soup = BeautifulSoup(page_source, "html.parser")
        body = soup.find('body')
        if not body:
            st.warning(f"No <body> tag found in {url}. Cannot extract main content.")
            return None # Return None if no body tag

        # Decompose header and footer before getting text
        for tag in body.find_all(['header', 'footer', 'nav', 'script', 'style']): # Also remove nav, script, style
            tag.decompose()

        # Get text, join lines, strip extra whitespace
        text = body.get_text(separator='\n', strip=True)
        # Clean up multiple newlines
        text = re.sub(r'\n\s*\n', '\n', text)
        return text

    except TimeoutException:
        st.warning(f"Timeout error fetching {url}. The page took too long to load or find the 'body' element.")
        if driver: driver.quit() # Ensure driver quits on timeout
        return None
    except WebDriverException as e:
        error_str = str(e)
        if "net::ERR_NAME_NOT_RESOLVED" in error_str or "dns error" in error_str.lower():
             st.warning(f"Could not resolve hostname for {url}. Check the URL or your DNS settings.")
        elif "unable to connect" in error_str.lower():
             st.warning(f"Unable to connect to {url}. The server might be down or blocking requests.")
        else:
            st.error(f"Selenium WebDriver error fetching {url}: {error_str}")
        if driver: driver.quit() # Ensure driver quits on WebDriverException
        return None
    except Exception as e:
        st.error(f"Unexpected error fetching {url}: {str(e)}")
        if driver: driver.quit() # Ensure driver quits on other exceptions
        return None


@st.cache_data(ttl=86400)
def extract_relevant_text_from_url(url):
    # Check if URL is valid format before proceeding
    if not url or not url.startswith(('http://', 'https://')):
        st.warning(f"Invalid or missing URL scheme: {url}. Skipping.")
        return None
    try:
        enforce_rate_limit()
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        user_agent = get_random_user_agent()
        chrome_options.add_argument(f"user-agent={user_agent}")

        driver = None
        try:
            driver = webdriver.Chrome(options=chrome_options)
        except WebDriverException as driver_err:
            st.error(f"Failed to start WebDriver: {driver_err}")
            st.error("Ensure ChromeDriver is installed and accessible in your system's PATH, or specify its path.")
            return None

        driver.get(url)
        wait = WebDriverWait(driver, 25) # Increased wait time
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "body")))
        page_source = driver.page_source
        driver.quit()

        soup = BeautifulSoup(page_source, "html.parser")

        # Try finding a main content area first (common patterns)
        main_content = soup.find('main') or \
                       soup.find('article') or \
                       soup.find(id='main-content') or \
                       soup.find(class_='main-content') or \
                       soup.find(id='content') or \
                       soup.find(class_='content')

        # If no main content area found, fall back to body excluding header/footer/nav
        if not main_content:
            main_content = soup.find('body')
            if main_content:
                for tag in main_content.find_all(['header', 'footer', 'nav', 'aside', 'script', 'style']):
                    tag.decompose()
            else:
                st.warning(f"No <body> tag found in {url}. Cannot extract relevant content.")
                return None # Return None if no body tag

        # Remove elements we generally don't want (can be customized)
        for undesirable in main_content.find_all(['script', 'style', 'noscript', 'button', 'form', 'iframe']):
            undesirable.decompose()

        # Extract text from relevant tags within the main content area
        tags_to_extract = ['p', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'table', 'blockquote', 'pre', 'code', 'dt', 'dd']
        texts = []
        for tag_name in tags_to_extract:
             for tag in main_content.find_all(tag_name):
                 # Get text, strip leading/trailing whitespace, ignore empty strings
                 tag_text = tag.get_text(separator=" ", strip=True)
                 if tag_text:
                     texts.append(tag_text)

        # Join texts and clean up multiple spaces/newlines
        relevant_text = " ".join(texts)
        relevant_text = re.sub(r'\s+', ' ', relevant_text).strip() # Replace multiple whitespace with single space
        return relevant_text

    except TimeoutException:
        st.warning(f"Timeout error fetching {url} for relevant text. The page took too long to load.")
        if driver: driver.quit()
        return None
    except WebDriverException as e:
        error_str = str(e)
        if "net::ERR_NAME_NOT_RESOLVED" in error_str or "dns error" in error_str.lower():
             st.warning(f"Could not resolve hostname for {url}. Check the URL or your DNS settings.")
        elif "unable to connect" in error_str.lower():
             st.warning(f"Unable to connect to {url}. The server might be down or blocking requests.")
        else:
            st.error(f"Selenium WebDriver error fetching relevant text from {url}: {error_str}")
        if driver: driver.quit()
        return None
    except Exception as e:
        st.error(f"Error extracting relevant content from {url}: {str(e)}")
        if driver: driver.quit()
        return None


@st.cache_data
def count_videos(_soup):
    if _soup is None: return 0
    video_count = len(_soup.find_all("video"))
    iframe_videos = 0
    for iframe in _soup.find_all("iframe"):
        src = iframe.get("src")
        if src and any(domain in src for domain in ["youtube.com", "youtube-nocookie.com", "vimeo.com"]):
            iframe_videos += 1
    return video_count + iframe_videos

def preprocess_text(text, nlp_model):
    if not text or not nlp_model:
        return ""
    doc = nlp_model(text.lower()) # Convert to lowercase
    # Keep lemmas, remove punctuation, spaces, stop words, and short tokens
    lemmatized_tokens = [
        token.lemma_ for token in doc
        if not token.is_punct and not token.is_space and not token.is_stop and len(token.lemma_) > 2
    ]
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
            <img src="{logo_url}" width="350" alt="The SEO Consultant AI Logo">
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
          flex-wrap: wrap; /* Allow wrapping on smaller screens */
          margin-bottom: 35px;
          border-radius: 5px; /* Added subtle rounding */
        }
        .topnav a {
          float: left; /* Keep for alignment */
          display: block;
          color: black;
          text-align: center;
          padding: 14px 16px;
          text-decoration: none;
          transition: background-color 0.3s, color 0.3s; /* Smooth transition */
        }
        .topnav a:hover {
          background-color: #ddd;
          color: black;
        }
        .topnav a:active { /* Style for when link is clicked */
             background-color: #ccc;
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
    try:
        # The aggregation_strategy="simple" groups tokens into complete entities.
        return pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    except Exception as e:
        st.error(f"Error loading BERT NER pipeline: {e}")
        st.error("Ensure 'transformers' and 'torch' (or 'tensorflow') are installed.")
        return None

# --------------------------------------------------
# UPDATED: Use BERT for Named Entity Recognition
# --------------------------------------------------
def identify_entities(text, nlp_model):
    """
    Extracts named entities from the input text using a BERT-based NER pipeline.
    'nlp_model' is used for lemmatization inside count_entities, not here.
    Returns a list of tuples (entity_text, entity_label).
    """
    if not text:
        return []
    ner_pipe = load_bert_ner_pipeline()
    if not ner_pipe:
        return [] # Return empty if pipeline failed to load

    entities = []
    # Handle potential errors during NER processing
    try:
        bert_entities = ner_pipe(text)
        for ent in bert_entities:
            # ent is a dict with keys like 'word' and 'entity_group'
            entity_text = ent["word"].strip()
            # Basic cleaning: remove leading/trailing punctuation potentially included by BERT
            entity_text = entity_text.strip('.,;:!?"\'()[]{}')
            entity_label = ent["entity_group"] # e.g., 'PER', 'ORG', 'LOC'
            if entity_text: # Ensure entity is not empty after stripping
                entities.append((entity_text, entity_label))
    except Exception as e:
        st.warning(f"Error during BERT NER processing: {e}. Results might be incomplete.")
        # Decide how to handle: return partial results, empty list, or raise error
        # Returning empty list for now to avoid breaking downstream processes
        return []
    return entities

# ORIGINAL count_entities (for unique counts per source)
def count_entities(entities: List[Tuple[str, str]], nlp_model) -> Counter:
    """Counts unique lemmatized entities per source."""
    if not nlp_model:
        st.error("spaCy model needed for lemmatization in count_entities not loaded.")
        return Counter()

    entity_counts = Counter()
    seen_entities = set() # Tracks (lemma, label) combinations already counted for this source

    for entity_text, label in entities:
        # Basic cleaning (replace newlines, check length)
        entity_text = entity_text.replace('\n', ' ').replace('\r', '').strip()
        if len(entity_text) > 1: # Allow slightly shorter entities like 'AI'
             # Lemmatize using spaCy model passed in
            doc = nlp_model(entity_text)
            # Simple lemma: join lemmas of non-stopword tokens, lowercase
            lemma = " ".join([token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and token.lemma_])
            # Fallback if lemmatization results in empty string (e.g., only stopwords/punctuation)
            if not lemma:
                lemma = entity_text.lower()

            if (lemma, label) not in seen_entities:
                entity_counts[(lemma, label)] += 1
                seen_entities.add((lemma, label))
    return entity_counts

# New function to count every occurrence
def count_entities_total(entities: List[Tuple[str, str]], nlp_model) -> Counter:
    """Counts total occurrences of lemmatized entities."""
    if not nlp_model:
        st.error("spaCy model needed for lemmatization in count_entities_total not loaded.")
        return Counter()

    entity_counts = Counter()
    for entity_text, label in entities:
        entity_text = entity_text.replace('\n', ' ').replace('\r', '').strip()
        if len(entity_text) > 1:
            doc = nlp_model(entity_text)
            lemma = " ".join([token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and token.lemma_])
            if not lemma:
                lemma = entity_text.lower()
            # Count every occurrence
            entity_counts[(lemma, label)] += 1
    return entity_counts

def display_entity_barchart(entity_counts, top_n=30):
    # Filter out less common entity types if desired (optional)
    excluded_labels = {"CARDINAL", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL"}
    filtered_entity_counts = {
        (entity, label): count
        for (entity, label), count in entity_counts.items()
        if label not in excluded_labels and count > 0 # Ensure count is positive
    }

    if not filtered_entity_counts:
        st.info("No entities found matching the criteria for the bar chart.")
        return

    # Create DataFrame and sort
    entity_data = pd.DataFrame.from_dict(filtered_entity_counts, orient='index', columns=['count'])
    entity_data.index = pd.MultiIndex.from_tuples(entity_data.index, names=['entity', 'label']) # Use MultiIndex
    entity_data = entity_data.sort_values('count', ascending=False).head(top_n).reset_index()

    # Prepare for plotting
    # Combine entity and label for unique bar labels if entities can have multiple labels
    entity_labels = [f"{row['entity']} ({row['label']})" for index, row in entity_data.iterrows()]
    counts = entity_data['count']

    # Create Plot
    fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.3))) # Dynamic height
    bars = ax.barh(entity_labels, counts) # Horizontal bar chart is often better for many labels
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Entities (Label)")
    ax.set_title(f"Top {top_n} Most Frequent Entities")
    ax.invert_yaxis() # Display top entity at the top

    # Add count labels to bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + counts.max() * 0.01, # Position label slightly right of bar end
                bar.get_y() + bar.get_height()/2., # Center label vertically
                f'{int(width)}',
                ha='left', va='center')

    plt.tight_layout()
    st.pyplot(fig)


def display_entity_wordcloud(entity_counts):
    # Aggregate counts by the entity text only for word cloud generation
    aggregated = {}
    for key, count in entity_counts.items():
        entity_text = key[0] if isinstance(key, tuple) else key # Get entity text
        aggregated[entity_text] = aggregated.get(entity_text, 0) + count

    if not aggregated:
        st.info("No entities found to generate a word cloud.")
        return

    try:
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis', max_words=150).generate_from_frequencies(aggregated)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout()
        st.pyplot(fig)
    except ValueError as ve:
         # Handle case where generate_from_frequencies receives empty data or invalid values
         st.warning(f"Could not generate word cloud: {ve}")
    except Exception as e:
        st.error(f"An unexpected error occurred during word cloud generation: {e}")


# Ensure NLTK stopwords are downloaded (moved to main execution block for better practice)
# nltk.download('stopwords')
stop_words_nltk = set(stopwords.words('english')) # Use a distinct variable name

def generate_topic_label(queries_in_topic, max_label_words=3):
    """Generates a concise label for a topic based on frequent non-stopword terms."""
    words = []
    lemmatizer = WordNetLemmatizer()
    for query in queries_in_topic:
        tokens = word_tokenize(str(query).lower()) # Ensure query is string, tokenize
        filtered = [lemmatizer.lemmatize(t) for t in tokens if t.isalnum() and t not in stop_words_nltk and len(t) > 2]
        words.extend(filtered)

    if not words:
        return "Misc./Undefined Topic" # More descriptive default

    # Count word frequencies
    freq = collections.Counter(words)
    # Get the most common words, up to max_label_words
    common_words = freq.most_common(max_label_words)

    if not common_words:
         return "Misc./Undefined Topic"

    # Create the label by joining the most common words
    label = " | ".join([word for word, count in common_words])
    return label.capitalize() # Capitalize the first word


# ------------------------------------
# Cosine Similarity Functions
# ------------------------------------
def calculate_overall_similarity(urls: List[str], search_term: str, model) -> List[Tuple[str, float | None]]:
    """Calculates similarity between scraped URL text and a search term."""
    if not search_term:
        st.warning("Search term is empty. Returning no similarity scores.")
        return [(url, None) for url in urls]
    if not model:
        st.error("SentenceTransformer model not loaded. Cannot calculate similarity.")
        return [(url, None) for url in urls]

    search_term_embedding = get_embedding(search_term, model)
    if search_term_embedding is None:
        return [(url, None) for url in urls] # Handle embedding failure

    results = []
    total_urls = len(urls)
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, url in enumerate(urls):
        status_text.text(f"Processing URL {i+1}/{total_urls}: {url}")
        text = extract_text_from_url(url) # Use the main extraction function
        similarity = None # Default to None
        if text:
            text_embedding = get_embedding(text, model)
            if text_embedding is not None:
                try:
                    # Reshape for cosine_similarity (expects 2D arrays)
                    similarity_score = cosine_similarity(text_embedding.reshape(1, -1), search_term_embedding.reshape(1, -1))
                    similarity = float(similarity_score[0][0]) # Extract float value
                except ValueError as e:
                    st.warning(f"Could not calculate similarity for {url}: {e}")
                except Exception as e:
                    st.warning(f"Unexpected error calculating similarity for {url}: {e}")
        else:
             st.warning(f"No text extracted from {url}, cannot calculate similarity.")

        results.append((url, similarity))
        progress_bar.progress((i + 1) / total_urls)

    status_text.text("Similarity calculation complete.")
    return results


def calculate_similarity(text: str, search_term: str, model) -> Tuple[List[str], List[float]]:
    """Calculates similarity for each sentence in a text block."""
    if not text or not search_term:
        return [], []
    if not model:
        st.error("SentenceTransformer model not loaded.")
        return [], []

    # Improved sentence splitting (handles more edge cases like abbreviations)
    # Consider using nltk.sent_tokenize for more robustness if needed
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+', text) # Added !
    sentences = [s.strip() for s in sentences if s.strip() and len(s.split()) > 3] # Filter short/empty sentences

    if not sentences:
        return [], []

    try:
        search_term_embedding = get_embedding(search_term, model)
        if search_term_embedding is None: return [], []

        # Batch sentence embeddings for efficiency
        sentence_embeddings = get_embedding(sentences, model) # Model handles list input
        if sentence_embeddings is None or len(sentence_embeddings) != len(sentences):
             st.warning("Failed to generate embeddings for some sentences.")
             return sentences, [0.0] * len(sentences) # Return sentences with 0 similarity

        # Calculate cosine similarities
        similarities = cosine_similarity(sentence_embeddings, search_term_embedding.reshape(1, -1))
        # Flatten the result and convert to list of floats
        similarities_list = [float(s[0]) for s in similarities]
        return sentences, similarities_list

    except Exception as e:
        st.error(f"Error during sentence similarity calculation: {e}")
        return sentences, [0.0] * len(sentences) # Return sentences with 0 similarity on error


def rank_sentences_by_similarity(text: str, search_term: str) -> List[Tuple[str, float]]:
    """Ranks sentences by normalized cosine similarity."""
    model = initialize_sentence_transformer()
    if not model: return []

    sentences, similarities = calculate_similarity(text, search_term, model)
    if not sentences: return []

    # Normalize similarities to 0-1 range
    min_similarity = min(similarities) if similarities else 0.0
    max_similarity = max(similarities) if similarities else 0.0
    range_similarity = max_similarity - min_similarity

    if range_similarity == 0: # Avoid division by zero if all similarities are the same
        normalized_similarities = [0.5] * len(similarities) # Assign a mid-value or 0
    else:
        normalized_similarities = [(s - min_similarity) / range_similarity for s in similarities]

    return list(zip(sentences, normalized_similarities))

def highlight_text(text: str, search_term: str) -> str:
    """Highlights sentences based on normalized similarity."""
    sentences_with_similarity = rank_sentences_by_similarity(text, search_term)
    if not sentences_with_similarity:
        return "<p>Could not process text for highlighting.</p>"

    highlighted_html = ""
    # Define color thresholds (adjust as needed)
    low_threshold = 0.35
    high_threshold = 0.65

    for sentence, normalized_similarity in sentences_with_similarity:
        if normalized_similarity < low_threshold:
            color = "#FF7F7F" # Lighter red
        elif normalized_similarity >= high_threshold:
            color = "#90EE90" # Lighter green
        else:
            color = "#E0E0E0" # Light gray for mid-range

        # Use background color for highlighting instead of text color for better readability
        highlighted_html += f'<span style="background-color:{color}; padding: 2px 0; margin-bottom: 2px; display: inline-block;">{sentence}</span><br/>\n' # Wrap in span, add line break

    # Wrap the whole thing in a div for better control if needed
    return f'<div style="line-height: 1.6;">{highlighted_html}</div>'


def rank_sections_by_similarity_bert(text: str, search_term: str, top_n: int = 10) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """Ranks sentences by raw similarity score."""
    model = initialize_sentence_transformer()
    if not model: return [], []

    sentences, similarities = calculate_similarity(text, search_term, model)
    if not sentences: return [], []

    section_scores = list(zip(sentences, similarities))

    # Sort by similarity score (raw score, not normalized)
    sorted_sections = sorted(section_scores, key=lambda item: item[1], reverse=True)

    top_sections = sorted_sections[:top_n]
    # Get bottom N by taking the last N elements of the sorted list
    bottom_sections = sorted_sections[-top_n:]

    return top_sections, bottom_sections


# ------------------------------------
# NEW: Helper function using SPARQLWrapper to get Wikidata link for an entity
# ------------------------------------
@st.cache_data(ttl=86400) # Cache Wikidata results
def get_wikidata_link(entity_name: str) -> str | None:
    """Queries Wikidata for a link to the given entity name."""
    if not entity_name:
        return None

    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.agent = f"StreamlitSEOTools/1.0 ({get_random_user_agent()})" # Be polite, identify agent

    # Escape quotes in the entity name for the SPARQL query
    safe_entity = entity_name.replace('"', '\\"')

    # Query for entities matching the label exactly (case-insensitive might be too broad)
    # Limit results to reduce ambiguity
    query = f"""
    SELECT ?item ?itemLabel WHERE {{
      ?item rdfs:label "{safe_entity}"@en. # Match English label exactly
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }} # Also get label back
    }} LIMIT 5
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    try:
        results = sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])
        if bindings:
            # Potentially return the first result, or let user choose if multiple?
            # For simplicity, return the first one.
            item_url = bindings[0].get("item", {}).get("value")
            return item_url
        else:
            # Optional: Add a fallback query for aliases if exact match fails
            # query_alias = ...
            # sparql.setQuery(query_alias) ... etc.
            return None # No match found
    except Exception as e:
        # Log error but don't necessarily stop the app
        print(f"Warning: Error querying Wikidata for '{entity_name}': {e}")
        # st.warning(f"Could not query Wikidata for '{entity_name}'.") # Optional user warning
        return None # Return None on error


# ------------------------------------
# Streamlit UI Functions
# ------------------------------------

def url_analysis_dashboard_page():
    st.header("URL Analysis Dashboard")
    st.markdown("Analyze multiple URLs and gather key SEO metrics.")

    urls_input = st.text_area("Enter URLs (one per line):", height=150, key="dashboard_urls", placeholder="https://example.com/page1\nhttps://example.com/page2")
    urls = [url.strip() for url in urls_input.splitlines() if url.strip()]

    search_term = st.text_input("Enter Search Term (Optional, for Cosine Similarity):", key="dashboard_search_term", value="")

    if st.button("Analyze URLs", key="dashboard_button"):
        if not urls:
            st.warning("Please enter at least one URL.")
            st.session_state.dashboard_results_df = None # Clear previous results if button clicked with no URLs
            return

        # --- Initialize or clear session state for results ---
        st.session_state.dashboard_results_df = None
        data = [] # Store results for DataFrame

        with st.spinner("Analyzing URLs... This may take some time depending on the number of URLs."):
            # Load models (cached)
            nlp_model = load_spacy_model()
            sentence_model = initialize_sentence_transformer()

            if not nlp_model or not sentence_model:
                st.error("Required NLP models failed to load. Cannot proceed.")
                return # Stop if models fail

            # Calculate overall similarities first (if search term provided)
            similarity_results_map = {}
            if search_term:
                 similarity_results = calculate_overall_similarity(urls, search_term, sentence_model)
                 similarity_results_map = {url: score for url, score in similarity_results}


            # --- Analysis Loop ---
            total_urls = len(urls)
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, url in enumerate(urls):
                status_text.text(f"Processing URL {i+1}/{total_urls}: {url}")
                row_data = {"URL": url} # Initialize dict for this row's data

                try:
                    # --- Selenium/BS4 Fetching ---
                    chrome_options = Options()
                    chrome_options.add_argument("--headless")
                    chrome_options.add_argument("--no-sandbox")
                    chrome_options.add_argument("--disable-dev-shm-usage")
                    user_agent = get_random_user_agent()
                    chrome_options.add_argument(f"user-agent={user_agent}")

                    driver = None
                    page_source = None
                    meta_title = "N/A" # Default value

                    try:
                        enforce_rate_limit()
                        driver = webdriver.Chrome(options=chrome_options)
                        driver.get(url)
                        wait = WebDriverWait(driver, 20) # Wait for body
                        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                        page_source = driver.page_source
                        meta_title = driver.title if driver.title else "N/A"
                        driver.quit()
                    except TimeoutException:
                         st.warning(f"Timeout loading {url}. Data might be incomplete.")
                         if driver: driver.quit()
                         page_source = None # Ensure page_source is None on timeout
                         # Continue processing but mark metrics as N/A or Error
                    except WebDriverException as wd_err:
                         st.warning(f"WebDriver error for {url}: {wd_err}. Data might be incomplete.")
                         if driver: driver.quit()
                         page_source = None
                         # Continue processing
                    except Exception as fetch_err:
                         st.error(f"Unexpected error fetching {url}: {fetch_err}")
                         if driver: driver.quit()
                         page_source = None
                         # Skip to next URL or mark as error
                         data.append([url] + ["Fetch Error"] * 13) # Use explicit error marker
                         progress_bar.progress((i + 1) / total_urls)
                         continue # Skip rest of processing for this URL

                    # --- BS4 Parsing & Content Extraction ---
                    if page_source:
                        soup = BeautifulSoup(page_source, "html.parser")
                        body = soup.find('body')

                        # Initialize metrics with defaults
                        h1_tag = "N/A"
                        total_text = ""
                        total_word_count = 0
                        custom_word_count = 0
                        header_links = 0
                        footer_links = 0
                        total_links = 0
                        schema_types = set()
                        lists_tables_str = "OL: No | UL: No | Table: No"
                        num_images = 0
                        num_videos = 0
                        entities = []
                        unique_entity_count = 0
                        flesch_kincaid = None

                        if body:
                            # Find H1
                            h1 = body.find("h1")
                            h1_tag = h1.get_text(strip=True) if h1 else "N/A"

                            # Extract total text (excluding header/footer for word count)
                            body_copy = BeautifulSoup(str(body), "html.parser") # Work on a copy
                            for tag in body_copy.find_all(['header', 'footer', 'nav', 'script', 'style']):
                                tag.decompose()
                            total_text = body_copy.get_text(separator="\n", strip=True)
                            total_text = re.sub(r'\n\s*\n', '\n', total_text) # Clean newlines
                            total_word_count = len(total_text.split()) if total_text else 0

                            # Custom word count (p, li, h1-h6, table cells)
                            custom_words = []
                            content_elements = body.find_all(["p", "li", "h1", "h2", "h3", "h4", "h5", "h6"])
                            for el in content_elements:
                                custom_words.extend(el.get_text(strip=True).split())
                            for table in body.find_all("table"):
                                for cell in table.find_all(["td", "th"]):
                                     custom_words.extend(cell.get_text(strip=True).split())
                            custom_word_count = len(custom_words)

                            # Link Counts
                            header = soup.find("header")
                            footer = soup.find("footer")
                            header_links = len(header.find_all("a", href=True)) if header else 0
                            footer_links = len(footer.find_all("a", href=True)) if footer else 0
                            total_links = len(soup.find_all("a", href=True)) # All links on page

                            # Schema Markup
                            ld_json_scripts = soup.find_all("script", type="application/ld+json")
                            for script in ld_json_scripts:
                                try:
                                    script_content = script.string
                                    if script_content:
                                        data_json = json.loads(script_content)
                                        items_to_check = []
                                        if isinstance(data_json, list):
                                            items_to_check.extend(data_json)
                                        elif isinstance(data_json, dict):
                                            items_to_check.append(data_json)

                                        for item in items_to_check:
                                            if isinstance(item, dict):
                                                schema_type = item.get("@type")
                                                if isinstance(schema_type, str):
                                                    schema_types.add(schema_type)
                                                elif isinstance(schema_type, list): # Handle type as list
                                                    for t in schema_type:
                                                        if isinstance(t, str):
                                                            schema_types.add(t)
                                except (json.JSONDecodeError, TypeError):
                                    # Ignore script tags that are not valid JSON or have unexpected structure
                                    continue

                            # Lists/Tables Presence
                            ol_present = 'Yes' if body.find('ol') else 'No'
                            ul_present = 'Yes' if body.find('ul') else 'No'
                            table_present = 'Yes' if body.find('table') else 'No'
                            lists_tables_str = f"OL: {ol_present} | UL: {ul_present} | Table: {table_present}"

                            # Images and Videos
                            num_images = len(soup.find_all("img"))
                            num_videos = count_videos(soup) # Use helper function

                            # --- NLP Metrics ---
                            if total_text:
                                # Entities (using BERT via identify_entities)
                                entities = identify_entities(total_text, nlp_model)
                                unique_entity_count = len(set(ent[0].lower() for ent in entities)) # Count unique lowercased entity text

                                # Readability
                                try:
                                    flesch_kincaid = textstat.flesch_kincaid_grade(total_text)
                                except Exception: # Catch potential errors in textstat
                                    flesch_kincaid = None # Assign None if calculation fails
                        else:
                             st.warning(f"No <body> tag content found for {url}.")
                             # Append row with N/A or Error markers
                             data.append([url] + ["Parse Error"] * 13)
                             progress_bar.progress((i + 1) / total_urls)
                             continue

                        # Get Similarity Score (already calculated if search term provided)
                        similarity_val = similarity_results_map.get(url, np.nan if search_term else None) # Use map, handle no search term case

                        # Append results for this URL
                        data.append([
                            url,
                            meta_title,
                            h1_tag,
                            total_word_count,
                            custom_word_count,
                            similarity_val if search_term else "N/A", # Show N/A if no search term
                            unique_entity_count,
                            header_links + footer_links, # Combine nav links
                            total_links,
                            ", ".join(sorted(list(schema_types))) if schema_types else "None",
                            lists_tables_str,
                            num_images,
                            num_videos,
                            flesch_kincaid
                        ])

                    else: # If page_source is None (fetch failed)
                         # Already handled by the continue statement in fetch error block
                         # This else block might not be strictly needed anymore
                         data.append([url] + ["Fetch Failed"] * 13)


                except Exception as e:
                    st.error(f"Unexpected critical error processing URL {url}: {e}")
                    # Ensure driver is quit if it exists and an error occurred mid-processing
                    if 'driver' in locals() and driver:
                        try:
                            driver.quit()
                        except Exception: pass # Ignore errors during cleanup quit
                    data.append([url] + ["Processing Error"] * 13) # Append error placeholders

                finally:
                    # Ensure driver is always quit if it was successfully created
                    if 'driver' in locals() and driver:
                        try:
                            driver.quit()
                        except Exception: pass
                    # Update progress bar for this URL
                    progress_bar.progress((i + 1) / total_urls)
            # --- End Analysis Loop ---

            status_text.text("Analysis complete. Preparing results...")

            # --- DataFrame Creation and Display ---
            if data: # Only create DataFrame if data was collected
                df = pd.DataFrame(data, columns=[
                    "URL",
                    "Meta Title",
                    "H1 Tag",
                    "Total Word Count",
                    "Content Word Count",
                    "Overall Cosine Similarity Score", # Column name kept generic
                    "# of Unique Entities",
                    "Nav Links", # Combined Header/Footer
                    "Total # of Links",
                    "Schema Markup Types",
                    "Lists/Tables Present",
                    "# of Images",
                    "# of Videos",
                    "Flesch-Kincaid Grade Level"
                ])

                # Reorder for better presentation and rename columns
                df = df[[
                    "URL",
                    "Meta Title",
                    "H1 Tag", # Renamed below
                    "Total Word Count",
                    "Content Word Count", # Renamed below
                    "Overall Cosine Similarity Score", # Renamed below
                    "Flesch-Kincaid Grade Level", # Renamed below
                    "# of Unique Entities",
                    "Nav Links", # Keep name
                    "Total # of Links", # Renamed below
                    "Schema Markup Types", # Renamed below
                    "Lists/Tables Present", # Renamed below
                    "# of Images", # Renamed below
                    "# of Videos", # Renamed below
                ]]

                # Rename columns for final display
                df.columns = [
                    "URL",
                    "Meta Title",
                    "H1",
                    "Total Word Count",
                    "Content Word Count",
                    "Cosine Similarity" if search_term else "Similarity (N/A)", # Adjust name based on search term
                    "Grade Level",
                    "# Unique Entities", # Shortened
                    "Nav Links",
                    "Total Links",
                    "Schema Types",
                    "Lists/Tables",
                    "Images",
                    "Videos",
                ]

                # Convert columns to numeric where applicable, coercing errors
                numeric_cols = ["Total Word Count", "Content Word Count", "Grade Level",
                                "# Unique Entities", "Nav Links", "Total Links",
                                "Images", "Videos"]
                if search_term: # Only convert similarity if calculated
                    numeric_cols.append("Cosine Similarity")

                for col in numeric_cols:
                     if col in df.columns: # Check if column exists
                         df[col] = pd.to_numeric(df[col], errors='coerce') # Coerce errors to NaN

                # Define formatting for display
                format_dict = {
                    "Total Word Count": "{:,.0f}",
                    "Content Word Count": "{:,.0f}",
                    "Grade Level": "{:.1f}",
                    "# Unique Entities": "{:,.0f}",
                    "Nav Links": "{:,.0f}",
                    "Total Links": "{:,.0f}",
                    "Images": "{:,.0f}",
                    "Videos": "{:,.0f}",
                }
                if search_term and "Cosine Similarity" in df.columns:
                    format_dict["Cosine Similarity"] = "{:.4f}" # Format similarity if present


                # Display the dataframe with formatting
                st.dataframe(df.style.format(format_dict, na_rep="N/A")) # Use na_rep for NaNs

                # --- Store DataFrame in session state ---
                st.session_state.dashboard_results_df = df.copy() # Use .copy() to be safe
                st.success("Dashboard analysis complete. You can now use the 'Gemini Analysis' tool in the sidebar.")
                # --- END Store ---
            else:
                st.warning("Analysis did not produce any results for the given URLs.")
                st.session_state.dashboard_results_df = None # Ensure state is None if no results


def cosine_similarity_competitor_analysis_page():
    st.title("Cosine Similarity Competitor Analysis")
    st.markdown("Compare content relevance against competitors using Cosine Similarity.")
    search_term = st.text_input("Enter Search Term:", "", key="cs_search_term")

    source_option = st.radio(
        "Select content source for competitors:",
        options=["Extract from URL", "Paste Content"],
        index=0,
        key="cs_source_option"
    )

    competitor_sources = []
    if source_option == "Extract from URL":
        urls_input = st.text_area("Enter Competitor URLs (one per line):", "", key="cs_urls")
        competitor_sources = [url.strip() for url in urls_input.splitlines() if url.strip()]
        source_type = "URL"
    else:
        st.markdown("Paste the competitor content below. If you have multiple competitors, separate each content block with `---` on a new line.")
        pasted_content = st.text_area("Enter Competitor Content:", height=200, key="cs_paste")
        competitor_sources = [content.strip() for content in pasted_content.split('---') if content.strip()]
        source_type = "Pasted Content"

    if st.button("Calculate Similarity", key="cs_button"):
        model = initialize_sentence_transformer()
        if not model:
            st.error("SentenceTransformer model failed to load.")
            return
        if not search_term:
            st.warning("Please enter a search term.")
            return
        if not competitor_sources:
            st.warning(f"Please provide at least one {source_type}.")
            return

        similarity_scores = []
        content_lengths = []
        source_labels = [] # To store URL or "Competitor X"

        with st.spinner(f"Analyzing competitor {source_type.lower()}..."):
            if source_type == "URL":
                # Use the optimized calculate_overall_similarity
                results = calculate_overall_similarity(competitor_sources, search_term, model)
                valid_results = [(url, score) for url, score in results if score is not None]
                similarity_scores = [score for url, score in valid_results]
                source_labels = [url for url, score in valid_results]

                # Get content lengths for the valid URLs
                temp_lengths = {}
                for url in source_labels:
                    text = extract_text_from_url(url) # Re-extract (or cache could help)
                    temp_lengths[url] = len(text.split()) if text else 0
                content_lengths = [temp_lengths[url] for url in source_labels] # Ensure order matches

            else: # Pasted Content
                search_embedding = get_embedding(search_term, model)
                if search_embedding is None: return # Handle error

                for idx, content in enumerate(competitor_sources):
                    label = f"Competitor {idx+1}"
                    source_labels.append(label)
                    content_lengths.append(len(content.split())) # Count words
                    text_embedding = get_embedding(content, model)
                    if text_embedding is not None:
                        similarity = cosine_similarity(text_embedding.reshape(1, -1), search_embedding.reshape(1, -1))[0][0]
                        similarity_scores.append(float(similarity))
                    else:
                        similarity_scores.append(0.0) # Assign 0 if embedding fails
                        st.warning(f"Could not generate embedding for {label}. Similarity set to 0.")


        if not source_labels:
             st.warning("No valid competitor data could be processed.")
             return

        # --- Create DataFrame ---
        df = pd.DataFrame({
            'Competitor': source_labels,
            'Cosine Similarity': similarity_scores,
            'Content Length (Words)': content_lengths
        })

        # --- Visualization Tabs ---
        tab1, tab2, tab3 = st.tabs(["📊 Scatter Plot", "📊 Bar Chart", "📊 Bubble Chart"])

        with tab1:
            st.subheader("Scatter Plot: Similarity vs. Content Length")
            if not df.empty:
                fig_scatter = px.scatter(
                    df, x='Cosine Similarity', y='Content Length (Words)',
                    title='Competitor Analysis: Similarity vs. Content Length',
                    hover_data=['Competitor', 'Cosine Similarity', 'Content Length (Words)'],
                    color='Cosine Similarity',
                    color_continuous_scale=px.colors.sequential.Viridis,
                    text='Competitor' # Add text labels to points
                )
                fig_scatter.update_traces(textposition='top center')
                fig_scatter.update_layout(
                    xaxis_title="Cosine Similarity (Higher = More Relevant)",
                    yaxis_title="Content Length (Words)",
                    width=800, height=600
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
                st.dataframe(df.style.format({"Cosine Similarity": "{:.4f}", "Content Length (Words)": "{:,.0f}"}))
            else:
                st.info("No data to display in scatter plot.")

        with tab2:
            st.subheader("Bar Chart: Similarity and Content Length")
            if not df.empty:
                df_sorted_bar = df.sort_values('Cosine Similarity', ascending=False)
                fig_bar = go.Figure()
                # Similarity Bars (Primary Y-axis)
                fig_bar.add_trace(go.Bar(
                    name='Cosine Similarity',
                    x=df_sorted_bar['Competitor'], y=df_sorted_bar['Cosine Similarity'],
                    marker_color=df_sorted_bar['Cosine Similarity'], # Color by similarity value
                    marker_colorscale='Viridis',
                    text=[f"{sim:.3f}" for sim in df_sorted_bar['Cosine Similarity']], # Format text labels
                    textposition='outside',
                    yaxis='y1' # Explicitly assign to y1
                ))
                # Content Length Line (Secondary Y-axis)
                fig_bar.add_trace(go.Scatter(
                    name='Content Length',
                    x=df_sorted_bar['Competitor'], y=df_sorted_bar['Content Length (Words)'],
                    yaxis='y2', # Assign to y2
                    mode='lines+markers', marker=dict(color='red')
                ))

                fig_bar.update_layout(
                    title='Competitor Analysis: Similarity and Content Length',
                    xaxis_title="Competitor",
                    yaxis=dict(title='Cosine Similarity (Higher = More Relevant)', side='left'), # y1 config
                    yaxis2=dict(title='Content Length (Words)', overlaying='y', side='right', showgrid=False), # y2 config
                    width=800, height=600,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    xaxis={'categoryorder':'array', 'categoryarray': df_sorted_bar['Competitor']} # Keep sorted order
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                st.dataframe(df_sorted_bar.style.format({"Cosine Similarity": "{:.4f}", "Content Length (Words)": "{:,.0f}"}))
            else:
                st.info("No data to display in bar chart.")

        with tab3:
            st.subheader("Bubble Chart: Similarity vs. Content Length")
            if not df.empty:
                # Use Content Length for size, normalize for better visualization
                size_data = df['Content Length (Words)'].fillna(0).clip(lower=1) # Ensure positive size
                size_norm = (size_data / size_data.max()) * 50 + 10 # Normalize and scale size

                fig_bubble = px.scatter(
                    df, x='Cosine Similarity', y='Content Length (Words)',
                    size=size_norm, # Use normalized size
                    title='Competitor Analysis: Similarity vs. Content Length (Bubble Chart)',
                    hover_data=['Competitor', 'Cosine Similarity', 'Content Length (Words)'],
                    color='Cosine Similarity',
                    color_continuous_scale=px.colors.sequential.Viridis,
                    text='Competitor' # Add text labels
                )
                fig_bubble.update_traces(textposition='top center')
                fig_bubble.update_layout(
                    xaxis_title="Cosine Similarity (Higher = More Relevant)",
                    yaxis_title="Content Length (Words)",
                    width=800, height=600
                )
                st.plotly_chart(fig_bubble, use_container_width=True)
                st.dataframe(df.style.format({"Cosine Similarity": "{:.4f}", "Content Length (Words)": "{:,.0f}"}))
            else:
                st.info("No data to display in bubble chart.")


def cosine_similarity_every_embedding_page():
    st.header("Cosine Similarity Score - Every Sentence")
    st.markdown("Calculates the cosine similarity score for each sentence in your input compared to a search term.")
    url = st.text_input("Enter URL (Optional):", key="every_embed_url", value="")
    use_url = st.checkbox("Use URL for Text Input", key="every_embed_use_url", value=bool(url)) # Default check if URL is entered
    text_input = st.text_area("Enter Text:", key="every_embed_text", value="", height=250, disabled=use_url)
    search_term = st.text_input("Enter Search Term:", key="every_embed_search", value="")

    # Determine text source
    text_to_analyze = ""
    if use_url:
        if url:
            with st.spinner(f"Extracting text from {url}..."):
                text_to_analyze = extract_text_from_url(url)
                if not text_to_analyze:
                    st.error(f"Could not extract text from {url}. Please check the URL or paste text directly.")
                    return # Stop if extraction fails
        else:
            st.warning("Please enter a URL to extract text, or uncheck the 'Use URL' box and paste text.")
            return
    else:
        text_to_analyze = text_input
        if not text_to_analyze:
             st.warning("Please enter text to analyze, or check the 'Use URL' box and enter a URL.")
             return

    if st.button("Calculate Sentence Similarities", key="every_embed_button"):
        if not search_term:
            st.warning("Please enter a search term.")
            return
        if not text_to_analyze:
             st.warning("No text available to analyze.") # Should be caught above, but good safeguard
             return

        model = initialize_sentence_transformer()
        if not model: return # Stop if model fails

        with st.spinner("Calculating Similarities..."):
            sentences, similarities = calculate_similarity(text_to_analyze, search_term, model)

        if sentences:
            st.subheader("Sentence Similarity Scores:")
            df_sentences = pd.DataFrame({
                "Sentence": sentences,
                "Similarity Score": similarities
            })
            # Sort by score descending
            df_sentences = df_sentences.sort_values("Similarity Score", ascending=False).reset_index(drop=True)
            st.dataframe(df_sentences.style.format({"Similarity Score": "{:.4f}"}))

            # Optionally, display sentences directly
            # for i, (sentence, score) in enumerate(zip(sentences, similarities), 1):
            #     st.write(f"{i}. (Score: {score:.4f}) {sentence}")
        else:
             st.info("No sentences were found or processed in the provided text.")


def cosine_similarity_content_heatmap_page():
    st.header("Content Relevance Heatmap")
    st.markdown("""
    Visualize sentence relevance using background color intensity.
    Sentences with higher cosine similarity to your search term will have a **greener** background,
    while less relevant sentences will have a **redder** background. Mid-range similarity uses gray.
    """)
    url = st.text_input("Enter URL (Optional):", key="heatmap_url", value="")
    use_url = st.checkbox("Use URL for Text Input", key="heatmap_use_url", value=bool(url))
    input_text_area = st.text_area("Enter your text:", key="heatmap_input", height=300, value="", disabled=use_url)
    search_term = st.text_input("Enter your search term:", key="heatmap_search", value="")

    # Determine text source
    text_to_analyze = ""
    if use_url:
        if url:
            with st.spinner(f"Extracting text from {url}..."):
                text_to_analyze = extract_text_from_url(url)
                if not text_to_analyze:
                    st.error(f"Could not extract text from {url}.")
                    return
        else:
            st.warning("Please enter a URL or uncheck the box.")
            return
    else:
        text_to_analyze = input_text_area
        if not text_to_analyze:
            st.warning("Please enter text or use the URL option.")
            return

    if st.button("Generate Heatmap", key="heatmap_button"):
        if not search_term:
            st.warning("Please enter a search term.")
            return
        if not text_to_analyze:
            st.warning("No text to analyze.")
            return

        with st.spinner("Generating highlighted text..."):
            # Use the updated highlight_text function
            highlighted_html_output = highlight_text(text_to_analyze, search_term)

        st.markdown("### Highlighted Content:")
        # Use components.html for better rendering control if needed, but markdown often works
        st.markdown(highlighted_html_output, unsafe_allow_html=True)


def top_bottom_embeddings_page():
    st.header("Top & Bottom Relevant Sentences")
    st.markdown("Identify the sentences most and least similar to your search term. Consider reviewing/rewriting the bottom sentences for relevance.")
    url = st.text_input("Enter URL (Optional):", key="tb_url", value="")
    use_url = st.checkbox("Use URL for Text Input", key="tb_use_url", value=bool(url))
    text_area = st.text_area("Enter your text:", key="top_bottom_text", height=300, value="", disabled=use_url)
    search_term = st.text_input("Enter your search term:", key="top_bottom_search", value="")
    top_n = st.slider("Number of top/bottom results:", min_value=1, max_value=25, value=10, key="top_bottom_slider")

    # Determine text source
    text_to_analyze = ""
    if use_url:
        if url:
            with st.spinner(f"Extracting text from {url}..."):
                text_to_analyze = extract_text_from_url(url)
                if not text_to_analyze:
                    st.error(f"Could not extract text from {url}.")
                    return
        else:
            st.warning("Please enter a URL or uncheck the box.")
            return
    else:
        text_to_analyze = text_area
        if not text_to_analyze:
            st.warning("Please enter text or use the URL option.")
            return

    if st.button("Find Top & Bottom Sentences", key="top_bottom_button"):
        if not search_term:
            st.warning("Please enter a search term.")
            return
        if not text_to_analyze:
            st.warning("No text to analyze.")
            return

        model = initialize_sentence_transformer() # Ensure model is loaded inside button click if needed
        if not model: return # Stop if model fails

        with st.spinner("Ranking sentences by similarity..."):
            # Use the function that returns raw similarity scores
            top_sections, bottom_sections = rank_sections_by_similarity_bert(text_to_analyze, search_term, top_n)

        st.subheader(f"Top {len(top_sections)} Sentences (Highest Cosine Similarity):")
        if top_sections:
            for i, (sentence, score) in enumerate(top_sections, 1):
                st.write(f"{i}. (Score: {score:.4f}) {sentence}")
        else:
            st.info("No top sentences found.")

        st.subheader(f"Bottom {len(bottom_sections)} Sentences (Lowest Cosine Similarity):")
        if bottom_sections:
             # Display bottom sections in ascending order of similarity (least similar first)
             for i, (sentence, score) in enumerate(bottom_sections, 1): # Iterate directly as it's already sorted [-n:]
                 st.write(f"{i}. (Score: {score:.4f}) {sentence}")
        else:
            st.info("No bottom sentences found.")


def entity_analysis_page():
    st.header("Entity Topic Gap Analysis")
    st.markdown("Analyze multiple competitor sources to identify common entities missing on your target site, *and* entities unique to your target site.")
    st.markdown("Uses BERT for entity recognition and spaCy for lemmatization.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Competitor Sources")
        competitor_source_option = st.radio(
            "Select competitor content source:",
            options=["Extract from URL", "Paste Content"],
            index=0,
            key="entity_comp_source"
        )
        if competitor_source_option == "Extract from URL":
            competitor_input = st.text_area("Enter Competitor URLs (one per line):", height=150, key="entity_urls", value="")
            competitor_list = [url.strip() for url in competitor_input.splitlines() if url.strip()]
            competitor_ids = competitor_list # Use URLs as IDs
        else:
            st.markdown("Paste competitor content below. Separate each source with `---` on a new line.")
            competitor_input = st.text_area("Enter Competitor Content:", key="entity_competitor_text", value="", height=200)
            competitor_list = [content.strip() for content in competitor_input.split('---') if content.strip()]
            competitor_ids = [f"Pasted Source {i+1}" for i in range(len(competitor_list))] # Assign generic IDs

    with col2:
        st.subheader("Target Site")
        target_option = st.radio(
            "Select target content source:",
            options=["Extract from URL", "Paste Content"],
            index=0,
            key="target_source"
        )
        if target_option == "Extract from URL":
            target_input_id = st.text_input("Enter Target URL:", key="target_url", value="")
            target_input_content = None # Will be fetched
        else:
            target_input_content = st.text_area("Paste target content:", key="target_text", value="", height=100)
            target_input_id = "Pasted Target Content" # Generic ID

    st.subheader("Analysis Options")
    exclude_input = st.text_area("Exclude Entities Mentioned in Pasted Text (Optional):", key="exclude_text", value="", height=100,
                                 help="Paste text containing brand names or irrelevant entities you want to ignore in the gap analysis.")
    all_entity_types = ["PER", "ORG", "LOC", "MISC", # Common BERT types (dslim/bert-base-NER)
                         "GPE", "NORP", "FAC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", # spaCy types often mapped
                         "CARDINAL", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL"] # spaCy specific types
    default_exclude = ["CARDINAL", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL"]

    exclude_types = st.multiselect(
        "Select entity types to exclude from analysis:",
        options=sorted(list(set(all_entity_types))), # Show unique sorted list
        default=[t for t in default_exclude if t in all_entity_types], # Default only if type exists
        key="entity_exclude_types"
    )
    min_sites_threshold = st.slider("Minimum # of Competitor Sites for Gap Entity:", min_value=1, max_value=max(1, len(competitor_list)), value=max(1, min(2, len(competitor_list))), step=1, key="min_sites_gap",
                                    help="Only show entities missing from your target but present on at least this many competitor sites.")

    if st.button("Analyze Entity Gaps", key="entity_button"):
        if not competitor_list:
            st.warning("Please provide at least one competitor source (URL or pasted text).")
            return
        if (target_option == "Extract from URL" and not target_input_id) or \
           (target_option == "Paste Content" and not target_input_content):
            st.warning("Please provide the target source (URL or pasted text).")
            return

        with st.spinner("Performing entity analysis... This may take some time."):
            # Load models
            nlp_spacy = load_spacy_model() # For lemmatization
            ner_bert_pipe = load_bert_ner_pipeline() # For NER

            if not nlp_spacy or not ner_bert_pipe:
                 st.error("Required NLP models failed to load. Aborting analysis.")
                 return

            # --- Process Target ---
            st.write("Processing Target...")
            if target_option == "Extract from URL":
                target_text = extract_text_from_url(target_input_id)
            else:
                target_text = target_input_content

            target_entities = []
            target_entity_counts = Counter() # Lemmatized unique counts
            target_entities_set = set() # Set of (lemma, label)

            if target_text:
                raw_target_entities = identify_entities(target_text, nlp_spacy) # Pass spaCy for identify_entities (though it uses BERT now)
                # Filter raw entities by type *before* counting/lemmatizing
                filtered_raw_target = [(entity, label) for entity, label in raw_target_entities if label not in exclude_types]
                # Count unique lemmatized entities for the target
                target_entity_counts = count_entities(filtered_raw_target, nlp_spacy)
                target_entities_set = set(target_entity_counts.keys()) # Get the set of (lemma, label) tuples
            else:
                st.warning(f"Could not retrieve or process target content ({target_input_id}).")


            # --- Process Exclude List ---
            exclude_lemmas_set = set()
            if exclude_input:
                st.write("Processing Exclude List...")
                exclude_doc = nlp_spacy(exclude_input)
                # Lemmatize excluded entities
                for ent in exclude_doc.ents:
                    lemma = " ".join([token.lemma_.lower() for token in ent if not token.is_stop and not token.is_punct and token.lemma_])
                    if not lemma: lemma = ent.text.lower()
                    exclude_lemmas_set.add(lemma)


            # --- Process Competitors ---
            st.write(f"Processing {len(competitor_list)} Competitor Sources...")
            entity_counts_per_source = {} # Stores { source_id: Counter((lemma, label): count) }
            all_competitor_entities_raw = [] # For overall analysis if needed later
            total_competitors = len(competitor_list)
            competitor_progress = st.progress(0)

            for i, source_content in enumerate(competitor_list):
                source_id = competitor_ids[i] # Get URL or "Pasted Source X"
                st.write(f"...analyzing {source_id}")

                text_to_process = ""
                if competitor_source_option == "Extract from URL":
                    text_to_process = extract_text_from_url(source_content) # source_content is URL here
                else:
                    text_to_process = source_content # source_content is the pasted text

                source_counts = Counter() # Unique lemmatized counts for this source
                if text_to_process:
                    raw_entities = identify_entities(text_to_process, nlp_spacy)
                    all_competitor_entities_raw.extend(raw_entities) # Add to overall list

                    # Filter raw entities by type *and* check against exclude list lemmas
                    filtered_raw = []
                    for entity, label in raw_entities:
                        if label not in exclude_types:
                            # Check if the lemmatized version is in the exclude set
                            doc = nlp_spacy(entity)
                            lemma = " ".join([token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and token.lemma_])
                            if not lemma: lemma = entity.lower()

                            if lemma not in exclude_lemmas_set:
                                filtered_raw.append((entity, label))

                    # Count unique lemmatized entities for this source
                    source_counts = count_entities(filtered_raw, nlp_spacy)
                else:
                     st.warning(f"Could not retrieve or process content for competitor: {source_id}")

                entity_counts_per_source[source_id] = source_counts
                competitor_progress.progress((i + 1) / total_competitors)


            # --- Calculate Gap & Unique Entities ---
            st.write("Calculating Gap and Unique Entities...")
            # Gap: Entities present in competitors but NOT in target
            gap_entities_site_count = Counter() # Counts how many *sites* an entity (lemma, label) appears on
            for source_id, counts in entity_counts_per_source.items():
                for entity_lemma_label, count in counts.items(): # count here is 1 because count_entities gives unique per site
                    if entity_lemma_label not in target_entities_set:
                         # Increment site count for this gap entity
                         gap_entities_site_count[entity_lemma_label] += 1

            # Filter gap entities by the minimum site threshold
            final_gap_entities = {
                entity_lemma_label: site_count
                for entity_lemma_label, site_count in gap_entities_site_count.items()
                if site_count >= min_sites_threshold
            }

            # Unique Target: Entities present in target but NOT in ANY competitor
            competitor_entities_set_all = set() # Set of all (lemma, label) found across all competitors
            for source_id, counts in entity_counts_per_source.items():
                competitor_entities_set_all.update(counts.keys())

            unique_target_entities_counts = Counter() # Counts from target doc
            for entity_lemma_label, count in target_entity_counts.items():
                 # Check if the entity (lemma, label) is NOT in the set of all competitor entities
                 if entity_lemma_label not in competitor_entities_set_all:
                      # Check if the lemma is not in the excluded lemmas set
                      if entity_lemma_label[0] not in exclude_lemmas_set:
                          unique_target_entities_counts[entity_lemma_label] = count


            # --- Display Results ---
            st.markdown("---")
            st.subheader(f"Gap Entities (Missing from Target, Found on ≥ {min_sites_threshold} Competitors)")
            if final_gap_entities:
                 # Create DataFrame for gap entities
                 gap_data = []
                 # Sort gap entities by the number of sites they appear on
                 sorted_gap = sorted(final_gap_entities.items(), key=lambda item: item[1], reverse=True)

                 st.write("Fetching Wikidata links for gap entities...")
                 wikidata_progress = st.progress(0)
                 total_gap = len(sorted_gap)

                 for i, ((lemma, label), site_count) in enumerate(sorted_gap):
                     wikidata_url = get_wikidata_link(lemma) # Query Wikidata using the lemma
                     gap_data.append({
                         "Entity (Lemma)": lemma,
                         "Type": label,
                         "# Competitor Sites": site_count,
                         "Wikidata Link": f"[Link]({wikidata_url})" if wikidata_url else "Not Found"
                     })
                     wikidata_progress.progress((i+1)/total_gap)

                 df_gap = pd.DataFrame(gap_data)
                 # Display as clickable links in the dataframe
                 st.dataframe(df_gap, use_container_width=True,
                              column_config={
                                   "Wikidata Link": st.column_config.LinkColumn(
                                        "Wikidata Link", display_text="Link"
                                   )
                              })

                 # Display Barchart for Gap Entities (using site count)
                 gap_counter_for_chart = Counter(dict(sorted_gap)) # Convert sorted list back to counter for chart func
                 display_entity_barchart(gap_counter_for_chart, top_n=30)

            else:
                st.info("No significant gap entities found based on the criteria.")


            st.markdown("---")
            st.subheader("Unique Entities (Found on Target, Not on Competitors)")
            if unique_target_entities_counts:
                 # Create DataFrame for unique entities
                 unique_data = []
                 sorted_unique = unique_target_entities_counts.most_common()

                 st.write("Fetching Wikidata links for unique entities...")
                 wikidata_progress_unique = st.progress(0)
                 total_unique = len(sorted_unique)

                 for i, ((lemma, label), count) in enumerate(sorted_unique):
                      wikidata_url = get_wikidata_link(lemma)
                      unique_data.append({
                          "Entity (Lemma)": lemma,
                          "Type": label,
                          "Frequency on Target": count,
                          "Wikidata Link": f"[Link]({wikidata_url})" if wikidata_url else "Not Found"
                      })
                      wikidata_progress_unique.progress((i+1)/total_unique)

                 df_unique = pd.DataFrame(unique_data)
                 st.dataframe(df_unique, use_container_width=True,
                              column_config={
                                   "Wikidata Link": st.column_config.LinkColumn(
                                        "Wikidata Link", display_text="Link"
                                   )
                              })

                 # Display Barchart for Unique Target Entities (using frequency on target)
                 display_entity_barchart(unique_target_entities_counts, top_n=30)
            else:
                 st.info("No unique entities found on the target site compared to competitors (after filtering).")

            # Optional: Display entities per competitor
            with st.expander("Show Top Entities per Competitor Source"):
                 for source_id, entity_counts_local in entity_counts_per_source.items():
                     st.markdown(f"#### Source: {source_id}")
                     if entity_counts_local:
                         # Convert Counter to list of dicts for DataFrame
                         competitor_entity_data = [{ "Entity (Lemma)": lemma, "Type": label, "Count": count}
                                                   for (lemma, label), count in entity_counts_local.most_common(50)]
                         df_comp_entities = pd.DataFrame(competitor_entity_data)
                         st.dataframe(df_comp_entities, height=200) # Limit height
                         # Optionally show a bar chart per competitor
                         # display_entity_barchart(entity_counts_local, top_n=10)
                     else:
                         st.write("No relevant entities found for this source after filtering.")


def displacy_visualization_page():
    st.header("Entity Visualizer (spaCy displaCy)")
    st.markdown("Visualize named entities identified by spaCy within your content.")
    url = st.text_input("Enter a URL (Optional):", key="displacy_url", value="")
    use_url = st.checkbox("Use URL for Text Input", key="displacy_use_url", value=bool(url))
    text_input = st.text_area("Enter Text:", key="displacy_text", value="", height=300, disabled=use_url)

    # Determine text source
    text_to_analyze = ""
    if use_url:
        if url:
            with st.spinner("Extracting text from URL..."):
                text_to_analyze = extract_text_from_url(url)
                if not text_to_analyze:
                    st.error("Could not extract text from the URL.")
                    return
        else:
            st.warning("Please enter a URL or uncheck the box.")
            return
    else:
        text_to_analyze = text_input
        if not text_to_analyze:
            st.warning("Please enter text or use the URL option.")
            return

    if st.button("Visualize Entities", key="displacy_button"):
        if not text_to_analyze:
            st.warning("No text to visualize.")
            return

        nlp_model = load_spacy_model() # Load the specific spaCy model needed for displaCy
        if not nlp_model:
            st.error("spaCy model ('en_core_web_md') could not be loaded for visualization.")
            return

        with st.spinner("Processing text with spaCy and rendering visualization..."):
            # Process the text with the spaCy model
            doc = nlp_model(text_to_analyze)

            # Define entity types and colors (customize as needed)
            # Colors should be visually distinct
            colors = {
                "PERSON": "#FFADAD", "NORP": "#FFD6A5", "FAC": "#FDFFB6",
                "ORG": "#CAFFBF", "GPE": "#9BF6FF", "LOC": "#A0C4FF",
                "PRODUCT": "#BDB2FF", "EVENT": "#FFC6FF", "WORK_OF_ART": "#FFFFFC",
                "LAW": "#E0E0E0", "LANGUAGE": "#D1D1D1", "DATE": "#C2C2C2",
                "TIME": "#B3B3B3", "PERCENT": "#A4A4A4", "MONEY": "#959595",
                "QUANTITY": "#868686", "ORDINAL": "#777777", "CARDINAL": "#686868",
                "MISC": "#FDE4CF" # Added color for MISC from BERT
            }
            options = {"ents": list(colors.keys()), "colors": colors} # Include all defined ents

            try:
                # Render the visualization
                # Use page=False to get HTML string, page=True tries to serve a page
                html = spacy.displacy.render(doc, style="ent", options=options, jupyter=False)

                # Display using st.components.v1.html
                # Add some styling for better presentation
                st.markdown("### Entity Visualization:")
                st.components.v1.html(f"<div style='border: 1px solid #eee; padding: 15px; border-radius: 5px; line-height: 2.0;'>{html}</div>", height=600, scrolling=True)

            except Exception as e:
                st.error(f"Error rendering visualization with displaCy: {e}")


def named_entity_barchart_page():
    st.header("Entity Frequency Charts (Total Occurrences)")
    st.markdown("Visualize the most frequent named entities (total count) across one or more sources (URLs or pasted text).")
    st.markdown("Uses BERT for entity recognition and spaCy for lemmatization.")

    input_method = st.radio(
        "Select content input method:",
        options=["Extract from URL", "Paste Content"],
        key="entity_barchart_input", index=0
    )

    sources_data = {} # Store {source_id: text}

    if input_method == "Paste Content":
        st.markdown("Please paste your content. For multiple sources, separate each block with `---` on a new line.")
        text_input = st.text_area("Enter Text:", key="barchart_text", height=300, value="")
        pasted_sources = [content.strip() for content in text_input.split('---') if content.strip()]
        for i, content in enumerate(pasted_sources):
            sources_data[f"Pasted Source {i+1}"] = content
    else:
        st.markdown("Enter one or more URLs (one per line). The app will fetch and combine the text from each URL.")
        urls_input = st.text_area("Enter URLs (one per line):", key="barchart_url", height=150, value="")
        urls = [url.strip() for url in urls_input.splitlines() if url.strip()]
        if urls:
            with st.spinner("Extracting text from URLs..."):
                 url_progress = st.progress(0)
                 for i, url in enumerate(urls):
                     extracted_text = extract_text_from_url(url)
                     if extracted_text:
                         sources_data[url] = extracted_text
                     else:
                         st.warning(f"Couldn't retrieve text from {url}. It will be excluded.")
                     url_progress.progress((i+1)/len(urls))

    if st.button("Generate Visualizations", key="barchart_button"):
        if not sources_data:
             st.warning(f"Please provide content using the '{input_method}' method.")
             return

        with st.spinner("Analyzing entities and generating visualizations..."):
            nlp_spacy = load_spacy_model()
            ner_bert_pipe = load_bert_ner_pipeline() # Load BERT pipe (cached)

            if not nlp_spacy or not ner_bert_pipe:
                st.error("Required NLP models failed to load.")
                return

            all_entities_combined = []
            entities_by_source = {} # Store {source_id: Counter}

            for source_id, text in sources_data.items():
                # Identify entities using BERT
                raw_entities = identify_entities(text, nlp_spacy)
                all_entities_combined.extend(raw_entities)

                # Count total occurrences (lemmatized) for this specific source
                source_entity_counts = count_entities_total(raw_entities, nlp_spacy)
                entities_by_source[source_id] = source_entity_counts

            # Count total occurrences (lemmatized) across *all* combined sources
            combined_entity_counts = count_entities_total(all_entities_combined, nlp_spacy)


            if combined_entity_counts:
                st.subheader("Overall Entity Frequency (All Sources Combined)")
                # Display Bar Chart for combined counts
                display_entity_barchart(combined_entity_counts, top_n=30)

                st.subheader("Overall Entity Word Cloud (All Sources Combined)")
                # Display Word Cloud for combined counts
                display_entity_wordcloud(combined_entity_counts)

                # Display counts per source in an expander
                with st.expander("Show Top Entities per Source"):
                    for source_id, source_counts in entities_by_source.items():
                         st.markdown(f"#### Source: {source_id}")
                         if source_counts:
                              # Convert Counter to list of dicts for DataFrame
                              source_entity_data = [{"Entity (Lemma)": lemma, "Type": label, "Total Count": count}
                                                     for (lemma, label), count in source_counts.most_common(50)]
                              df_source_entities = pd.DataFrame(source_entity_data)
                              st.dataframe(df_source_entities, height=200) # Limit height
                         else:
                              st.write("No relevant entities found for this source after filtering.")
            else:
                st.warning("No relevant entities found in the provided content after processing.")


# ------------------------------------
# Semantic Gap Analyzer (TF-IDF + Similarity)
# ------------------------------------
def ngram_tfidf_analysis_page():
    st.header("Semantic Gap Analyzer (TF-IDF & Embeddings)")
    st.markdown("""
        Identify potentially important phrases (n-grams) used by competitors that might be underrepresented or missing in your target content.
        This tool combines TF-IDF frequency analysis with semantic similarity using sentence embeddings.
    """)

    # --- Input Columns ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Competitors")
        competitor_source_option = st.radio(
            "Select competitor content source:",
            options=["Extract from URL", "Paste Content"],
            index=0,
            key="competitor_source_gap" # Unique key
        )
        if competitor_source_option == "Extract from URL":
            competitor_input = st.text_area("Competitor URLs (one per line):", key="competitor_urls_gap", height=150, value="")
            competitor_list = [url.strip() for url in competitor_input.splitlines() if url.strip()]
            competitor_ids = competitor_list # Use URLs as identifiers
        else:
            st.markdown("Paste competitor content below (separate with `---` on new lines).")
            competitor_input = st.text_area("Competitor Content:", key="competitor_text_gap", value="", height=200)
            competitor_list = [content.strip() for content in competitor_input.split('---') if content.strip()]
            competitor_ids = [f"Pasted Comp {i+1}" for i in range(len(competitor_list))]

    with col2:
        st.subheader("Your Site (Target)")
        target_source_option = st.radio(
            "Select target content source:",
            options=["Extract from URL", "Paste Content"],
            index=0,
            key="target_source_gap" # Unique key
        )
        if target_source_option == "Extract from URL":
            target_input_id = st.text_input("Your Target URL:", key="target_url_gap", value="")
            target_input_content = None
        else:
            target_input_content = st.text_area("Paste your target content:", key="target_text_gap", value="", height=150)
            target_input_id = "Pasted Target"

    # --- Analysis Options ---
    st.subheader("Analysis Options")
    n_value = st.selectbox("Select Phrase Length (N-grams):", options=[1, 2, 3, 4, 5], index=1, key="ngram_n_gap",
                           help="Number of words per phrase (e.g., 2 for bigrams, 3 for trigrams).")
    # TF-IDF Parameters
    min_df = st.number_input("Minimum Document Frequency (Min DF):", value=1, min_value=1, key="min_df_gap_tfidf",
                             help="Ignore terms that appear in fewer than this many competitor documents.")
    max_df = st.number_input("Maximum Document Frequency (Max DF):", value=1.0, min_value=0.0, max_value=1.0, step=0.05, key="max_df_gap_tfidf",
                             help="Ignore terms that appear in more than this fraction of competitor documents (e.g., 0.95 to ignore very common terms).")
    # Results / LDA
    top_n = st.slider("Max N-grams per Competitor for Gap Calc:", min_value=5, max_value=100, value=25, key="top_n_gap_calc",
                      help="Consider only the top N most important n-grams from each competitor when calculating gaps.")
    num_topics_lda = st.slider("Number of Topics for Competitor LDA:", min_value=2, max_value=15, value=5, key="lda_topics_gap",
                             help="Group competitor content into this many topics using Latent Dirichlet Allocation.")
    tfidf_weight = st.slider("Weight for TF-IDF Score in Gap Calc (vs. Similarity):", min_value=0.0, max_value=1.0, value=0.4, step=0.1, key="tfidf_weight_gap",
                             help="Adjust the importance of TF-IDF difference (frequency) vs. Semantic Similarity difference in the final gap score (0=Similarity only, 1=TF-IDF only).")


    if st.button("Analyze Content Gaps", key="content_gap_button"):
        # --- Input Validation ---
        if not competitor_list:
            st.warning("Please provide at least one competitor source.")
            return
        if (target_source_option == "Extract from URL" and not target_input_id) or \
           (target_source_option == "Paste Content" and not target_input_content):
            st.warning("Please provide the target source.")
            return

        with st.spinner("Analyzing content gaps... This involves fetching, preprocessing, TF-IDF, LDA, and similarity calculations."):
            # --- Load Models ---
            nlp_spacy = load_spacy_model()
            sentence_model = initialize_sentence_transformer()
            if not nlp_spacy or not sentence_model:
                st.error("Required NLP models failed to load.")
                return

            # --- Fetch and Preprocess Competitor Content ---
            st.write("Processing Competitor Content...")
            competitor_texts = {} # {id: processed_text}
            competitor_raw_texts = {} # {id: raw_text} - needed for embeddings
            competitor_progress = st.progress(0)
            total_competitors = len(competitor_list)

            for i, source_content in enumerate(competitor_list):
                source_id = competitor_ids[i]
                raw_text = ""
                if competitor_source_option == "Extract from URL":
                    # Use the 'relevant' text extractor here for potentially better focus
                    raw_text = extract_relevant_text_from_url(source_content)
                else:
                    raw_text = source_content

                if raw_text:
                    competitor_raw_texts[source_id] = raw_text
                    # Preprocess for TF-IDF/LDA (lemmatization, stopwords etc.)
                    competitor_texts[source_id] = preprocess_text(raw_text, nlp_spacy)
                else:
                    st.warning(f"Could not get content for competitor: {source_id}. Skipping.")
                competitor_progress.progress((i+1)/total_competitors)

            if not competitor_texts:
                st.error("No valid competitor content could be processed.")
                return

            valid_competitor_ids = list(competitor_texts.keys())
            processed_competitor_corpus = list(competitor_texts.values())

            # --- Fetch and Preprocess Target Content ---
            st.write("Processing Target Content...")
            if target_source_option == "Extract from URL":
                target_raw_text = extract_relevant_text_from_url(target_input_id)
            else:
                target_raw_text = target_input_content

            if not target_raw_text:
                st.error(f"Could not retrieve or process target content ({target_input_id}). Aborting analysis.")
                return
            target_processed_text = preprocess_text(target_raw_text, nlp_spacy)


            # --- TF-IDF Vectorization (Competitors) ---
            st.write("Calculating TF-IDF for Competitors...")
            try:
                # Use ENGLISH_STOP_WORDS from sklearn + custom preprocessing handled by preprocess_text
                vectorizer = TfidfVectorizer(
                    ngram_range=(n_value, n_value),
                    min_df=min_df,
                    max_df=max_df,
                    stop_words=None # Stopwords already handled in preprocess_text
                )
                tfidf_matrix_competitors = vectorizer.fit_transform(processed_competitor_corpus)
                feature_names = vectorizer.get_feature_names_out()
                df_tfidf_competitors = pd.DataFrame(tfidf_matrix_competitors.toarray(), index=valid_competitor_ids, columns=feature_names)
            except ValueError as ve:
                st.error(f"TF-IDF Error: {ve}. This might happen if no n-grams meet the min/max df criteria after preprocessing.")
                if "empty vocabulary" in str(ve):
                     st.warning("Consider adjusting N-gram length, Min/Max DF, or check preprocessing steps.")
                return


            # --- LDA Topic Modeling (Competitors) ---
            st.write("Performing LDA Topic Modeling on Competitors...")
            topic_keywords = {}
            topic_distribution_df = pd.DataFrame() # Initialize empty dataframe
            try:
                # LDA uses CountVectorizer, not TF-IDF directly usually
                count_vectorizer_lda = CountVectorizer(stop_words='english', max_df=0.95, min_df=2)
                # Use the *processed* competitor corpus for LDA consistency
                count_matrix_lda = count_vectorizer_lda.fit_transform(processed_competitor_corpus)
                feature_names_lda = count_vectorizer_lda.get_feature_names_out()

                if count_matrix_lda.shape[1] == 0: # Check if vocabulary is empty
                     st.warning("LDA CountVectorizer created an empty vocabulary. Skipping LDA.")
                else:
                    lda_model = LatentDirichletAllocation(n_components=num_topics_lda, random_state=42, n_jobs=-1) # Use all CPU cores
                    lda_output = lda_model.fit_transform(count_matrix_lda)

                    # Display Topics
                    with st.expander("Show Identified Competitor Topics (LDA)"):
                        for i, topic_weights in enumerate(lda_model.components_):
                            top_keyword_indices = topic_weights.argsort()[-15:][::-1] # Top 15 keywords
                            keywords = [feature_names_lda[idx] for idx in top_keyword_indices]
                            st.markdown(f"**Topic {i+1}:** {', '.join(keywords)}")

                        # Topic Distribution per Competitor (Optional Display)
                        topic_distribution_df = pd.DataFrame(lda_output, index=valid_competitor_ids, columns=[f"Topic {i+1}" for i in range(num_topics_lda)])
                        st.dataframe(topic_distribution_df.style.format("{:.3f}"))

            except Exception as lda_e:
                 st.warning(f"LDA failed: {lda_e}. Skipping topic modeling display.")


            # --- TF-IDF Vectorization (Target) ---
            st.write("Calculating TF-IDF for Target...")
            try:
                target_tfidf_vector = vectorizer.transform([target_processed_text])
                df_tfidf_target = pd.DataFrame(target_tfidf_vector.toarray(), index=[target_input_id], columns=feature_names)
            except Exception as target_tfidf_e:
                 st.error(f"Error calculating TF-IDF for target: {target_tfidf_e}")
                 return


            # --- Embeddings Calculation ---
            st.write("Calculating Embeddings...")
            # Target Embedding (use RAW text)
            target_embedding = get_embedding(target_raw_text, sentence_model)
            if target_embedding is None:
                st.error("Failed to calculate embedding for target content.")
                return

            # Competitor Embeddings (use RAW text) - batch calculation
            competitor_embeddings_map = {}
            raw_competitor_corpus = [competitor_raw_texts[cid] for cid in valid_competitor_ids]
            try:
                competitor_embeddings_list = get_embedding(raw_competitor_corpus, sentence_model)
                if competitor_embeddings_list is not None and len(competitor_embeddings_list) == len(valid_competitor_ids):
                    for i, cid in enumerate(valid_competitor_ids):
                         competitor_embeddings_map[cid] = competitor_embeddings_list[i]
                else:
                     st.warning("Could not calculate embeddings for all competitors.")
                     # Handle partial failure - maybe skip similarity for those?
            except Exception as emb_e:
                st.error(f"Error calculating competitor embeddings: {emb_e}")
                return


            # --- Identify Top N-grams per Competitor ---
            top_ngrams_competitors = {}
            for source_id in valid_competitor_ids:
                # Ensure the source exists in the TF-IDF dataframe
                if source_id in df_tfidf_competitors.index:
                     row = df_tfidf_competitors.loc[source_id]
                     sorted_row = row.sort_values(ascending=False)
                     # Filter out zero-score ngrams before taking top N
                     top_ngrams = sorted_row[sorted_row > 0].head(top_n)
                     top_ngrams_competitors[source_id] = list(top_ngrams.index)
                else:
                     st.warning(f"Competitor {source_id} missing from TF-IDF results, skipping.")


            # --- Calculate Candidate Gap Scores ---
            st.write("Calculating Gap Scores...")
            candidate_scores = [] # Stores tuples: (source_id, ngram, tfidf_diff, similarity_diff, combined_score)
            epsilon = 1e-9 # Small value to prevent division by zero

            # Get embeddings for all unique candidate n-grams in one batch
            all_candidate_ngrams = set()
            for source_id in valid_competitor_ids:
                 if source_id in top_ngrams_competitors:
                     all_candidate_ngrams.update(top_ngrams_competitors[source_id])

            ngram_embedding_map = {}
            if all_candidate_ngrams:
                list_ngrams = list(all_candidate_ngrams)
                try:
                     ngram_embeddings_list = get_embedding(list_ngrams, sentence_model)
                     if ngram_embeddings_list is not None:
                         for i, ngram in enumerate(list_ngrams):
                              ngram_embedding_map[ngram] = ngram_embeddings_list[i]
                     else:
                          st.warning("Failed to calculate embeddings for n-grams.")
                except Exception as ngram_emb_e:
                     st.error(f"Error calculating n-gram embeddings: {ngram_emb_e}")
                     # Continue without similarity if n-gram embeddings fail? Or abort? Abort for now.
                     return


            # Calculate diffs and scores
            tfidf_diffs = []
            similarity_diffs = []

            for source_id in valid_competitor_ids:
                 if source_id not in top_ngrams_competitors or source_id not in competitor_embeddings_map:
                     continue # Skip if missing data

                 competitor_embedding = competitor_embeddings_map[source_id]

                 for ngram in top_ngrams_competitors[source_id]:
                     if ngram not in ngram_embedding_map: continue # Skip if ngram embedding failed

                     ngram_embedding = ngram_embedding_map[ngram]

                     # TF-IDF Difference
                     competitor_tfidf = df_tfidf_competitors.loc[source_id, ngram]
                     target_tfidf = df_tfidf_target.loc[target_input_id, ngram] if ngram in df_tfidf_target.columns else 0
                     tfidf_diff = competitor_tfidf - target_tfidf

                     # Similarity Difference
                     competitor_similarity = cosine_similarity(ngram_embedding.reshape(1, -1), competitor_embedding.reshape(1, -1))[0][0]
                     target_similarity = cosine_similarity(ngram_embedding.reshape(1, -1), target_embedding.reshape(1, -1))[0][0]
                     similarity_diff = competitor_similarity - target_similarity

                     # Store raw diffs for normalization later
                     # Only consider candidates where competitor score is higher in at least one metric (TFIDF or Sim)
                     # And TF-IDF diff must be positive (competitor TFIDF > target TFIDF)
                     if tfidf_diff > epsilon: # Check only positive TFIDF diff
                         candidate_scores.append([source_id, ngram, tfidf_diff, similarity_diff])
                         tfidf_diffs.append(tfidf_diff)
                         similarity_diffs.append(similarity_diff)

            if not candidate_scores:
                st.error("No potential gap n-grams were identified based on the criteria (e.g., positive TF-IDF difference). Try adjusting parameters.")
                return

            # --- Normalize and Combine Scores ---
            min_tfidf, max_tfidf = min(tfidf_diffs), max(tfidf_diffs)
            min_sim, max_sim = min(similarity_diffs), max(similarity_diffs)

            range_tfidf = max_tfidf - min_tfidf
            range_sim = max_sim - min_sim

            final_candidates = []
            for i in range(len(candidate_scores)):
                 source_id, ngram, tfidf_diff, similarity_diff = candidate_scores[i]

                 # Normalize TF-IDF difference (0-1)
                 norm_tfidf = (tfidf_diff - min_tfidf) / (range_tfidf + epsilon) if range_tfidf > 0 else 0.5

                 # Normalize Similarity difference (0-1)
                 norm_sim = (similarity_diff - min_sim) / (range_sim + epsilon) if range_sim > 0 else 0.5

                 # Combined Score
                 combined_score = (tfidf_weight * norm_tfidf) + ((1 - tfidf_weight) * norm_sim)

                 # Store with combined score
                 final_candidates.append({
                     "Competitor": source_id,
                     "N-gram": ngram,
                     "TFIDF_Diff": tfidf_diff,
                     "Similarity_Diff": similarity_diff,
                     "Gap Score": combined_score
                 })

            # Sort by combined score
            final_candidates.sort(key=lambda x: x["Gap Score"], reverse=True)

            # --- Display Results ---
            st.markdown("---")
            st.subheader("Top Semantic Gaps Identified")

            df_consolidated = pd.DataFrame(final_candidates)

            # Formatting for display
            format_dict_gap = {
                 "TFIDF_Diff": "{:.4f}",
                 "Similarity_Diff": "{:.4f}",
                 "Gap Score": "{:.4f}"
            }

            st.dataframe(df_consolidated.style.format(format_dict_gap), use_container_width=True)

            # --- Word Clouds ---
            st.subheader("Semantic Gap Word Clouds")

            # Combined Word Cloud (weighted by Gap Score)
            combined_gap_scores = {}
            for item in final_candidates:
                ngram = item["N-gram"]
                score = item["Gap Score"]
                # Use max score if ngram appears multiple times
                combined_gap_scores[ngram] = max(combined_gap_scores.get(ngram, 0), score)

            if combined_gap_scores:
                 st.markdown("**Combined Gap Cloud (All Competitors)**")
                 display_entity_wordcloud(combined_gap_scores) # Reuse entity wordcloud func
            else:
                 st.write("No combined gap n-grams for word cloud.")

            # Per-Competitor Word Clouds
            with st.expander("Show Word Clouds per Competitor"):
                for source_id in valid_competitor_ids:
                     comp_gap_scores = {}
                     for item in final_candidates:
                         if item["Competitor"] == source_id:
                              comp_gap_scores[item["N-gram"]] = max(comp_gap_scores.get(item["N-gram"], 0), item["Gap Score"])

                     if comp_gap_scores:
                          st.markdown(f"**Word Cloud for: {source_id}**")
                          display_entity_wordcloud(comp_gap_scores)
                     else:
                          st.write(f"No gap n-grams found for competitor: {source_id}")


def keyword_clustering_from_gap_page():
    st.header("Keyword Clustering from Semantic Gaps")
    st.markdown("""
        This tool first identifies semantic gaps (similar to the 'Semantic Gap Analyzer')
        and then clusters the identified gap n-grams based on their semantic meaning using embeddings.
        Helps group related content opportunities together.
    """)

    # --- Inputs (Similar to Gap Analyzer) ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Competitors")
        competitor_source_option = st.radio("Competitor Content Source:", options=["Extract from URL", "Paste Content"], index=0, key="comp_source_cluster")
        if competitor_source_option == "Extract from URL":
            competitor_input = st.text_area("Competitor URLs:", key="comp_urls_cluster", height=150, value="")
            competitor_list = [url.strip() for url in competitor_input.splitlines() if url.strip()]
            competitor_ids = competitor_list
        else:
            st.markdown("Paste competitor content (`---` separator).")
            competitor_input = st.text_area("Competitor Content:", key="competitor_content_cluster", value="", height=200)
            competitor_list = [content.strip() for content in competitor_input.split('---') if content.strip()]
            competitor_ids = [f"Pasted Comp {i+1}" for i in range(len(competitor_list))]
    with col2:
        st.subheader("Your Site (Target)")
        target_source_option = st.radio("Target Content Source:", options=["Extract from URL", "Paste Content"], index=0, key="target_source_cluster")
        if target_source_option == "Extract from URL":
            target_input_id = st.text_input("Your Target URL:", key="target_url_cluster", value="")
            target_input_content = None
        else:
            target_input_content = st.text_area("Paste your target content:", key="target_content_cluster", value="", height=150)
            target_input_id = "Pasted Target"

    st.subheader("Gap Analysis & Clustering Settings")
    n_value = st.selectbox("Phrase Length (N-grams):", options=[1, 2, 3, 4, 5], index=1, key="ngram_n_cluster")
    min_df = st.number_input("Min Document Frequency:", value=1, min_value=1, key="min_df_cluster")
    max_df = st.number_input("Max Document Frequency:", value=1.0, min_value=0.0, step=0.05, key="max_df_cluster")
    top_n = st.slider("Max N-grams per Competitor:", min_value=5, max_value=100, value=25, key="top_n_cluster")
    tfidf_weight = st.slider("TF-IDF Weight in Gap Score:", min_value=0.0, max_value=1.0, value=0.4, step=0.1, key="tfidf_weight_cluster")

    st.markdown("---") # Separator
    algorithm = st.selectbox("Clustering Algorithm:", options=["KMeans (Centroid-based)", "Agglomerative (Hierarchical)"], key="clustering_algo_gap")
    n_clusters = st.number_input("Desired Number of Clusters:", min_value=2, max_value=30, value=8, key="clusters_num_gap") # Default 8
    dim_reduction = st.selectbox("Dimension Reduction for Plot:", options=["UMAP", "PCA"], key="dim_reduction_cluster", help="UMAP often shows cluster structure better, PCA is faster.")


    if st.button("Analyze Gaps & Cluster Keywords", key="gap_cluster_button"):
        # --- Input Validation ---
        if not competitor_list: st.warning("Please provide competitor sources."); return
        if (target_source_option == "Extract from URL" and not target_input_id) or \
           (target_source_option == "Paste Content" and not target_input_content):
            st.warning("Please provide the target source."); return

        with st.spinner("Running Gap Analysis and Clustering... This may take a while."):
            # --- 1. Perform Semantic Gap Analysis (reuse logic/steps from ngram_tfidf_analysis_page) ---
            # Load Models
            nlp_spacy = load_spacy_model()
            sentence_model = initialize_sentence_transformer()
            if not nlp_spacy or not sentence_model: st.error("Models failed to load."); return

            # Process Competitors
            competitor_texts = {}
            competitor_raw_texts = {}
            for i, source_content in enumerate(competitor_list):
                source_id = competitor_ids[i]
                raw_text = extract_relevant_text_from_url(source_content) if competitor_source_option == "Extract from URL" else source_content
                if raw_text:
                    competitor_raw_texts[source_id] = raw_text
                    competitor_texts[source_id] = preprocess_text(raw_text, nlp_spacy)
                else: st.warning(f"Skipping competitor: {source_id} (no content)")
            if not competitor_texts: st.error("No valid competitor content."); return
            valid_competitor_ids = list(competitor_texts.keys())
            processed_competitor_corpus = list(competitor_texts.values())

            # Process Target
            target_raw_text = extract_relevant_text_from_url(target_input_id) if target_source_option == "Extract from URL" else target_input_content
            if not target_raw_text: st.error("Could not process target content."); return
            target_processed_text = preprocess_text(target_raw_text, nlp_spacy)

            # TF-IDF
            try:
                vectorizer = TfidfVectorizer(ngram_range=(n_value, n_value), min_df=min_df, max_df=max_df, stop_words=None)
                tfidf_matrix_competitors = vectorizer.fit_transform(processed_competitor_corpus)
                feature_names = vectorizer.get_feature_names_out()
                df_tfidf_competitors = pd.DataFrame(tfidf_matrix_competitors.toarray(), index=valid_competitor_ids, columns=feature_names)
                target_tfidf_vector = vectorizer.transform([target_processed_text])
                df_tfidf_target = pd.DataFrame(target_tfidf_vector.toarray(), index=[target_input_id], columns=feature_names)
            except ValueError as ve: st.error(f"TF-IDF Error: {ve}"); return

            # Embeddings
            target_embedding = get_embedding(target_raw_text, sentence_model)
            competitor_embeddings_map = {}
            raw_competitor_corpus = [competitor_raw_texts[cid] for cid in valid_competitor_ids]
            competitor_embeddings_list = get_embedding(raw_competitor_corpus, sentence_model)
            if competitor_embeddings_list is None or len(competitor_embeddings_list) != len(valid_competitor_ids):
                 st.warning("Could not get all competitor embeddings."); # Handle partially?
            else:
                 for i, cid in enumerate(valid_competitor_ids): competitor_embeddings_map[cid] = competitor_embeddings_list[i]
            if target_embedding is None or not competitor_embeddings_map: st.error("Embedding calculation failed."); return

            # Top N-grams per competitor
            top_ngrams_competitors = {}
            for source_id in valid_competitor_ids:
                if source_id in df_tfidf_competitors.index:
                     row = df_tfidf_competitors.loc[source_id]
                     top_ngrams = row[row > 0].sort_values(ascending=False).head(top_n)
                     top_ngrams_competitors[source_id] = list(top_ngrams.index)

            # Get N-gram Embeddings
            all_candidate_ngrams = set(ng for source_id in valid_competitor_ids if source_id in top_ngrams_competitors for ng in top_ngrams_competitors[source_id])
            ngram_embedding_map = {}
            if all_candidate_ngrams:
                list_ngrams = list(all_candidate_ngrams)
                ngram_embeddings_list = get_embedding(list_ngrams, sentence_model)
                if ngram_embeddings_list is not None:
                     for i, ngram in enumerate(list_ngrams): ngram_embedding_map[ngram] = ngram_embeddings_list[i]
                else: st.warning("Failed to get n-gram embeddings.")

            # Calculate Gap Scores
            candidate_scores = []
            tfidf_diffs = []
            similarity_diffs = []
            epsilon = 1e-9
            for source_id in valid_competitor_ids:
                if source_id not in top_ngrams_competitors or source_id not in competitor_embeddings_map: continue
                competitor_embedding = competitor_embeddings_map[source_id]
                for ngram in top_ngrams_competitors[source_id]:
                     if ngram not in ngram_embedding_map: continue
                     ngram_embedding = ngram_embedding_map[ngram]
                     competitor_tfidf = df_tfidf_competitors.loc[source_id, ngram]
                     target_tfidf = df_tfidf_target.loc[target_input_id, ngram] if ngram in df_tfidf_target.columns else 0
                     tfidf_diff = competitor_tfidf - target_tfidf
                     if tfidf_diff > epsilon: # Only positive TFIDF diff
                         competitor_similarity = cosine_similarity(ngram_embedding.reshape(1, -1), competitor_embedding.reshape(1, -1))[0][0]
                         target_similarity = cosine_similarity(ngram_embedding.reshape(1, -1), target_embedding.reshape(1, -1))[0][0]
                         similarity_diff = competitor_similarity - target_similarity
                         candidate_scores.append([source_id, ngram, tfidf_diff, similarity_diff])
                         tfidf_diffs.append(tfidf_diff)
                         similarity_diffs.append(similarity_diff)
            if not candidate_scores: st.error("No gap n-grams identified for clustering."); return

            # Normalize and Combine
            min_tfidf, max_tfidf = min(tfidf_diffs), max(tfidf_diffs)
            min_sim, max_sim = min(similarity_diffs), max(similarity_diffs)
            range_tfidf = max_tfidf - min_tfidf
            range_sim = max_sim - min_sim
            final_candidates = [] # Stores dicts: { "Competitor": ..., "N-gram": ..., "Gap Score": ... }
            gap_ngrams_scores = {} # Stores { ngram: max_gap_score } for clustering input
            for i in range(len(candidate_scores)):
                source_id, ngram, tfidf_diff, similarity_diff = candidate_scores[i]
                norm_tfidf = (tfidf_diff - min_tfidf) / (range_tfidf + epsilon) if range_tfidf > 0 else 0.5
                norm_sim = (similarity_diff - min_sim) / (range_sim + epsilon) if range_sim > 0 else 0.5
                combined_score = (tfidf_weight * norm_tfidf) + ((1 - tfidf_weight) * norm_sim)
                final_candidates.append({"Competitor": source_id, "N-gram": ngram, "Gap Score": combined_score})
                # Keep track of the highest score for each unique ngram
                gap_ngrams_scores[ngram] = max(gap_ngrams_scores.get(ngram, 0), combined_score)

            if not gap_ngrams_scores:
                 st.error("No gap n-grams with positive scores found after calculation.")
                 return

            # --- 2. Clustering Stage ---
            st.write("Clustering identified gap n-grams...")
            gap_ngrams_list = list(gap_ngrams_scores.keys())
            # Get embeddings specifically for the ngrams to be clustered
            gap_embeddings_list = [ngram_embedding_map[ng] for ng in gap_ngrams_list if ng in ngram_embedding_map]

            if len(gap_embeddings_list) != len(gap_ngrams_list):
                 st.warning("Mismatch between n-grams and embeddings. Clustering might be incomplete.")
                 # Recreate list based on available embeddings
                 gap_ngrams_list = [ng for ng in gap_ngrams_list if ng in ngram_embedding_map]
                 if not gap_ngrams_list:
                      st.error("No valid n-gram embeddings available for clustering.")
                      return

            gap_embeddings_np = np.array(gap_embeddings_list) # Convert to NumPy array

            # Perform Clustering
            try:
                 if algorithm == "KMeans (Centroid-based)":
                      # Ensure n_clusters is not more than samples
                      actual_n_clusters = min(n_clusters, len(gap_ngrams_list))
                      if actual_n_clusters < 2:
                          st.warning(f"Need at least 2 keywords to form clusters. Found {len(gap_ngrams_list)}. Skipping clustering.")
                          cluster_labels = [0] * len(gap_ngrams_list) # Assign all to cluster 0
                      else:
                          clustering_model = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init='auto')
                          cluster_labels = clustering_model.fit_predict(gap_embeddings_np)
                 elif algorithm == "Agglomerative (Hierarchical)":
                      actual_n_clusters = min(n_clusters, len(gap_ngrams_list))
                      if actual_n_clusters < 2:
                           st.warning(f"Need at least 2 keywords to form clusters. Found {len(gap_ngrams_list)}. Skipping clustering.")
                           cluster_labels = [0] * len(gap_ngrams_list)
                      else:
                           # Linkage can be 'ward', 'average', 'complete', 'single'
                           clustering_model = AgglomerativeClustering(n_clusters=actual_n_clusters, linkage='ward')
                           cluster_labels = clustering_model.fit_predict(gap_embeddings_np)
            except Exception as cluster_e:
                 st.error(f"Clustering algorithm failed: {cluster_e}")
                 return


            # --- 3. Dimension Reduction and Visualization ---
            st.write("Reducing dimensions for visualization...")
            try:
                if dim_reduction == "UMAP":
                     reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(gap_ngrams_list)-1), min_dist=0.1) # Adjust UMAP params
                else: # PCA
                     reducer = PCA(n_components=2, random_state=42)

                if len(gap_embeddings_np) < 2:
                     st.warning("Need at least 2 data points for dimension reduction. Skipping plot.")
                     embeddings_2d = None
                else:
                     embeddings_2d = reducer.fit_transform(gap_embeddings_np)

            except Exception as dr_e:
                 st.error(f"Dimension reduction ({dim_reduction}) failed: {dr_e}")
                 embeddings_2d = None # Set to None if reduction fails


            # --- Display Results ---
            st.markdown("---")
            st.subheader("Keyword Cluster Visualization")

            if embeddings_2d is not None:
                df_plot = pd.DataFrame({
                    'x': embeddings_2d[:, 0],
                    'y': embeddings_2d[:, 1],
                    'Keyword': gap_ngrams_list,
                    'Cluster': [f"Cluster {label+1}" for label in cluster_labels], # 1-based cluster index
                    'Gap Score': [gap_ngrams_scores[ng] for ng in gap_ngrams_list]
                })

                fig = px.scatter(df_plot, x='x', y='y',
                                 color='Cluster',
                                 size='Gap Score', # Size bubbles by gap score
                                 text='Keyword',
                                 hover_data=['Keyword', 'Cluster', 'Gap Score'],
                                 title=f"Semantic Keyword Clusters ({dim_reduction})",
                                 color_discrete_sequence=px.colors.qualitative.Plotly) # Use a qualitative color scale
                fig.update_traces(textposition='top center', textfont_size=9)
                fig.update_layout(
                     xaxis_title=f"{dim_reduction} Dimension 1",
                     yaxis_title=f"{dim_reduction} Dimension 2",
                     height=700
                 )
                st.plotly_chart(fig, use_container_width=True)
            else:
                 st.info("Could not generate 2D visualization.")


            # --- Display Detailed Clusters ---
            st.subheader("Keyword Clusters Details")
            clusters = {} # { cluster_label: [ (ngram, score), ... ] }
            for i, gram in enumerate(gap_ngrams_list):
                label = cluster_labels[i] + 1 # 1-based index
                score = gap_ngrams_scores[gram]
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append((gram, score))

            # Sort keywords within each cluster by score
            for label in clusters:
                clusters[label].sort(key=lambda item: item[1], reverse=True)

            # Display in columns or expanders
            num_cluster_cols = min(3, actual_n_clusters) if 'actual_n_clusters' in locals() else 3 # Max 3 columns
            cluster_cols = st.columns(num_cluster_cols)

            cluster_labels_sorted = sorted(clusters.keys())

            for i, label in enumerate(cluster_labels_sorted):
                 col_index = i % num_cluster_cols
                 with cluster_cols[col_index]:
                      st.markdown(f"**Cluster {label}**")
                      df_cluster = pd.DataFrame(clusters[label], columns=["Keyword", "Gap Score"])
                      st.dataframe(df_cluster.style.format({"Gap Score": "{:.3f}"}), height=300)


def paa_extraction_clustering_page():
    st.header("People Also Asked (PAA) Topic Explorer")
    st.markdown("""
        Extracts "People Also Asked" questions from Google for a given query,
        along with related searches and autocomplete suggestions.
        It then clusters these related queries semantically to help identify sub-topics.
    """)

    search_query = st.text_input("Enter Search Query:", "", key="paa_query")
    max_paa_depth = st.slider("Max PAA Click Depth:", min_value=0, max_value=5, value=1, key="paa_depth",
                              help="How many times to click PAA questions to reveal more (0 = no clicks, higher values take longer). Be cautious with high values.")
    num_clusters_paa = st.number_input("Number of Clusters for Dendrogram:", min_value=2, max_value=20, value=5, key="paa_clusters")


    # Define get_paa inside the function to capture max_depth
    def get_paa(query, max_depth=1):
        paa_set = set()
        if not query: return paa_set

        st.write(f"Attempting to fetch PAA for '{query}' (Depth: {max_depth})...")
        driver = None
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            user_agent = get_random_user_agent()
            chrome_options.add_argument(f"user-agent={user_agent}")
            driver = webdriver.Chrome(options=chrome_options)

            driver.get("https://www.google.com/search?q=" + query.replace(" ", "+") + "&hl=en") # Specify English
            time.sleep(random.uniform(2, 4)) # Wait for page load

            # Function to recursively find and click PAA questions
            def extract_paa_recursive(current_depth, click_limit):
                if current_depth > click_limit:
                    return

                paa_box_present = False
                try:
                    # Find PAA container first - more robust selector
                    paa_container = driver.find_element(By.XPATH, "//div[g-accordion-expander]") # Common container pattern
                    paa_box_present = True
                except NoSuchElementException:
                     try: # Fallback selector
                          paa_container = driver.find_element(By.XPATH, "//div[@jsname='N760b']")
                          paa_box_present = True
                     except NoSuchElementException:
                          if current_depth == 0: # Only warn if not found on initial load
                               st.write("PAA box not found on initial load.")
                          return # Stop recursion if container not found

                # Find questions within the container
                # Use CSS selectors that target the question text directly
                # These selectors might need updating if Google changes layout
                questions_xpath = ".//div[@role='heading']//span[contains(@class, 'CSkcDe') or contains(@class, 'related-question')]" # Look for spans inside headings
                try:
                    elements = paa_container.find_elements(By.XPATH, questions_xpath)
                    initial_count = len(paa_set)

                    for el in elements:
                        try:
                            question_text = el.text.strip()
                            if question_text and question_text not in paa_set:
                                print(f"Found PAA: {question_text}") # Debug print
                                paa_set.add(question_text)

                                # Click to potentially reveal more (if depth allows)
                                if current_depth < click_limit:
                                     try:
                                         # Find the clickable parent element (often the g-accordion-expander)
                                         clickable_parent = el.find_element(By.XPATH, "./ancestor::g-accordion-expander[1]")
                                         driver.execute_script("arguments[0].scrollIntoView(true);", clickable_parent) # Scroll into view
                                         time.sleep(random.uniform(0.5, 1.0))
                                         clickable_parent.click()
                                         print(f"Clicked: {question_text}") # Debug print
                                         time.sleep(random.uniform(1.5, 2.5)) # Wait for potential new questions
                                         # Recursive call ONLY if a new question was added by this click cycle
                                         # We check this after the loop finishes this level
                                     except NoSuchElementException:
                                         print(f"Could not find clickable parent for: {question_text}")
                                     except Exception as click_err:
                                         print(f"Error clicking '{question_text}': {click_err}")
                        except Exception as inner_err:
                             print(f"Error processing a PAA element: {inner_err}")

                    # After trying all elements at this level, check if new questions were added
                    if len(paa_set) > initial_count and current_depth < click_limit:
                        print(f"New questions found at depth {current_depth}. Going deeper...")
                        extract_paa_recursive(current_depth + 1, click_limit)
                    else:
                         print(f"No new questions found at depth {current_depth} or depth limit reached.")


                except NoSuchElementException:
                     # This might happen if questions_xpath doesn't match
                     print("Could not find questions using specified XPath within the PAA container.")
                except Exception as e:
                     st.warning(f"Error during PAA extraction: {e}")


            # Initial call to start extraction
            extract_paa_recursive(0, max_paa_depth) # Start at depth 0

            driver.quit()
        except WebDriverException as wd_err:
             st.error(f"WebDriver failed for PAA: {wd_err}")
             if driver: driver.quit()
        except Exception as e:
             st.error(f"An unexpected error occurred during PAA fetch: {e}")
             if driver: driver.quit()

        return paa_set


    if st.button("Analyze PAA & Related Queries", key="paa_analyze"):
        if not search_query:
            st.warning("Please enter a search query.")
            return

        model = initialize_sentence_transformer()
        if not model: return

        paa_questions = set()
        suggestions = []
        related_searches = []

        with st.spinner("Fetching PAA, Autocomplete, and Related Searches..."):
            # 1. Fetch PAA
            paa_questions = get_paa(search_query, max_paa_depth)
            st.write(f"Found {len(paa_questions)} unique PAA questions.")

            # 2. Fetch Autocomplete Suggestions
            try:
                import requests
                autocomplete_url = "http://suggestqueries.google.com/complete/search"
                # Use a common browser user agent for requests
                headers = {'User-Agent': get_random_user_agent()}
                params = {"client": "firefox", "q": search_query, "hl": "en"} # Use firefox client, specify lang
                response = requests.get(autocomplete_url, params=params, headers=headers, timeout=5)
                response.raise_for_status() # Raise error for bad status codes
                suggestions = response.json()[1]
                st.write(f"Found {len(suggestions)} autocomplete suggestions.")
            except requests.exceptions.RequestException as e:
                st.warning(f"Error fetching autocomplete suggestions: {e}")
            except Exception as e:
                 st.warning(f"Unexpected error processing autocomplete: {e}")


            # 3. Fetch Related Searches (using Selenium)
            driver2 = None
            try:
                st.write("Fetching Related Searches...")
                chrome_options = Options()
                chrome_options.add_argument("--headless")
                chrome_options.add_argument("--no-sandbox")
                chrome_options.add_argument("--disable-dev-shm-usage")
                user_agent = get_random_user_agent()
                chrome_options.add_argument(f"user-agent={user_agent}")
                driver2 = webdriver.Chrome(options=chrome_options)
                driver2.get("https://www.google.com/search?q=" + search_query.replace(" ", "+") + "&hl=en")
                time.sleep(random.uniform(2, 4))

                # Try multiple selectors for related searches
                related_selectors = [
                    "div[jsname='yEVEwb'] a", # Common selector
                    "p.nVcaUb > a > span",    # Older selector pattern
                    "div.s75CSd",             # Another potential container
                    ".brs_col a"             # Can sometimes appear here
                ]
                related_elements = []
                for selector in related_selectors:
                     try:
                          elements = driver2.find_elements(By.CSS_SELECTOR, selector)
                          if elements:
                               related_elements = elements
                               print(f"Found related searches using selector: {selector}") # Debug
                               break # Stop searching once found
                     except NoSuchElementException:
                          continue
                     except Exception as find_err:
                          print(f"Error finding related searches with '{selector}': {find_err}") # Debug

                if related_elements:
                    for el in related_elements:
                        text = el.text.strip()
                        if text:
                            related_searches.append(text)
                else:
                     st.write("Could not find related searches using known selectors.")

                related_searches = list(set(related_searches)) # Make unique
                st.write(f"Found {len(related_searches)} unique related searches.")
                driver2.quit()
            except WebDriverException as wd_err:
                st.warning(f"WebDriver error fetching related searches: {wd_err}")
                if driver2: driver2.quit()
            except Exception as e:
                st.warning(f"Error fetching/processing related searches: {e}")
                if driver2: driver2.quit()


        # --- Combine and Analyze ---
        # Combine all sources, ensuring uniqueness and removing the original query if present
        combined_queries = list(paa_questions) + suggestions + related_searches
        combined_queries = list(set(q for q in combined_queries if q.lower() != search_query.lower())) # Unique and remove original query

        if not combined_queries:
            st.warning("No related questions or searches were found.")
            return

        st.info(f"Analyzing {len(combined_queries)} unique related queries...")

        # Calculate Similarity to Original Query (Optional but informative)
        query_embedding = get_embedding(search_query, model)
        question_similarities = []
        for q in combined_queries:
            q_embedding = get_embedding(q, model)
            if q_embedding is not None and query_embedding is not None:
                sim = cosine_similarity(q_embedding.reshape(1, -1), query_embedding.reshape(1, -1))[0][0]
                question_similarities.append((q, float(sim)))

        # Sort by similarity
        question_similarities.sort(key=lambda x: x[1], reverse=True)


        # --- Clustering & Visualization ---
        st.subheader("Semantic Topic Cluster (Dendrogram)")
        if len(combined_queries) >= 2: # Need at least 2 items to cluster
            with st.spinner("Generating embeddings and dendrogram..."):
                try:
                    # Get embeddings for all combined queries
                    embeddings = get_embedding(combined_queries, model)

                    if embeddings is not None and len(embeddings) == len(combined_queries):
                        # Create dendrogram using scipy/plotly
                        # Note: figure_factory is less maintained, consider scipy + plotly.graph_objects directly if issues arise
                        # Ensure embeddings is a 2D numpy array
                        embeddings_np = np.array(embeddings)
                        if embeddings_np.ndim == 1: embeddings_np = embeddings_np.reshape(-1, 1) # Reshape if needed

                        dendro = ff.create_dendrogram(
                            embeddings_np,
                            orientation='left',
                            labels=combined_queries,
                            linkagefun=lambda x: scipy.cluster.hierarchy.linkage(x, method='ward') # Use scipy linkage
                         )
                        dendro.update_layout(width=800, height=max(600, len(combined_queries) * 20), # Dynamic height
                                             margin=dict(l=250)) # Adjust left margin for labels
                        st.plotly_chart(dendro, use_container_width=True)
                    else:
                         st.warning("Could not generate embeddings for all queries needed for clustering.")

                except ImportError:
                     st.error("Scipy is required for dendrogram linkage. Please install it: pip install scipy")
                except Exception as dendro_err:
                     st.error(f"Error creating dendrogram: {dendro_err}")
        else:
            st.info("Not enough related queries (need at least 2) to generate a cluster dendrogram.")


        # --- Display Lists ---
        st.subheader("Related Queries by Similarity to Original")
        if question_similarities:
             df_sim = pd.DataFrame(question_similarities, columns=["Related Query", "Similarity Score"])
             st.dataframe(df_sim.style.format({"Similarity Score": "{:.4f}"}))
        else:
             st.info("Could not calculate similarities.")

        with st.expander("Show All Unique Extracted Queries/Searches"):
            if combined_queries:
                for q in sorted(combined_queries): # Sort alphabetically
                    st.write(f"- {q}")
            else:
                st.write("None found.")


# ------------------------------------
# Google Ads Search Term Analyzer
# ------------------------------------
def google_ads_search_term_analyzer_page():
    st.header("Google Ads Search Term N-gram Analyzer")
    st.markdown("""
        Upload your Google Ads search terms report (Excel format, **.xlsx**) to analyze the performance of multi-word phrases (n-grams).
        Helps identify high/low performing term patterns for optimization and negative keyword discovery.
        *Ensure your report includes at least 'Search term', 'Clicks', 'Impressions', 'Cost', and 'Conversions' columns.*
    """)

    uploaded_file = st.file_uploader("Upload Google Ads Search Terms Excel File (.xlsx)", type=["xlsx"], key="gads_upload")

    if uploaded_file is not None:
        try:
            # Read the Excel file, attempting to find the header row automatically
            # Often Google Ads reports have extra header/summary rows
            df = None
            # Try reading starting from different rows until 'Search term' is found in columns
            for skip in range(5): # Check first 5 rows
                try:
                    temp_df = pd.read_excel(uploaded_file, skiprows=skip)
                    # Basic check for required columns (case-insensitive)
                    if any(col.strip().lower() == 'search term' for col in temp_df.columns):
                         df = temp_df
                         st.info(f"Successfully read data skipping {skip} header rows.")
                         break
                except Exception:
                     continue # Try next skip value

            if df is None:
                 st.error("Could not automatically detect the header row or find the 'Search term' column. Please ensure the file format is correct.")
                 return

            # --- Column Renaming and Cleaning ---
            # Standardize column names (lowercase, replace spaces/symbols)
            df.columns = df.columns.str.strip().str.lower().str.replace('.', '', regex=False).str.replace(' ', '_', regex=False).str.replace('/', '_', regex=False)

            # Define potential variations of required column names
            col_mapping = {
                "search_term": ["search_term", "search_query"],
                "clicks": ["clicks", "clics"],
                "impressions": ["impressions", "impr"],
                "cost": ["cost", "spent"],
                "conversions": ["conversions", "conv", "all_conv"] # Add common variations
                # Add other optional columns if needed: "match_type", "added_excluded", "campaign", "ad_group", "conv_rate", "cost_per_conversion", etc.
            }
            required_original_cols = ["search_term", "clicks", "impressions", "cost", "conversions"]
            rename_dict = {}
            found_cols = {}

            for target_col, possible_names in col_mapping.items():
                found = False
                for name in possible_names:
                    if name in df.columns:
                        rename_dict[name] = target_col
                        found_cols[target_col] = name # Store the original name found
                        found = True
                        break
                if not found and target_col in required_original_cols:
                    st.error(f"Required column '{target_col}' (or a variation like {possible_names}) not found.")
                    # Optionally list found columns for debugging: st.write(f"Found columns: {df.columns.tolist()}")
                    return

            df = df.rename(columns=rename_dict)

            # Keep only the renamed columns we care about for this analysis
            cols_to_keep = list(rename_dict.values())
            # Add back any other columns the user might want to see (optional)
            # Example: if 'match_type' was found, keep it
            # if 'match_type' in rename_dict.values(): cols_to_keep.append('match_type')
            df = df[cols_to_keep]


            # --- Data Type Conversion ---
            # Convert numeric columns, coercing errors and filling NaNs
            numeric_cols_ads = ["clicks", "impressions", "cost", "conversions"]
            for col in numeric_cols_ads:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(0).astype(int) # Fill NaN with 0 and convert to int if desired, or float
                else:
                     # This case should be caught by the check above, but as a safeguard:
                     st.warning(f"Numeric column '{col}' missing after renaming.")
                     df[col] = 0 # Create column with zeros if missing


            st.subheader("N-gram Analysis Settings")
            col_set1, col_set2 = st.columns(2)
            with col_set1:
                 extraction_method = st.radio("N-gram Extraction Method:", options=["Contiguous n-grams", "Skip-grams"], index=0, key="gads_extract",
                                              help="Contiguous: 'red running shoes'. Skip-grams: 'red shoes' (skips 'running').")
                 n_value = st.selectbox("Select N (words per phrase):", options=[1, 2, 3, 4], index=1, key="gads_n")
            with col_set2:
                 min_frequency = st.number_input("Minimum N-gram Frequency:", value=2, min_value=1, key="gads_min_freq",
                                                 help="Only analyze n-grams that appear at least this many times across all search terms.")
                 # Add option to exclude specific words/stopwords
                 custom_stopwords_input = st.text_input("Custom Stopwords (comma-separated):", key="gads_stopwords", placeholder="e.g., near,me,buy")
                 use_default_stopwords = st.checkbox("Use default English stopwords", value=True, key="gads_use_default_stop")


            # --- N-gram Extraction ---
            # Prepare stopwords list
            custom_stopwords = set(word.strip().lower() for word in custom_stopwords_input.split(',') if word.strip())
            combined_stopwords = set(custom_stopwords)
            if use_default_stopwords:
                 combined_stopwords.update(stop_words_nltk) # Use nltk stopwords loaded earlier

            lemmatizer = WordNetLemmatizer()

            # Define extraction functions incorporating stopwords and lemmatization
            def preprocess_and_tokenize(text):
                 text = str(text).lower()
                 tokens = word_tokenize(text)
                 # Lemmatize, remove stopwords, non-alphanumeric, and short tokens
                 return [lemmatizer.lemmatize(t) for t in tokens if t.isalnum() and t not in combined_stopwords and len(t) > 1]

            def extract_ngrams(tokens, n):
                 if len(tokens) < n: return []
                 ngrams_list = list(nltk.ngrams(tokens, n))
                 return [" ".join(gram) for gram in ngrams_list]

            def extract_skipgrams(tokens, n):
                 import itertools
                 if len(tokens) < n: return []
                 skipgrams_list = []
                 # Generate combinations of indices, then get tokens at those indices
                 for combo_indices in itertools.combinations(range(len(tokens)), n):
                      skipgram = " ".join(tokens[i] for i in combo_indices)
                      skipgrams_list.append(skipgram)
                 return skipgrams_list

            # Apply extraction
            st.write("Extracting and analyzing n-grams...")
            all_ngrams = []
            search_term_to_ngrams_map = {} # Map original search term to its extracted ngrams

            for term in df["search_term"]:
                 processed_tokens = preprocess_and_tokenize(term)
                 term_ngrams = []
                 if extraction_method == "Contiguous n-grams":
                      term_ngrams = extract_ngrams(processed_tokens, n_value)
                 else: # Skip-grams
                      term_ngrams = extract_skipgrams(processed_tokens, n_value)

                 all_ngrams.extend(term_ngrams)
                 search_term_to_ngrams_map[term] = term_ngrams

            # Filter by minimum frequency
            ngram_counts = Counter(all_ngrams)
            filtered_ngrams = {ngram: count for ngram, count in ngram_counts.items() if count >= min_frequency}

            if not filtered_ngrams:
                 st.warning(f"No n-grams found with N={n_value} and minimum frequency={min_frequency} after preprocessing. Try adjusting settings.")
                 return

            # --- Aggregate Performance by N-gram ---
            st.write("Aggregating performance metrics...")
            ngram_performance = {} # { ngram: {metric: value} }

            for index, row in df.iterrows():
                 search_term_text = row["search_term"]
                 # Use the mapped ngrams for this search term
                 term_ngrams = search_term_to_ngrams_map.get(search_term_text, [])

                 for ngram in term_ngrams:
                      # Aggregate only if the ngram met the frequency threshold
                      if ngram in filtered_ngrams:
                          if ngram not in ngram_performance:
                               # Initialize metrics for this ngram
                               ngram_performance[ngram] = {
                                   "Frequency": filtered_ngrams[ngram], # Store frequency
                                   "Clicks": 0, "Impressions": 0, "Cost": 0, "Conversions": 0
                               }
                          # Sum up performance metrics
                          ngram_performance[ngram]["Clicks"] += row["clicks"]
                          ngram_performance[ngram]["Impressions"] += row["impressions"]
                          ngram_performance[ngram]["Cost"] += row["cost"]
                          ngram_performance[ngram]["Conversions"] += row["conversions"]

            if not ngram_performance:
                 st.warning("Could not aggregate performance. Check n-gram filtering.")
                 return

            # Convert aggregated data to DataFrame
            df_ngram_perf = pd.DataFrame.from_dict(ngram_performance, orient='index')
            df_ngram_perf.index.name = "N-gram"
            df_ngram_perf = df_ngram_perf.reset_index()

            # --- Calculate Derived Metrics ---
            # Handle potential division by zero
            df_ngram_perf["CTR (%)"] = (df_ngram_perf["Clicks"] / df_ngram_perf["Impressions"] * 100).fillna(0)
            df_ngram_perf["Conv Rate (%)"] = (df_ngram_perf["Conversions"] / df_ngram_perf["Clicks"] * 100).fillna(0)
            df_ngram_perf["Cost per Conv"] = (df_ngram_perf["Cost"] / df_ngram_perf["Conversions"]).replace([np.inf, -np.inf], 0).fillna(0)
            df_ngram_perf["Avg CPC"] = (df_ngram_perf["Cost"] / df_ngram_perf["Clicks"]).replace([np.inf, -np.inf], 0).fillna(0)

            # --- Display Results ---
            st.subheader("N-gram Performance Analysis")

            # Sorting options
            sortable_cols = ["N-gram", "Frequency", "Clicks", "Impressions", "Cost", "Conversions", "CTR (%)", "Conv Rate (%)", "Cost per Conv", "Avg CPC"]
            # Ensure only existing columns are offered for sorting
            available_sort_cols = [col for col in sortable_cols if col in df_ngram_perf.columns]
            default_sort = "Conversions" if "Conversions" in available_sort_cols else "Clicks" if "Clicks" in available_sort_cols else available_sort_cols[0]

            sort_column = st.selectbox("Sort table by:", options=available_sort_cols, index=available_sort_cols.index(default_sort), key="gads_sort_col")
            sort_ascending = st.checkbox("Sort Ascending", value=False, key="gads_sort_asc")

            df_ngram_perf_sorted = df_ngram_perf.sort_values(by=sort_column, ascending=sort_ascending, na_position='last')


            # Formatting dictionary
            format_dict_ads = {
                 "Frequency": "{:,.0f}",
                 "Clicks": "{:,.0f}",
                 "Impressions": "{:,.0f}",
                 "Cost": "${:,.2f}",
                 "Conversions": "{:,.1f}", # Allow decimals for conversions
                 "CTR (%)": "{:,.2f}%",
                 "Conv Rate (%)": "{:,.2f}%",
                 "Cost per Conv": "${:,.2f}",
                 "Avg CPC": "${:,.2f}"
            }
            # Filter format_dict to only include columns present in the sorted dataframe
            valid_format_dict = {k: v for k, v in format_dict_ads.items() if k in df_ngram_perf_sorted.columns}

            st.dataframe(df_ngram_perf_sorted.style.format(valid_format_dict), use_container_width=True)

            # Optional: Add a download button for the results
            @st.cache_data # Cache the conversion
            def convert_df_to_csv(df_to_convert):
                 return df_to_convert.to_csv(index=False).encode('utf-8')

            csv_download = convert_df_to_csv(df_ngram_perf_sorted)
            st.download_button(
                 label="Download N-gram Analysis as CSV",
                 data=csv_download,
                 file_name=f"google_ads_ngram_analysis_{n_value}gram.csv",
                 mime='text/csv',
                 key='gads_download'
            )


        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
            import traceback
            st.error("Traceback:")
            st.code(traceback.format_exc()) # Show detailed traceback for debugging


# ------------------------------------
# Google Search Console Analyzer
# ------------------------------------
def google_search_console_analysis_page():
    st.header("Google Search Console Time Comparison Analysis")
    st.markdown("""
        Compare GSC query performance between two periods. Upload two CSV exports (e.g., last 28 days vs. previous period).
        The tool identifies changes in Clicks, Impressions, CTR, and Position, groups queries into topics using LDA,
        and aggregates performance by topic to highlight areas of growth or decline.
        *CSV files must include 'Top queries', 'Clicks', 'Impressions', 'CTR', and 'Position' columns.*
    """)

    st.markdown("### Upload GSC Data (CSV Format)")
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        uploaded_file_before = st.file_uploader("Upload GSC CSV for 'Before' period", type=["csv"], key="gsc_before")
    with col_up2:
        uploaded_file_after = st.file_uploader("Upload GSC CSV for 'After' period", type=["csv"], key="gsc_after")

    if uploaded_file_before is not None and uploaded_file_after is not None:
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Processing uploaded files...")

        try:
            # --- Step 1: Read and Validate CSVs ---
            df_before = pd.read_csv(uploaded_file_before)
            df_after = pd.read_csv(uploaded_file_after)
            progress_bar.progress(10)

            # Standardize column names (similar to Ads Analyzer)
            def standardize_gsc_cols(df):
                df.columns = df.columns.str.strip().str.lower().str.replace('.', '', regex=False).str.replace(' ', '_', regex=False)
                rename_map = {
                    'top_queries': 'query', 'queries': 'query',
                    'clicks': 'clicks',
                    'impressions': 'impressions', 'impr': 'impressions',
                    'ctr': 'ctr',
                    'position': 'position', 'average_position': 'position'
                }
                df = df.rename(columns=rename_map)
                # Keep only standardized columns we need + query
                required = ['query', 'clicks', 'impressions', 'ctr', 'position']
                cols_found = [col for col in required if col in df.columns]
                missing = [col for col in required if col not in df.columns]
                if missing:
                     raise ValueError(f"Missing required columns: {', '.join(missing)}. Found: {', '.join(df.columns)}")
                return df[cols_found]

            df_before = standardize_gsc_cols(df_before)
            df_after = standardize_gsc_cols(df_after)
            progress_bar.progress(15)
            status_text.text("Standardized columns...")

            # --- Step 2: Data Cleaning & Type Conversion ---
            # Handle CTR (string with %) and other numerics
            def clean_gsc_data(df):
                 # Convert CTR string ('5.5%') to float (5.5)
                 if 'ctr' in df.columns:
                      df['ctr'] = df['ctr'].astype(str).str.replace('%', '', regex=False)
                      df['ctr'] = pd.to_numeric(df['ctr'], errors='coerce')

                 # Convert other columns to numeric
                 for col in ['clicks', 'impressions', 'position']:
                      if col in df.columns:
                           df[col] = pd.to_numeric(df[col], errors='coerce')

                 # Fill NaNs - decide strategy (e.g., 0 for clicks/impressions, mean/median for position/ctr?)
                 # Using 0 for counts, and perhaps drop rows with NaN position/CTR or use median?
                 df['clicks'] = df['clicks'].fillna(0).astype(int)
                 df['impressions'] = df['impressions'].fillna(0).astype(int)
                 # For position and CTR, NaN might mean the query didn't exist or had no impressions/clicks
                 # Keep NaNs for now, handle in calculations
                 # df['position'] = df['position'].fillna(df['position'].median()) # Example: fill with median
                 # df['ctr'] = df['ctr'].fillna(df['ctr'].median()) # Example: fill with median
                 return df

            df_before = clean_gsc_data(df_before)
            df_after = clean_gsc_data(df_after)
            progress_bar.progress(25)
            status_text.text("Cleaned data types...")


            # --- Step 3: Dashboard Summary (Before Merge) ---
            st.markdown("## Overall Performance Change Summary")
            total_clicks_before = df_before["clicks"].sum()
            total_clicks_after = df_after["clicks"].sum()
            total_impressions_before = df_before["impressions"].sum()
            total_impressions_after = df_after["impressions"].sum()
            # Calculate weighted average position/CTR
            avg_pos_before = np.average(df_before['position'].dropna(), weights=df_before.loc[df_before['position'].notna(), 'impressions']) if not df_before['position'].dropna().empty else np.nan
            avg_pos_after = np.average(df_after['position'].dropna(), weights=df_after.loc[df_after['position'].notna(), 'impressions']) if not df_after['position'].dropna().empty else np.nan
            avg_ctr_before = np.average(df_before['ctr'].dropna(), weights=df_before.loc[df_before['ctr'].notna(), 'impressions']) if not df_before['ctr'].dropna().empty else np.nan
            avg_ctr_after = np.average(df_after['ctr'].dropna(), weights=df_after.loc[df_after['ctr'].notna(), 'impressions']) if not df_after['ctr'].dropna().empty else np.nan

            def calculate_change(before, after):
                 if pd.isna(before) or pd.isna(after) or before == 0: return np.nan, np.nan
                 delta = after - before
                 delta_pct = (delta / before) * 100
                 return delta, delta_pct

            clicks_delta, clicks_delta_pct = calculate_change(total_clicks_before, total_clicks_after)
            impr_delta, impr_delta_pct = calculate_change(total_impressions_before, total_impressions_after)
            # Position change is inverted (lower is better)
            pos_delta, pos_delta_pct = calculate_change(avg_pos_before, avg_pos_after)
            pos_delta_display = -pos_delta if not pd.isna(pos_delta) else np.nan # Show positive for improvement
            # CTR change
            ctr_delta, ctr_delta_pct = calculate_change(avg_ctr_before, avg_ctr_after)


            cols_summary = st.columns(4)
            with cols_summary[0]:
                 st.metric(label="Total Clicks Change", value=f"{total_clicks_after:,.0f}", delta=f"{clicks_delta:,.0f} ({clicks_delta_pct:+.1f}%)" if not pd.isna(clicks_delta) else "N/A")
            with cols_summary[1]:
                 st.metric(label="Total Impressions Change", value=f"{total_impressions_after:,.0f}", delta=f"{impr_delta:,.0f} ({impr_delta_pct:+.1f}%)" if not pd.isna(impr_delta) else "N/A")
            with cols_summary[2]:
                 st.metric(label="Average Position Change", value=f"{avg_pos_after:.2f}", delta=f"{pos_delta_display:.2f}", delta_color="inverse") # Lower is better
            with cols_summary[3]:
                 st.metric(label="Average CTR Change", value=f"{avg_ctr_after:.2f}%", delta=f"{ctr_delta:.2f}%" if not pd.isna(ctr_delta) else "N/A")
            progress_bar.progress(30)
            status_text.text("Calculated summary metrics...")

            # --- Step 4: Merge Data ---
            # Outer merge to keep queries present in only one period
            merged_df = pd.merge(df_before, df_after, on="query", suffixes=("_before", "_after"), how="outer")
            # Fill NaNs created by merge with 0 for counts, keep NaNs for position/ctr
            count_cols_merged = ['clicks_before', 'clicks_after', 'impressions_before', 'impressions_after']
            for col in count_cols_merged:
                 if col in merged_df.columns:
                      merged_df[col] = merged_df[col].fillna(0).astype(int)

            progress_bar.progress(35)
            status_text.text("Merged dataframes...")

            # --- Step 5: Calculate Changes per Query ---
            merged_df["Clicks_Change"] = merged_df["clicks_after"] - merged_df["clicks_before"]
            merged_df["Impressions_Change"] = merged_df["impressions_after"] - merged_df["impressions_before"]
            # Position change (lower is better, so change = before - after)
            merged_df["Position_Change"] = merged_df["position_before"] - merged_df["position_after"] # Positive change means improvement
            merged_df["CTR_Change"] = merged_df["ctr_after"] - merged_df["ctr_before"]

            # Calculate Percentage Changes (handle division by zero/NaN)
            merged_df["Clicks_Change_pct"] = merged_df.apply(lambda row: (row["Clicks_Change"] / row["clicks_before"] * 100) if row["clicks_before"] else np.nan, axis=1)
            merged_df["Impressions_Change_pct"] = merged_df.apply(lambda row: (row["Impressions_Change"] / row["impressions_before"] * 100) if row["impressions_before"] else np.nan, axis=1)
            merged_df["Position_Change_pct"] = merged_df.apply(lambda row: (row["Position_Change"] / row["position_before"] * 100) if row["position_before"] else np.nan, axis=1)
            merged_df["CTR_Change_pct"] = merged_df.apply(lambda row: (row["CTR_Change"] / row["ctr_before"] * 100) if row["ctr_before"] else np.nan, axis=1)
            progress_bar.progress(45)
            status_text.text("Calculated changes per query...")


            # --- Step 6: Topic Classification using LDA ---
            st.markdown("---")
            st.subheader("Topic Modeling of Search Queries (LDA)")
            n_topics_gsc = st.slider("Select number of topics:", min_value=3, max_value=30, value=10, key="lda_topics_gsc",
                                      help="How many distinct topic groups to identify among the search queries.")
            min_queries_per_topic = st.number_input("Minimum queries required per topic (for display):", min_value=1, value=3, key="min_q_topic")


            queries = merged_df["query"].astype(str).tolist() # Ensure queries are strings
            with st.spinner(f"Performing LDA Topic Modeling ({n_topics_gsc} topics)..."):
                try:
                    # Use CountVectorizer for LDA
                    vectorizer_queries_lda = CountVectorizer(stop_words="english", max_df=0.9, min_df=3) # Adjust min/max df
                    query_matrix_lda = vectorizer_queries_lda.fit_transform(queries)
                    feature_names_queries_lda = vectorizer_queries_lda.get_feature_names_out()

                    if query_matrix_lda.shape[1] == 0: # Check if vocabulary is empty
                         st.warning("LDA vocabulary is empty after vectorization. Cannot perform topic modeling. Check query data or vectorizer settings.")
                         merged_df["Topic"] = "N/A" # Assign default topic
                    else:
                         lda_model = LatentDirichletAllocation(n_components=n_topics_gsc, random_state=42, n_jobs=-1)
                         lda_topic_distribution = lda_model.fit_transform(query_matrix_lda)
                         # Assign the topic with the highest probability
                         merged_df["Topic_Num"] = lda_topic_distribution.argmax(axis=1) + 1 # 1-based topic number

                         # Generate descriptive labels for topics
                         topic_labels_map = {}
                         for topic_idx in range(n_topics_gsc):
                              topic_queries = merged_df[merged_df["Topic_Num"] == (topic_idx + 1)]["query"].tolist()
                              topic_labels_map[topic_idx + 1] = generate_topic_label(topic_queries) # Use helper function

                         # Apply labels
                         merged_df["Topic"] = merged_df["Topic_Num"].map(topic_labels_map)

                         # Display top keywords per topic in expander
                         with st.expander("Show Top Keywords per Identified Topic"):
                              for topic_idx, topic_weights in enumerate(lda_model.components_):
                                   topic_num = topic_idx + 1
                                   label = topic_labels_map.get(topic_num, f"Topic {topic_num}")
                                   top_keyword_indices = topic_weights.argsort()[-10:][::-1]
                                   keywords = [feature_names_queries_lda[i] for i in top_keyword_indices]
                                   st.write(f"**{label} (Topic {topic_num}):** {', '.join(keywords)}")
                except Exception as lda_e:
                    st.error(f"LDA Topic Modeling failed: {lda_e}")
                    merged_df["Topic"] = "Error" # Assign error topic

            progress_bar.progress(60)
            status_text.text("Assigned queries to topics...")


            # --- Step 7: Display Merged Data Table (Optional) ---
            with st.expander("Show Detailed Merged Data Table with Topics"):
                 # Select and reorder columns for display
                 display_cols_merged = [
                     "query", "Topic",
                     "clicks_before", "clicks_after", "Clicks_Change", "Clicks_Change_pct",
                     "impressions_before", "impressions_after", "Impressions_Change", "Impressions_Change_pct",
                     "ctr_before", "ctr_after", "CTR_Change", "CTR_Change_pct",
                     "position_before", "position_after", "Position_Change", "Position_Change_pct"
                 ]
                 # Filter out columns that might not exist if input was missing them
                 display_cols_merged = [col for col in display_cols_merged if col in merged_df.columns]

                 # Define formatting for the merged table
                 format_dict_merged_gsc = {
                     "clicks_before": "{:,.0f}", "clicks_after": "{:,.0f}", "Clicks_Change": "{:,.0f}", "Clicks_Change_pct": "{:+.1f}%",
                     "impressions_before": "{:,.0f}", "impressions_after": "{:,.0f}", "Impressions_Change": "{:,.0f}", "Impressions_Change_pct": "{:+.1f}%",
                     "ctr_before": "{:.2f}%", "ctr_after": "{:.2f}%", "CTR_Change": "{:+.2f}%", "CTR_Change_pct": "{:+.1f}%",
                     "position_before": "{:.1f}", "position_after": "{:.1f}", "Position_Change": "{:+.1f}", "Position_Change_pct": "{:+.1f}%",
                 }
                 valid_format_merged = {k: v for k, v in format_dict_merged_gsc.items() if k in display_cols_merged}

                 st.dataframe(merged_df[display_cols_merged].style.format(valid_format_merged, na_rep="N/A"))


            # --- Step 8: Aggregated Metrics by Topic ---
            st.markdown("---")
            st.subheader("Aggregated Performance Change by Topic")
            status_text.text("Aggregating performance by topic...")

            # Define aggregation logic
            # Sum clicks/impressions, calculate weighted averages for position/ctr
            agg_functions = {
                 'clicks_before': 'sum', 'clicks_after': 'sum',
                 'impressions_before': 'sum', 'impressions_after': 'sum',
                 # We need weighted averages for position and CTR based on impressions
                 # This requires a custom aggregation function or post-aggregation calculation
            }
            # Initial aggregation for sums
            aggregated = merged_df.groupby("Topic").agg(agg_functions).reset_index()
            aggregated['Query_Count'] = merged_df.groupby("Topic")['query'].count().values # Add query count per topic

            # Custom function for weighted average
            def weighted_avg(df_group, value_col, weight_col):
                try:
                     notna = df_group[[value_col, weight_col]].dropna()
                     if notna.empty or notna[weight_col].sum() == 0: return np.nan
                     return np.average(notna[value_col], weights=notna[weight_col])
                except ZeroDivisionError:
                     return np.nan
                except Exception: # Catch any other errors during calc
                     return np.nan

            # Calculate weighted averages post-aggregation
            avg_pos_before_topic = merged_df.groupby('Topic').apply(weighted_avg, 'position_before', 'impressions_before')
            avg_pos_after_topic = merged_df.groupby('Topic').apply(weighted_avg, 'position_after', 'impressions_after')
            avg_ctr_before_topic = merged_df.groupby('Topic').apply(weighted_avg, 'ctr_before', 'impressions_before')
            avg_ctr_after_topic = merged_df.groupby('Topic').apply(weighted_avg, 'ctr_after', 'impressions_after')

            # Merge averages back into aggregated df
            aggregated = pd.merge(aggregated, avg_pos_before_topic.rename('Position_Before_Avg'), left_on='Topic', right_index=True, how='left')
            aggregated = pd.merge(aggregated, avg_pos_after_topic.rename('Position_After_Avg'), left_on='Topic', right_index=True, how='left')
            aggregated = pd.merge(aggregated, avg_ctr_before_topic.rename('CTR_Before_Avg'), left_on='Topic', right_index=True, how='left')
            aggregated = pd.merge(aggregated, avg_ctr_after_topic.rename('CTR_After_Avg'), left_on='Topic', right_index=True, how='left')

            # Calculate aggregated changes
            aggregated["Clicks_Change"] = aggregated["clicks_after"] - aggregated["clicks_before"]
            aggregated["Impressions_Change"] = aggregated["impressions_after"] - aggregated["impressions_before"]
            aggregated["Position_Change_Avg"] = aggregated["Position_Before_Avg"] - aggregated["Position_After_Avg"] # Positive = improvement
            aggregated["CTR_Change_Avg"] = aggregated["CTR_After_Avg"] - aggregated["CTR_Before_Avg"]

            # Calculate aggregated percentage changes
            aggregated["Clicks_Change_pct"] = aggregated.apply(lambda row: (row["Clicks_Change"] / row["clicks_before"] * 100) if row["clicks_before"] else np.nan, axis=1)
            aggregated["Impressions_Change_pct"] = aggregated.apply(lambda row: (row["Impressions_Change"] / row["impressions_before"] * 100) if row["impressions_before"] else np.nan, axis=1)
            aggregated["Position_Change_Avg_pct"] = aggregated.apply(lambda row: (row["Position_Change_Avg"] / row["Position_Before_Avg"] * 100) if row["Position_Before_Avg"] else np.nan, axis=1)
            aggregated["CTR_Change_Avg_pct"] = aggregated.apply(lambda row: (row["CTR_Change_Avg"] / row["CTR_Before_Avg"] * 100) if row["CTR_Before_Avg"] else np.nan, axis=1)

            progress_bar.progress(75)
            status_text.text("Calculated aggregated changes by topic...")

            # Filter topics with minimum queries
            aggregated_filtered = aggregated[aggregated['Query_Count'] >= min_queries_per_topic].copy()

            # --- Display Aggregated Table ---
            # Select and reorder columns for the aggregated display
            display_cols_agg = [
                 "Topic", "Query_Count",
                 "clicks_before", "clicks_after", "Clicks_Change", "Clicks_Change_pct",
                 "impressions_before", "impressions_after", "Impressions_Change", "Impressions_Change_pct",
                 "CTR_Before_Avg", "CTR_After_Avg", "CTR_Change_Avg", "CTR_Change_Avg_pct",
                 "Position_Before_Avg", "Position_After_Avg", "Position_Change_Avg", "Position_Change_Avg_pct"
            ]
            display_cols_agg = [col for col in display_cols_agg if col in aggregated_filtered.columns] # Ensure cols exist

            # Define formatting for the aggregated table
            format_dict_agg_gsc = {
                 "Query_Count": "{:,.0f}",
                 "clicks_before": "{:,.0f}", "clicks_after": "{:,.0f}", "Clicks_Change": "{:,.0f}", "Clicks_Change_pct": "{:+.1f}%",
                 "impressions_before": "{:,.0f}", "impressions_after": "{:,.0f}", "Impressions_Change": "{:,.0f}", "Impressions_Change_pct": "{:+.1f}%",
                 "CTR_Before_Avg": "{:.2f}%", "CTR_After_Avg": "{:.2f}%", "CTR_Change_Avg": "{:+.2f}%", "CTR_Change_Avg_pct": "{:+.1f}%",
                 "Position_Before_Avg": "{:.1f}", "Position_After_Avg": "{:.1f}", "Position_Change_Avg": "{:+.1f}", "Position_Change_Avg_pct": "{:+.1f}%",
             }
            valid_format_agg = {k: v for k, v in format_dict_agg_gsc.items() if k in display_cols_agg}

            # Add sorting option for the aggregated table
            agg_sort_options = ["Topic", "Query_Count", "Clicks_Change_pct", "Impressions_Change_pct", "Position_Change_Avg", "CTR_Change_Avg_pct"]
            agg_sort_options = [opt for opt in agg_sort_options if opt in aggregated_filtered.columns] # Filter available sort options
            agg_sort_col = st.selectbox("Sort aggregated table by:", options=agg_sort_options, index=2, key="gsc_agg_sort") # Default sort by Clicks % Change
            agg_sort_asc = st.checkbox("Sort Ascending", key="gsc_agg_asc")

            aggregated_sorted = aggregated_filtered.sort_values(by=agg_sort_col, ascending=agg_sort_asc, na_position='last')

            st.dataframe(aggregated_sorted[display_cols_agg].style.format(valid_format_agg, na_rep="N/A"))
            progress_bar.progress(85)


            # --- Step 9: Visualization - Bar Chart of YOY % Change by Topic ---
            st.markdown("---")
            st.subheader("Visualization: Performance % Change by Topic")
            status_text.text("Generating visualizations...")

            # Prepare data for plotting (melt the percentage change columns)
            plot_data_pct = aggregated_sorted.melt(
                 id_vars="Topic",
                 value_vars=["Clicks_Change_pct", "Impressions_Change_pct", "Position_Change_Avg_pct", "CTR_Change_Avg_pct"],
                 var_name="Metric (% Change)",
                 value_name="Percentage Change"
            )
            # Clean up metric names for display
            plot_data_pct["Metric (% Change)"] = plot_data_pct["Metric (% Change)"].replace({
                 "Clicks_Change_pct": "Clicks %",
                 "Impressions_Change_pct": "Impressions %",
                 "Position_Change_Avg_pct": "Position %", # Remember positive is improvement
                 "CTR_Change_Avg_pct": "CTR %"
            })
            # Drop rows with NaN change values for cleaner plotting
            plot_data_pct.dropna(subset=["Percentage Change"], inplace=True)

            # Allow user to select metrics to plot
            available_metrics = plot_data_pct["Metric (% Change)"].unique().tolist()
            selected_metrics = st.multiselect("Select metrics to display in chart:", options=available_metrics, default=available_metrics, key="gsc_select_metrics")

            if selected_metrics:
                 plot_data_filtered = plot_data_pct[plot_data_pct["Metric (% Change)"].isin(selected_metrics)]

                 if not plot_data_filtered.empty:
                      fig_gsc = px.bar(
                           plot_data_filtered,
                           x="Topic",
                           y="Percentage Change",
                           color="Metric (% Change)",
                           barmode="group",
                           title="Percentage Change by Topic for Selected Metrics",
                           labels={"Percentage Change": "% Change (vs. Before Period)", "Topic": "Query Topic"},
                           height=max(500, len(aggregated_filtered['Topic'].unique()) * 50) # Dynamic height
                      )
                      fig_gsc.update_layout(yaxis_ticksuffix="%") # Add % suffix to y-axis
                      st.plotly_chart(fig_gsc, use_container_width=True)
                 else:
                      st.info("No data available for the selected metrics to plot.")
            else:
                 st.info("Please select at least one metric to display in the chart.")

            progress_bar.progress(100)
            status_text.text("Analysis Complete.")

        except ValueError as ve:
             st.error(f"Data Validation Error: {ve}")
             status_text.text("Analysis failed due to data validation issues.")
        except Exception as e:
             st.error(f"An unexpected error occurred during GSC analysis: {e}")
             import traceback
             st.error("Traceback:")
             st.code(traceback.format_exc())
             status_text.text("Analysis failed.")
        finally:
             # Clear progress bar and status text
             progress_bar.empty()
             status_text.empty()
    else:
        st.info("Please upload both GSC CSV files to start the analysis.")


# ------------------------------------
# Vector Embeddings Scatterplot (Site Focus Visualizer)
# ------------------------------------
# Cache the SentenceTransformer model so it loads only once (already cached via initialize_sentence_transformer)

@st.cache_data # Cache data loading
def load_screamingfrog_data(file):
    """Loads Screaming Frog crawl data (CSV) expecting 'Address' and 'H1-1'."""
    try:
        # Try reading with common encodings
        try:
            df = pd.read_csv(file, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file, encoding='latin1')

        # Standardize columns: Find 'Address' or 'URL', and 'H1-1' or 'H1' or 'Title 1'
        addr_col, content_col = None, None

        for col in df.columns:
            col_lower = col.strip().lower()
            if col_lower in ['address', 'url']:
                addr_col = col
            elif col_lower in ['h1-1', 'h1', 'title 1', 'title']: # Prioritize H1, fallback to Title
                if content_col is None or content_col.lower() not in ['h1-1', 'h1']: # Don't overwrite H1 with Title
                    content_col = col

        if not addr_col:
            raise ValueError("CSV must contain an 'Address' or 'URL' column.")
        if not content_col:
            # If no H1 or Title found, maybe use Meta Description? Or fail?
            # Let's try Meta Description as a last resort
            for col in df.columns:
                 if col.strip().lower() in ['meta description 1', 'meta description']:
                      content_col = col
                      st.warning("Using 'Meta Description' as content source as 'H1' or 'Title' was not found.")
                      break
            if not content_col:
                 raise ValueError("CSV must contain a content column like 'H1-1', 'H1', 'Title 1', or 'Meta Description 1'.")

        # Select and rename columns
        df_selected = df[[addr_col, content_col]].copy()
        df_selected.rename(columns={addr_col: 'URL', content_col: 'Content'}, inplace=True)

        # Handle missing content - fill with URL path or drop? Fill for now.
        df_selected['Content'] = df_selected['Content'].fillna(df_selected['URL'].apply(lambda x: urlparse(x).path.replace('/', ' ').strip()))
        df_selected.dropna(subset=['URL', 'Content'], inplace=True) # Drop rows where URL is missing

        return df_selected

    except Exception as e:
         raise ValueError(f"Error reading or processing CSV: {e}")


@st.cache_data # Cache vectorization
def vectorize_pages(contents: List[str], model):
    """Converts page content (e.g., H1s) into vector embeddings."""
    if not model:
        raise ValueError("SentenceTransformer model not loaded.")
    if not contents:
        return np.array([]) # Return empty array if no content

    try:
        embeddings = model.encode(contents, show_progress_bar=True, convert_to_numpy=True)
        return embeddings
    except Exception as e:
         st.error(f"Error during embedding generation: {e}")
         return np.array([])


@st.cache_data # Cache dimension reduction
def reduce_dimensions(embeddings, n_components=2, method='UMAP', umap_neighbors=15, umap_min_dist=0.1):
    """Reduces vector dimensionality using UMAP or PCA."""
    if embeddings.shape[0] < max(n_components, 2): # Need enough samples
        st.warning(f"Need at least {max(n_components, 2)} pages with valid content for dimension reduction. Found {embeddings.shape[0]}. Skipping reduction.")
        return None
    try:
        if method == 'UMAP':
             # Adjust n_neighbors dynamically if fewer samples than default
             actual_neighbors = min(umap_neighbors, embeddings.shape[0] - 1)
             if actual_neighbors < 2: actual_neighbors = 2 # UMAP needs at least 2 neighbors

             reducer = umap.UMAP(n_components=n_components,
                                 n_neighbors=actual_neighbors,
                                 min_dist=umap_min_dist,
                                 random_state=42,
                                 metric='cosine') # Use cosine distance for semantic similarity
             reduced_embeddings = reducer.fit_transform(embeddings)
        else: # PCA
            reducer = PCA(n_components=n_components, random_state=42)
            reduced_embeddings = reducer.fit_transform(embeddings)
        return reduced_embeddings
    except Exception as e:
         st.error(f"Error during {method} dimension reduction: {e}")
         return None


@st.cache_data # Cache clustering results
def cluster_embeddings(embeddings, n_clusters=5):
    """Clusters embeddings using KMeans."""
    if embeddings.shape[0] < n_clusters:
        st.warning(f"Number of pages ({embeddings.shape[0]}) is less than desired clusters ({n_clusters}). Reducing clusters.")
        n_clusters = max(1, embeddings.shape[0]) # At least 1 cluster

    if n_clusters <= 1:
         return np.zeros(embeddings.shape[0], dtype=int) # All in one cluster

    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(embeddings)
        return labels
    except Exception as e:
        st.error(f"Error during KMeans clustering: {e}")
        return np.zeros(embeddings.shape[0], dtype=int) # Return default cluster on error


def plot_embeddings_interactive(embeddings_2d, labels, urls, content):
    """Creates an interactive scatter plot using Plotly Express."""
    if embeddings_2d is None or embeddings_2d.shape[0] != len(urls):
        st.warning("Cannot plot: Mismatch in data dimensions or missing reduced embeddings.")
        return None

    # Create DataFrame for plotting
    df_plot = pd.DataFrame({
        "x": embeddings_2d[:, 0],
        "y": embeddings_2d[:, 1],
        "Cluster": [f"Cluster {label+1}" for label in labels], # 1-based cluster index
        "URL": urls,
        "Content": content # Include the original content used for embedding
    })

    # Create interactive scatter plot
    fig = px.scatter(
        df_plot,
        x="x", y="y",
        color="Cluster",
        hover_name="URL", # Show URL prominently on hover
        hover_data={ # Define data shown in hover box
            "x": False, # Hide coordinates from hover
            "y": False,
            "URL": True,
            "Cluster": True,
            "Content": True # Show the H1/Title content
        },
        title="Interactive Scatterplot of Website Pages by Content Similarity",
        color_discrete_sequence=px.colors.qualitative.Plotly # Use qualitative colors
    )
    fig.update_layout(
        xaxis_title="Semantic Dimension 1",
        yaxis_title="Semantic Dimension 2",
        height=700,
        hoverlabel=dict(bgcolor="white", font_size=12) # Customize hover box
    )
    fig.update_traces(marker=dict(size=8, opacity=0.8)) # Adjust marker style
    return fig


def semantic_clustering_page():
    st.header("Site Content Focus Visualizer")
    st.markdown("""
        Upload a CSV export from Screaming Frog (or similar crawl) to visualize the semantic relationships between your pages based on their content (e.g., H1 tags or Titles).
        The CSV must include a URL column (e.g., 'Address') and a content column (e.g., 'H1-1', 'Title 1').
        Pages closer together on the plot are semantically more similar. Clusters group related pages.
    """)

    uploaded_file = st.file_uploader("Upload Screaming Frog Crawl CSV", type=["csv"], key="sf_csv_upload")

    if uploaded_file is not None:
        try:
            with st.spinner("Loading and processing CSV data..."):
                data = load_screamingfrog_data(uploaded_file)
                st.write(f"Loaded {len(data)} pages with URL and Content.")
        except ValueError as e:
            st.error(f"Error loading data: {e}")
            return
        except Exception as e:
             st.error(f"An unexpected error occurred loading the file: {e}")
             return

        st.dataframe(data.head())

        # Extract URLs and Content for processing
        urls = data['URL'].tolist()
        page_content = data['Content'].astype(str).tolist() # Ensure content is string

        # Load the transformer model (cached)
        model = initialize_sentence_transformer()
        if not model: return # Stop if model failed

        with st.spinner("Generating vector embeddings for page content..."):
            embeddings = vectorize_pages(page_content, model)
            if embeddings.size == 0:
                 st.error("Failed to generate embeddings for the provided content.")
                 return
            st.write(f"Generated embeddings with shape: {embeddings.shape}")

        # --- Clustering Settings ---
        st.sidebar.subheader("Clustering & Visualization")
        n_clusters = st.sidebar.number_input("Select number of clusters:", min_value=2, max_value=50, value=10, step=1, key="semclus_num_clusters")
        dim_reduction_method = st.sidebar.selectbox("Dimension Reduction:", options=["UMAP", "PCA"], index=0, key="semclus_dim_red")

        # UMAP specific settings (optional)
        umap_neighbors = 15
        umap_min_dist = 0.1
        if dim_reduction_method == "UMAP":
             umap_neighbors = st.sidebar.slider("UMAP Neighbors:", min_value=2, max_value=50, value=15, key="semclus_umap_n", help="Controls local vs global structure focus.")
             umap_min_dist = st.sidebar.slider("UMAP Min Distance:", min_value=0.0, max_value=0.99, value=0.1, step=0.05, key="semclus_umap_d", help="Controls how tightly points are packed.")


        with st.spinner(f"Reducing dimensions using {dim_reduction_method}..."):
            reduced_embeddings = reduce_dimensions(embeddings, n_components=2, method=dim_reduction_method, umap_neighbors=umap_neighbors, umap_min_dist=umap_min_dist)
            if reduced_embeddings is None:
                 # Error message handled within reduce_dimensions
                 return

        with st.spinner(f"Clustering embeddings into {n_clusters} groups..."):
            # Cluster using the *original* high-dimensional embeddings for better accuracy
            labels = cluster_embeddings(embeddings, n_clusters=n_clusters)
            # Find the actual number of unique clusters generated
            actual_clusters = len(set(labels))
            st.write(f"Clustering complete. Found {actual_clusters} clusters.")

        # --- Plotting ---
        st.subheader("Interactive Content Cluster Plot")
        if reduced_embeddings is not None:
             fig = plot_embeddings_interactive(reduced_embeddings, labels, urls, page_content)
             if fig:
                  st.plotly_chart(fig, use_container_width=True)
             else:
                  st.warning("Could not generate plot.")
        else:
             st.warning("Plotting skipped due to dimension reduction failure.")


        # --- Display Cluster Details ---
        with st.expander("Show Pages per Cluster"):
             cluster_data = {} # { cluster_num: [ (URL, Content), ... ] }
             for i, url in enumerate(urls):
                  cluster_num = labels[i] + 1 # 1-based index
                  if cluster_num not in cluster_data:
                       cluster_data[cluster_num] = []
                  cluster_data[cluster_num].append({"URL": url, "Content": page_content[i]})

             # Sort clusters by number
             sorted_clusters = sorted(cluster_data.keys())
             for cluster_num in sorted_clusters:
                  st.markdown(f"**Cluster {cluster_num}** ({len(cluster_data[cluster_num])} pages)")
                  df_cluster_pages = pd.DataFrame(cluster_data[cluster_num])
                  st.dataframe(df_cluster_pages, height=200) # Limit display height


# ------------------------------------
# Entity Relationship Graph Generator
# ------------------------------------
# --- Helper Functions (OUTSIDE entity_relationship_graph_page) ---

@st.cache_data(ttl=3600) # Cache extraction for an hour
def extract_entities_and_relationships(sentences: List[str], nlp_spacy, ner_bert) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], Counter]:
    """Extracts entities using BERT and relationships based on co-occurrence."""
    entities_map = {} # {lemma: (original_text, label)} to store unique entities
    relationships = []
    entity_counts = Counter() # Counts occurrences of the *lemma*

    # Define entity types to exclude from graph
    excluded_types = {"DATE", "TIME", "CARDINAL", "ORDINAL", "PERCENT", "MONEY", "QUANTITY", "LANGUAGE"}

    if not sentences or not nlp_spacy or not ner_bert:
        return [], [], Counter()

    st.write(f"Processing {len(sentences)} sentences for entities...")
    combined_text = " ".join(sentences) # Process all text at once for BERT efficiency
    raw_entities_bert = []
    try:
        # BERT NER pipeline expects a single string or list of strings
        # Processing sentence by sentence might be too slow. Process chunks?
        # For simplicity, process the whole text, relationships are sentence-based later.
        raw_entities_bert = ner_bert(combined_text)
    except Exception as ner_err:
        st.warning(f"BERT NER failed during entity extraction: {ner_err}")
        # Fallback to spaCy NER if BERT fails? Or just return empty?
        # For now, return empty:
        return [], [], Counter()

    st.write(f"Found {len(raw_entities_bert)} raw entities. Lemmatizing and filtering...")
    # Process raw BERT entities: lemmatize and filter
    for ent_dict in raw_entities_bert:
        original_text = ent_dict['word'].strip().strip('.,;:!?"\'()[]{}')
        label = ent_dict['entity_group']

        if label not in excluded_types and len(original_text) > 1:
            doc = nlp_spacy(original_text)
            lemma = " ".join([token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and token.lemma_])
            if not lemma: lemma = original_text.lower() # Fallback lemma

            entity_counts[lemma] += 1 # Count occurrence of lemma
            # Store the first encountered original text and label for the lemma
            if lemma not in entities_map:
                 entities_map[lemma] = (original_text, label)

    # Extract relationships based on co-occurrence within original sentences
    st.write("Identifying relationships based on co-occurrence...")
    sentence_progress = st.progress(0)
    total_sentences = len(sentences)
    for i, sentence in enumerate(sentences):
        doc = nlp_spacy(sentence) # Use spaCy for sentence-level entity finding for co-occurrence
        sentence_entity_lemmas = set()
        for ent in doc.ents:
             if ent.label_ not in excluded_types:
                 lemma = " ".join([token.lemma_.lower() for token in ent if not token.is_stop and not token.is_punct and token.lemma_])
                 if not lemma: lemma = ent.text.lower()
                 # Only consider lemmas that were found by the main BERT process
                 if lemma in entities_map:
                      sentence_entity_lemmas.add(lemma)

        # Generate pairs of co-occurring entity lemmas within the sentence
        if len(sentence_entity_lemmas) > 1:
             from itertools import combinations
             for lemma1, lemma2 in combinations(sorted(list(sentence_entity_lemmas)), 2):
                  relationships.append(tuple(sorted((lemma1, lemma2)))) # Store sorted tuple

        if total_sentences > 0 : sentence_progress.progress((i+1)/total_sentences)

    # Final entity list using the stored original text and label
    final_entities = [(details[0], details[1]) for lemma, details in entities_map.items()]

    st.write("Extraction complete.")
    return final_entities, relationships, entity_counts


def create_entity_graph(entities_map: Dict[str, Tuple[str, str]], relationships: List[Tuple[str, str]], entity_counts: Counter) -> nx.Graph:
    """Creates a NetworkX graph from extracted entities and relationships."""
    G = nx.Graph()

    st.write("Building graph...")
    # Add nodes with attributes (using lemma as node ID)
    for lemma, (original_text, label) in entities_map.items():
        G.add_node(lemma, label=original_text, type=label, count=entity_counts.get(lemma, 0))

    # Add edges with weights based on relationship frequency
    relationship_counts = Counter(relationships)
    for (lemma1, lemma2), count in relationship_counts.items():
         # Ensure both nodes exist in the graph before adding edge
         if G.has_node(lemma1) and G.has_node(lemma2):
              G.add_edge(lemma1, lemma2, weight=count)

    st.write(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G


def visualize_graph(G: nx.Graph, source_identifier: str):
    """Visualizes the entity relationship graph using Matplotlib."""
    if not G or G.number_of_nodes() == 0:
        st.warning("Graph is empty, cannot visualize.")
        return

    st.write("Calculating graph layout (this can take time for large graphs)...")
    plt.figure(figsize=(18, 14)) # Increased size
    # Experiment with different layouts: spring_layout, kamada_kawai_layout, spectral_layout
    try:
         # k controls distance between nodes, iterations for convergence
         pos = nx.spring_layout(G, seed=42, k=0.8 / np.sqrt(G.number_of_nodes()), iterations=50)
         # pos = nx.kamada_kawai_layout(G) # Alternative layout
    except Exception as layout_err:
         st.warning(f"Graph layout calculation failed ({layout_err}). Using random layout.")
         pos = nx.random_layout(G, seed=42)


    st.write("Drawing graph nodes and edges...")
    # Node sizing based on count (log scale can help with large differences)
    min_size, max_size = 50, 2000
    counts = np.array([G.nodes[node]['count'] for node in G.nodes()])
    # Use logarithmic scaling if counts vary widely, otherwise linear
    if counts.max() / (counts.min() + 1e-6) > 100: # If max is >100x min
        node_sizes = np.log1p(counts) # log(1 + count)
        node_sizes = min_size + (node_sizes - node_sizes.min()) / (node_sizes.max() - node_sizes.min() + 1e-6) * (max_size - min_size)
    else: # Linear scaling
         node_sizes = min_size + (counts - counts.min()) / (counts.max() - counts.min() + 1e-6) * (max_size - min_size)
    node_sizes = node_sizes.fillna(min_size) # Handle potential NaNs


    # Node coloring based on entity type (consistent colors)
    type_colors = {
        'ORG': 'skyblue', 'PER': 'lightcoral', 'GPE': 'lightgreen', 'LOC': 'khaki',
        'PRODUCT': 'plum', 'EVENT': 'lightsalmon', 'WORK_OF_ART': 'wheat',
        'FAC': 'aquamarine', 'LAW': 'silver', 'NORP': 'gold', 'MISC': 'lightgrey'
        # Add more types and colors as needed based on BERT model output
    }
    default_color = 'grey'
    node_colors = [type_colors.get(G.nodes[node]['type'], default_color) for node in G.nodes()]

    # Edge widths based on weight (log scale?)
    weights = np.array([data['weight'] for _, _, data in G.edges(data=True)])
    min_width, max_width = 0.5, 5.0
    if weights.size > 0: # Check if there are any edges
        edge_widths = np.log1p(weights)
        edge_widths = min_width + (edge_widths - edge_widths.min()) / (edge_widths.max() - edge_widths.min() + 1e-6) * (max_width - min_width)
        edge_widths = edge_widths.fillna(min_width)
    else:
        edge_widths = []


    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes.tolist(), alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=edge_widths.tolist() if isinstance(edge_widths, np.ndarray) else edge_widths, alpha=0.3, edge_color='gray')

    # Labels - only draw labels for larger nodes to avoid clutter?
    labels_to_draw = {}
    median_size = np.median(node_sizes) if node_sizes.size > 0 else min_size
    threshold_size = median_size * 1.2 # Only label nodes > 1.2x median size

    for i, node in enumerate(G.nodes()):
         if node_sizes[i] > threshold_size:
              labels_to_draw[node] = G.nodes[node]['label'] # Use original text for label

    nx.draw_networkx_labels(G, pos, labels=labels_to_draw, font_size=9, font_family="sans-serif", font_weight='bold')

    plt.title(f"Entity Relationships for: {source_identifier}", fontsize=16)
    plt.axis("off")
    st.pyplot(plt)
    plt.close() # Close the figure to free memory

# --- Main Page Function ---
def entity_relationship_graph_page():
    st.header("Entity Relationship Graph Generator")
    st.markdown("""
    Visualize the relationships between named entities found on a webpage.
    Entities are nodes, sized by frequency. Edges represent co-occurrence within the same sentence, weighted by frequency.
    Uses BERT for entity detection and spaCy for processing.
    """)
    # Input: Single URL or Text Area
    source_option_erg = st.radio("Select Input Source:", options=["URL", "Pasted Text"], key="erg_source", horizontal=True)

    source_input = ""
    source_identifier = "" # For graph title

    if source_option_erg == "URL":
        source_input = st.text_input("Enter a website URL:", key="erg_url")
        source_identifier = source_input if source_input else "Entered URL"
    else:
        source_input = st.text_area("Paste text content here:", key="erg_text", height=250)
        source_identifier = "Pasted Text"


    if st.button("Generate Entity Graph", key="erg_button"):
        if not source_input:
            st.warning("Please provide a URL or paste text.")
            return

        text_content = ""
        with st.spinner(f"Processing source: {source_identifier}..."):
            if source_option_erg == "URL":
                text_content = extract_text_from_url(source_input) # Use standard text extraction
            else:
                text_content = source_input

            if not text_content:
                st.error(f"Could not retrieve or process content from the source.")
                return

            # Split into sentences
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+', text_content)
            sentences = [s.strip() for s in sentences if s.strip() and len(s.split()) > 3] # Basic sentence filter

            if not sentences:
                st.warning("No valid sentences found in the content to analyze for relationships.")
                return

        with st.spinner("Extracting entities and relationships..."):
            nlp_spacy = load_spacy_model()
            ner_bert_pipe = load_bert_ner_pipeline()

            if not nlp_spacy or not ner_bert_pipe:
                st.error("Required NLP models failed to load.")
                return

            # Extract entities (map), relationships (pairs), counts (lemma frequency)
            entities_map, relationships, entity_counts = extract_entities_and_relationships(sentences, nlp_spacy, ner_bert_pipe)

            if not entities_map:
                st.warning("No relevant entities found in the content after filtering.")
                return

            # Create the graph
            graph = create_entity_graph(entities_map, relationships, entity_counts)

        with st.spinner("Visualizing graph..."):
            visualize_graph(graph, source_identifier)


# ------------------------------------
# SEMRush Organic Pages Sub-Directories
# ------------------------------------
def semrush_organic_pages_by_subdirectory_page():
    st.header("SEMRush Organic Pages - Top Subdirectory Analysis")
    st.markdown("""
    Upload your SEMRush Organic Pages report (Excel format) to aggregate key metrics (like Traffic, Keywords) by the **top-level subdirectory**.
    Helps understand which main sections of a site drive organic performance.
    *File must contain a 'URL' column and numeric columns to aggregate.*
    """)

    uploaded_file = st.file_uploader(
        "Upload SEMRush Organic Pages Excel file (.xlsx)",
        type=["xlsx"],
        key="semrush_subdir_file"
    )

    if uploaded_file is not None:
        try:
            # Read the Excel file
            df = pd.read_excel(uploaded_file)
            st.write(f"File read successfully. Found {len(df)} rows and {len(df.columns)} columns.")
        except Exception as e:
            st.error(f"Error reading the Excel file: {e}")
            return

        # --- Column Handling ---
        # Find URL column (case-insensitive)
        url_col = None
        for col in df.columns:
            if col.strip().lower() == 'url':
                url_col = col
                break
        if not url_col:
            st.error("No 'URL' column found in the file.")
            return

        # Identify potential numeric columns to aggregate
        numeric_cols = []
        potential_metrics = ['traffic', 'keywords', 'traffic cost', 'position', 'volume'] # Add more as needed
        df_cols_lower = [c.strip().lower() for c in df.columns]

        for metric in potential_metrics:
             if metric in df_cols_lower:
                  original_col_name = df.columns[df_cols_lower.index(metric)]
                  # Attempt conversion, keep if successful
                  try:
                       df[original_col_name] = pd.to_numeric(df[original_col_name], errors='coerce')
                       # Only add if it contains *some* numeric data after coercion
                       if df[original_col_name].notna().any():
                            numeric_cols.append(original_col_name)
                  except Exception:
                       st.warning(f"Could not convert column '{original_col_name}' to numeric. It will not be aggregated.")


        if not numeric_cols:
            st.warning("No standard numeric metric columns (like 'Traffic', 'Keywords') found or convertible to numeric. Cannot perform aggregation.")
            st.dataframe(df.head()) # Show raw data preview
            return

        st.info(f"Identified numeric columns for aggregation: {', '.join(numeric_cols)}")

        # --- Extract Top Subdirectory ---
        def get_top_subdirectory(url):
            try:
                parsed = urlparse(str(url)) # Ensure URL is string
                # Remove leading/trailing slashes and split path
                path_segments = [seg for seg in parsed.path.strip('/').split('/') if seg]
                # Return first segment or 'Homepage/' if no path
                return '/' + path_segments[0] if path_segments else "Homepage/"
            except Exception:
                return "Invalid URL Path" # Handle potential parsing errors

        df["Subdirectory"] = df[url_col].apply(get_top_subdirectory)

        # --- Display Preview with Subdirectory ---
        st.markdown("### Data Preview with Subdirectory")
        preview_cols = [url_col, "Subdirectory"] + numeric_cols
        st.dataframe(df[preview_cols].head())

        # --- Aggregation ---
        st.markdown("### Aggregated Metrics by Top-Level Subdirectory")
        try:
            # Build aggregation dictionary (sum numeric cols)
            agg_dict = {col: "sum" for col in numeric_cols}
            # Add count of URLs per subdirectory
            agg_dict[url_col] = 'count'

            # Group by Subdirectory and aggregate
            subdir_agg = df.groupby("Subdirectory").agg(agg_dict).reset_index()

            # Rename the URL count column
            subdir_agg.rename(columns={url_col: "URL Count"}, inplace=True)

            # Sort by a primary metric (e.g., Traffic or URL Count)
            sort_col = None
            if 'Traffic' in numeric_cols: sort_col = 'Traffic'
            elif 'Keywords' in numeric_cols: sort_col = 'Keywords'
            else: sort_col = 'URL Count'

            if sort_col in subdir_agg.columns:
                 subdir_agg = subdir_agg.sort_values(by=sort_col, ascending=False)

            # Define formatting for display
            format_dict_semrush = {"URL Count": "{:,.0f}"}
            for col in numeric_cols:
                 # Simple default formatting, can be customized
                 format_dict_semrush[col] = "{:,.0f}" if 'position' not in col.lower() else "{:.1f}"

            st.dataframe(subdir_agg.style.format(format_dict_semrush, na_rep="N/A"))

        except Exception as agg_e:
            st.error(f"Error during aggregation: {agg_e}")
            return

        # --- Visualization ---
        st.markdown("### Visualization")
        # Allow user to select metric to plot
        plot_metric = st.selectbox("Select metric to visualize:", options=numeric_cols + ["URL Count"], index=numeric_cols.index('Traffic') if 'Traffic' in numeric_cols else 0)

        if plot_metric in subdir_agg.columns:
             # Sort for plotting consistency
             subdir_agg_plot = subdir_agg.sort_values(by=plot_metric, ascending=False)
             fig = px.bar(
                 subdir_agg_plot,
                 x="Subdirectory",
                 y=plot_metric,
                 title=f"{plot_metric} by Top-Level Subdirectory",
                 labels={"Subdirectory": "Subdirectory", plot_metric: plot_metric},
                 height=500
             )
             fig.update_layout(xaxis={'categoryorder':'total descending'})
             st.plotly_chart(fig, use_container_width=True)
        else:
             st.info(f"Column '{plot_metric}' not available for plotting.")

    else:
        st.info("Please upload a SEMRush Organic Pages Excel file to begin.")


# ------------------------------------
# SEMRush Organic Pages Hierarchical Sub-Directories (No Leaf Nodes)
# ------------------------------------
def semrush_hierarchical_subdirectories_minimal_no_leaf_with_intent_filter():
    st.header("SEMRush Hierarchical Subdirectory Aggregation (No Leaf Nodes)")
    st.markdown("""
    Analyzes SEMRush data by expanding each URL into its full subdirectory path (e.g., `/blog/`, `/blog/topic/`, `/blog/topic/post`).
    It then **removes leaf nodes** (the final posts/pages, keeping only the directory levels) and aggregates metrics (Keywords, Traffic, and optional Intent Traffic) at each remaining hierarchical level.
    Useful for understanding performance of entire site sections and sub-sections.
    """)
    st.markdown("Required columns: **URL**, **Number of Keywords**, **Traffic**. Optional: User intent traffic columns (e.g., 'Traffic with commercial intents in top 20').")


    uploaded_file = st.file_uploader(
        "Upload SEMRush Organic Pages Excel file (.xlsx)",
        type=["xlsx"],
        key="semrush_hierarchical_file"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error reading Excel file: {e}")
            return

        # --- Column Handling & Validation ---
        # Standardize column names (lowercase, underscore)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_', regex=False).str.replace('/', '_', regex=False).str.replace('[^A-Za-z0-9_]+', '', regex=True) # Remove special chars except underscore

        # Define required and optional columns with standardized names
        required_std_cols = {"url", "number_of_keywords", "traffic"}
        optional_intent_std_cols = {
            "traffic_with_commercial_intents_in_top_20",
            "traffic_with_informational_intents_in_top_20",
            "traffic_with_navigational_intents_in_top_20",
            "traffic_with_transactional_intents_in_top_20",
            "traffic_with_unknown_intents_in_top_20"
        }
        # Find which standardized columns exist in the DataFrame
        found_std_cols = set(df.columns)
        missing_required = required_std_cols - found_std_cols
        if missing_required:
            st.error(f"Missing required standardized columns: {', '.join(missing_required)}. Found columns: {', '.join(sorted(list(found_std_cols)))}")
            return

        available_intent_cols = list(optional_intent_std_cols.intersection(found_std_cols))
        cols_to_keep = list(required_std_cols.union(available_intent_cols))
        df = df[cols_to_keep]


        # --- Data Type Conversion ---
        numeric_cols = list(required_std_cols - {"url"}) + available_intent_cols
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)


        # --- Hierarchical Path Expansion ---
        st.write("Expanding URLs into hierarchical paths...")
        def get_all_subdirectory_levels(url):
            try:
                parsed = urlparse(str(url))
                segments = [seg for seg in parsed.path.strip("/").split("/") if seg]
                # Always include root
                paths = ["/"]
                # Generate paths like /seg1, /seg1/seg2, ...
                for i in range(len(segments)):
                    path = "/" + "/".join(segments[:i+1]) + "/" # Ensure trailing slash for directories
                    paths.append(path)
                return list(set(paths)) # Return unique paths (handles potential duplicates)
            except Exception:
                 return ["/"] # Default to root on error

        # Explode the DataFrame: one row per URL per hierarchical level
        df_exploded = df.assign(Hierarchical_Path=df['url'].apply(get_all_subdirectory_levels)).explode('Hierarchical_Path')
        df_exploded = df_exploded.drop(columns=['url']) # Drop original URL


        # --- Identify and Remove Leaf Nodes ---
        st.write("Identifying and removing leaf nodes...")
        all_paths = set(df_exploded["Hierarchical_Path"].unique())

        # A path is a leaf if NO OTHER path starts with (path + something)
        def is_leaf_node(path, all_existing_paths):
            # A path is NOT a leaf if there exists another path that is strictly longer
            # and starts with the current path.
            path = path.rstrip('/') # Ensure consistent comparison
            for other_path in all_existing_paths:
                 other_path = other_path.rstrip('/')
                 if other_path.startswith(path) and len(other_path) > len(path):
                      return False # Found a longer path starting with this one, so it's not a leaf
            return True # No longer path found, it's a leaf

        # Apply the leaf check
        df_exploded['IsLeaf'] = df_exploded["Hierarchical_Path"].apply(lambda p: is_leaf_node(p, all_paths))

        # Filter out the leaf nodes
        df_non_leaf = df_exploded[~df_exploded['IsLeaf']].copy()
        df_non_leaf = df_non_leaf.drop(columns=['IsLeaf']) # Remove the helper column

        if df_non_leaf.empty:
             st.warning("No non-leaf directory paths found after processing. Check URL structures.")
             # Optionally show the exploded data before filtering: st.dataframe(df_exploded)
             return

        # --- Aggregation by Hierarchical Path ---
        st.write("Aggregating metrics by hierarchical path...")
        # Aggregate all numeric columns by the hierarchical path
        df_agg = df_non_leaf.groupby("Hierarchical_Path")[numeric_cols].sum().reset_index()

        # Sort by Traffic by default for better overview
        if "traffic" in df_agg.columns:
            df_agg = df_agg.sort_values(by="traffic", ascending=False)

        st.markdown("### Aggregated Data by Hierarchical Subdirectory (Excluding Leaf Nodes)")
        # Define formatting for aggregated table
        format_dict_hierarchical = {}
        for col in numeric_cols:
            format_dict_hierarchical[col] = "{:,.0f}" # Default comma format
        st.dataframe(df_agg.style.format(format_dict_hierarchical), use_container_width=True)


        # --- Plotly Sunburst Chart ---
        st.markdown("### Interactive Sunburst Chart")
        st.write("Visualize the hierarchy and aggregated metrics. Hover over segments for details.")

        # Prepare data for sunburst: need 'ids', 'parents', 'labels', 'values'
        # Split Hierarchical_Path into components
        def get_path_components(path):
            return [seg for seg in path.strip('/').split('/') if seg]

        sunburst_data = df_agg.copy()
        sunburst_data['path_components'] = sunburst_data['Hierarchical_Path'].apply(get_path_components)
        sunburst_data['label'] = sunburst_data['path_components'].apply(lambda x: x[-1] if x else 'Homepage') # Label is the last part
        sunburst_data['parent'] = sunburst_data['path_components'].apply(lambda x: "/" + "/".join(x[:-1]) + "/" if len(x) > 1 else "/") # Parent path
        sunburst_data['id'] = sunburst_data['Hierarchical_Path'] # ID is the full path

        # Ensure the root node '/' exists for parenting
        if '/' not in sunburst_data['id'].values:
            # Calculate root metrics by summing metrics where parent is '/'
            root_metrics = sunburst_data[sunburst_data['parent'] == '/'][numeric_cols].sum().to_dict()
            root_node = pd.DataFrame([{
                 'Hierarchical_Path': '/', 'label': 'Homepage', 'parent': '', 'id': '/',
                 **root_metrics # Add aggregated metrics for root
            }])
            sunburst_data = pd.concat([root_node, sunburst_data], ignore_index=True)


        # Select metric for sunburst color/value
        sunburst_metric = st.selectbox(
             "Select Metric for Sunburst Size/Color:",
             options=numeric_cols,
             index=numeric_cols.index('traffic') if 'traffic' in numeric_cols else 0,
             key="sunburst_metric"
        )

        if sunburst_metric in sunburst_data.columns:
             try:
                  fig_sunburst = px.sunburst(
                       sunburst_data,
                       ids='id',
                       parents='parent',
                       names='label',
                       values=sunburst_metric, # Size segments by selected metric
                       color=sunburst_metric,  # Color segments by selected metric
                       color_continuous_scale=px.colors.sequential.Blues, # Choose a color scale
                       hover_data={col: ':, .0f' for col in numeric_cols}, # Show all metrics on hover, formatted
                       title=f"Hierarchical Subdirectory Performance ({sunburst_metric}) - No Leaf Nodes",
                       height=800
                  )
                  fig_sunburst.update_layout(margin=dict(t=50, l=25, r=25, b=25))
                  st.plotly_chart(fig_sunburst, use_container_width=True)
             except Exception as plot_err:
                  st.error(f"Error creating Sunburst chart: {plot_err}")
        else:
             st.warning(f"Selected metric '{sunburst_metric}' not found in aggregated data for plotting.")

    else:
        st.info("Please upload an Excel file to begin the analysis.")


# ------------------------------------
# NEW TOOL: Gemini Analyzer for Dashboard Results
# ------------------------------------
def gemini_dashboard_analyzer_page():
    st.header("Gemini Analysis of Dashboard Results")
    st.markdown("""
    Use Google's Gemini model to analyze the results generated by the **URL Analysis Dashboard**.

    **Instructions:**
    1.  Run the **URL Analysis Dashboard** tool first to generate data for the URLs you want to analyze.
    2.  Return to this tool.
    3.  Enter your Google Gemini API Key below. You can get one from [Google AI Studio](https://aistudio.google.com/app/apikey).
    4.  Specify what aspects of the dashboard results you want Gemini to analyze in the text area.
    5.  Click "Analyze with Gemini".
    """)
    st.info("Ensure you have the `google-generativeai` library installed (`pip install google-generativeai`).")

    # 1. API Key Input
    api_key = st.text_input("Enter your Google Gemini API Key:", type="password", key="gemini_api_key_dashboard", help="Your key is not stored.")

    # 2. Check for Dashboard Data in Session State
    if 'dashboard_results_df' not in st.session_state or st.session_state.dashboard_results_df is None or st.session_state.dashboard_results_df.empty:
        st.warning("⚠️ Please run the 'URL Analysis Dashboard' tool first to generate data for analysis.")
        st.info("Navigate to 'URL Analysis Dashboard', enter URLs (and optionally search term), click 'Analyze URLs', then come back here.")
        # Optionally display placeholder or stop execution cleanly
        return # Stop further execution if no data

    # 3. Display the Data to be Analyzed (if available)
    st.subheader("Data from URL Analysis Dashboard (Will be sent to Gemini):")
    df_results = st.session_state.dashboard_results_df
    st.dataframe(df_results) # Display the data that will be sent

    # 4. User Prompt Input
    user_prompt = st.text_area("What do you want Gemini to analyze from this data?", height=150, key="gemini_user_prompt_dashboard",
                               placeholder="Examples:\n"
                                           "- Summarize the key findings and identify the top 3 performing URLs based on Cosine Similarity and Content Word Count.\n"
                                           "- Compare the URLs with the highest and lowest Grade Level. What differences do you observe in their other metrics?\n"
                                           "- Are there any potential content gaps suggested by low Cosine Similarity scores? Which URLs?\n"
                                           "- Analyze the relationship between the number of Unique Entities and Cosine Similarity scores.\n"
                                           "- Which URLs are lacking schema markup? Suggest potential schema types based on their H1 or Meta Title if possible.\n"
                                           "- Identify outliers in the data (e.g., unusually high/low word count, link count, etc.).")

    # 5. Analysis Button
    if st.button("Analyze with Gemini", key="gemini_analyze_button_dashboard"):
        if not api_key:
            st.error("❌ Please enter your Gemini API Key.")
            return
        if not user_prompt:
            st.error("❌ Please enter what you want Gemini to analyze.")
            return
        if df_results.empty:
            st.error("❌ No dashboard data available to analyze. Please run the dashboard first.")
            return

        try:
            with st.spinner("🧠 Contacting Gemini... Please wait."):
                # Configure Gemini
                genai.configure(api_key=api_key)

                # Select model and configure safety settings if needed
                # Simple configuration for gemini-pro
                generation_config = genai.types.GenerationConfig(
                     temperature=0.7 # Adjust creativity/factuality
                )
                # Set safety settings to allow more content (be cautious)
                safety_settings = [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                ]

                model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest", # Or gemini-pro
                                              generation_config=generation_config,
                                              safety_settings=safety_settings)

                # Format the DataFrame for the prompt (Markdown is generally good for LLMs)
                # Handle potentially large dataframes - inform the user if truncated
                max_rows_to_send = 50 # Limit the number of rows sent to avoid exceeding token limits
                if len(df_results) > max_rows_to_send:
                    st.info(f"Note: The dashboard data has {len(df_results)} rows. Sending the first {max_rows_to_send} rows to Gemini for analysis to avoid token limits.")
                    data_markdown = df_results.head(max_rows_to_send).to_markdown(index=False)
                else:
                    data_markdown = df_results.to_markdown(index=False)

                # Construct the full prompt with clear instructions and context
                full_prompt = f"""You are an expert SEO analyst reviewing data from a URL Analysis Dashboard.

The following table contains metrics comparing several URLs based on various SEO factors:
- URL: The web page address.
- Meta Title: The title tag of the page.
- H1: The main heading (H1 tag) of the page.
- Total Word Count: Total words scraped from the page body (excluding header/footer).
- Content Word Count: Word count from main content elements (p, li, headers, tables).
- Cosine Similarity: Relevance score (0-1) compared to a target search term (if provided, otherwise N/A). Higher is better.
- Grade Level: Flesch-Kincaid Grade Level for readability. Lower is generally easier to read.
- # Unique Entities: Count of distinct named entities (people, places, orgs, etc.) found using BERT.
- Nav Links: Number of links found in header and footer.
- Total Links: Total number of links on the page.
- Schema Types: Types of structured data markup found (e.g., Article, FAQPage). 'None' if not detected.
- Lists/Tables: Indicates presence of OL, UL, or TABLE elements.
- Images: Number of image tags found.
- Videos: Number of video elements or embeds found.
(Note: N/A indicates data was not available or applicable, e.g., similarity without a search term, or calculation errors)

**Dashboard Data Table (Markdown Format):**
```markdown
{data_markdown}
```

**User's Analysis Request:**
{user_prompt}

**Your Task:**
Analyze the provided data table based *only* on the information within it and the user's request.
- Provide concise, insightful analysis.
- Focus on comparisons, correlations, potential opportunities, or weaknesses revealed by the data.
- Structure your response clearly using Markdown (e.g., bullet points, bold text).
- Do **not** make up information not present in the table.
- Do **not** attempt to access external websites or specific files.
- Address the user's specific request directly.
"""
                # Make the API call
                response = model.generate_content(full_prompt)

                # Display the result
                st.subheader("✨ Gemini Analysis Results:")
                st.markdown(response.text)

        except ImportError:
             st.error("❌ The 'google-generativeai' library is not installed. Please install it: `pip install google-generativeai`")
        except Exception as e:
            st.error(f"❌ An error occurred while communicating with the Gemini API: {e}")
            st.error("Troubleshooting steps:")
            st.error("- Verify your API key is correct and has permissions.")
            st.error("- Check your internet connection.")
            st.error("- Review the Gemini API documentation for potential issues (e.g., rate limits, input constraints).")
            st.error("- Try a different model (e.g., gemini-pro) if using Flash.")
            st.error(f"- Details: {str(e)}") # Show error details


# ------------------------------------
# Main Streamlit App
# ------------------------------------
def main():
    st.set_page_config(
        page_title="Semantic Search SEO Analysis Tools | The SEO Consultant.ai",
        page_icon="📊", # Updated Icon
        layout="wide",
        initial_sidebar_state="expanded" # Keep sidebar open initially
    )
    # --- Hide default Streamlit elements ---
    hide_streamlit_elements = """
        <style>
        #MainMenu {visibility: hidden !important;}
        header {visibility: hidden !important;}
        /* Hide the Streamlit decoration line */
        [data-testid="stDecoration"] { display: none !important; }
        /* Hide the GitHub link in footer */
        footer > a[href*='streamlit.io'] { display: none !important; }
        /* Hide the profile container if logged in */
        div._profileContainer_gzau3_53 { display: none !important; }
        /* Adjust top padding */
        div.block-container {padding-top: 1.5rem;}
        </style>
        """
    st.markdown(hide_streamlit_elements, unsafe_allow_html=True)

    # --- Custom Navigation Header ---
    create_navigation_menu(logo_url)

    # --- Sidebar Navigation ---
    st.sidebar.header("SEO Analysis Tools")
    st.sidebar.markdown("Select a tool from the dropdown below:")

    tool_options = [
        "--- Analysis Dashboards ---",
        "URL Analysis Dashboard",
        "Gemini Analysis of Dashboard Results",
        "Google Search Console Analyzer",
        "Google Ads Search Term Analyzer",
        "--- Competitor & Gap Analysis ---",
        "Cosine Similarity - Competitor Analysis",
        "Semantic Gap Analyzer",
        "Entity Topic Gap Analysis",
        "--- Content Optimization ---",
        "Cosine Similarity - Content Heatmap",
        "Cosine Similarity - Every Sentence",
        "Top/Bottom Relevant Sentences",
        "Entity Frequency Charts",
        "Entity Visualizer (spaCy)",
        "People Also Asked Explorer",
        "--- Site Structure & Focus ---",
        "Site Content Focus Visualizer",
        "Entity Relationship Graph",
        "SEMRush - Top Subdirectory Analysis",
        "SEMRush - Hierarchical (No Leaf)",
    ]

    # Find the default index (avoiding separators)
    default_tool = "URL Analysis Dashboard"
    default_index = tool_options.index(default_tool) if default_tool in tool_options else 1

    tool = st.sidebar.selectbox(
        "Select Tool:",
        tool_options,
        index=default_index, # Start with a useful tool
        key="main_tool_select",
        # Filter out separator options before passing to the function
        format_func=lambda x: x if not x.startswith("---") else " " # Display separators as blank space
    )

    # --- Initialize session state for dashboard results ---
    if 'dashboard_results_df' not in st.session_state:
        st.session_state.dashboard_results_df = None

    # --- Routing Logic ---
    if tool == "URL Analysis Dashboard":
        url_analysis_dashboard_page()
    elif tool == "Gemini Analysis of Dashboard Results":
        gemini_dashboard_analyzer_page()
    elif tool == "Cosine Similarity - Competitor Analysis":
        cosine_similarity_competitor_analysis_page()
    elif tool == "Cosine Similarity - Every Sentence":
        cosine_similarity_every_embedding_page()
    elif tool == "Cosine Similarity - Content Heatmap":
        cosine_similarity_content_heatmap_page()
    elif tool == "Top/Bottom Relevant Sentences":
        top_bottom_embeddings_page()
    elif tool == "Entity Topic Gap Analysis":
        entity_analysis_page()
    elif tool == "Entity Visualizer (spaCy)":
        displacy_visualization_page()
    elif tool == "Entity Frequency Charts":
        named_entity_barchart_page()
    elif tool == "Semantic Gap Analyzer":
        ngram_tfidf_analysis_page()
    elif tool == "Keyword Clustering from Semantic Gaps": # Renamed option for clarity
        keyword_clustering_from_gap_page()
    elif tool == "People Also Asked Explorer": # Renamed option
        paa_extraction_clustering_page()
    elif tool == "Google Ads Search Term Analyzer":
        google_ads_search_term_analyzer_page()
    elif tool == "Google Search Console Analyzer":
        google_search_console_analysis_page()
    elif tool == "Site Content Focus Visualizer": # Renamed option
        semantic_clustering_page()
    elif tool == "Entity Relationship Graph":
        entity_relationship_graph_page()
    elif tool == "SEMRush - Top Subdirectory Analysis": # Renamed option
        semrush_organic_pages_by_subdirectory_page()
    elif tool == "SEMRush - Hierarchical (No Leaf)": # Renamed option
        semrush_hierarchical_subdirectories_minimal_no_leaf_with_intent_filter()
    elif tool.startswith("---"):
         st.info("Please select an analysis tool from the sidebar.") # Handle separator selection

    # --- Footer ---
    st.markdown("---")
    st.markdown("Powered by [The SEO Consultant.ai](https://theseoconsultant.ai)", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.8em; text-align: center;'>Note: Accuracy depends on data quality, model capabilities, and website structures. Tools may require significant resources.</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    # Ensure necessary NLTK data is available
    required_nltk_data = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'] # Added tagger
    for dataset in required_nltk_data:
        try:
            nltk.data.find(f'tokenizers/{dataset}' if dataset=='punkt' else f'corpora/{dataset}' if dataset in ['stopwords', 'wordnet'] else f'taggers/{dataset}')
            print(f"NLTK data '{dataset}' found.")
        except LookupError:
            print(f"NLTK data '{dataset}' not found. Downloading...")
            try:
                 nltk.download(dataset, quiet=True)
                 print(f"Downloaded '{dataset}'.")
            except Exception as e:
                 st.error(f"Failed to download NLTK data '{dataset}': {e}. Please ensure internet connection or download manually.")
                 # Decide if the app should exit or continue with potential errors
                 # exit() # Uncomment to stop if NLTK data is critical

    # Run the main application function
    main()
