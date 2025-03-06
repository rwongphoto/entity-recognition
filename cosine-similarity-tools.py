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

# NEW IMPORTS
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
            spacy.cli.download("en_core_web_md")
            nlp = spacy.load("en_core_web_md")
            print("en_core_web_md downloaded and loaded")
        except Exception as e:
            st.error(f"Failed to load spaCy model: {e}")
            return None
    return nlp

@st.cache_resource
def initialize_sentence_transformer():
    model = SentenceTransformer('all-MiniLM-L6-v2')
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

nltk.download('stopwords') #Ensure stopwords are downloaded if not already
stop_words = set(nltk.corpus.stopwords.words('english'))

def generate_topic_label(queries_in_topic):
    words = []
    for query in queries_in_topic:
        tokens = query.lower().split()
        filtered = [t for t in tokens if t not in stop_words]
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
                    chrome_options = Options()
                    chrome_options.add_argument("--headless")
                    chrome_options.add_argument("--no-sandbox")
                    chrome_options.add_argument("--disable-dev-shm-usage")
                    user_agent = get_random_user_agent()
                    chrome_options.add_argument(f"user-agent={user_agent}")
                    driver = webdriver.Chrome(options=chrome_options)
                    driver.get(url)
                    page_source = driver.page_source
                    soup = BeautifulSoup(page_source, "html.parser")
                    meta_title = driver.title  # Get meta title using Selenium
                    driver.quit()

                    content_soup = BeautifulSoup(page_source, "html.parser")
                    if content_soup.find("body"):
                         body = content_soup.find("body")
                         for tag in body.find_all(["header", "footer"]):
                             tag.decompose()
                         total_text = body.get_text(separator="\n", strip=True)
                    else:
                        total_text = ""

                    total_word_count = len(total_text.split())

                    custom_elements = body.find_all(["p", "li", "h1", "h2", "h3", "h4", "h5", "h6"]) if body else []
                    custom_words = []
                    for el in custom_elements:
                        custom_words.extend(el.get_text().split())

                    # Also include tables
                    for table in body.find_all("table"):
                        for row in table.find_all("tr"):
                            for cell in row.find_all(["td", "th"]):
                                custom_words.extend(cell.get_text().split())

                    # Ensure custom_word_count is never greater than total_word_count
                    custom_word_count = min(len(custom_words), total_word_count)

                    h1_tag = soup.find("h1").get_text(strip=True) if soup.find("h1") else "None"
                    header = soup.find("header")
                    footer = soup.find("footer")
                    header_links = len(header.find_all("a", href=True)) if header else 0
                    footer_links = len(footer.find_all("a", href=True)) if footer else 0
                    total_nav_links = header_links + footer_links
                    total_links = len(soup.find_all("a", href=True))

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

                    lists_tables = (
                        f"OL: {'Yes' if body.find('ol') else 'No'} | "
                        f"UL: {'Yes' if body.find('ul') else 'No'} | "
                        f"Table: {'Yes' if body.find('table') else 'No'}"
                    )
                    num_images = len(soup.find_all("img"))
                    num_videos = count_videos(soup)
                    similarity_val = similarity_results[i][1] if similarity_results[i][1] is not None else np.nan
                    entities = identify_entities(total_text, nlp_model) if total_text and nlp_model else []
                    unique_entity_count = len(set([ent[0] for ent in entities]))
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

                except Exception as e:
                    st.error(f"Error processing URL {url}: {e}")
                    data.append([url] + ["Error"] * 13)  # Append error placeholders

            df = pd.DataFrame(data, columns=[
                "URL",
                "Meta Title",
                "H1 Tag",
                "Total Word Count",
                "Custom Word Count (p, li, headers)",
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

            # Reorder for better presentation and rename columns
            df = df[[
                "URL",
                "Meta Title",
                "H1 Tag",
                "Total Word Count",
                "Custom Word Count (p, li, headers)",
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

            df.columns = [
                "URL",
                "Meta Title",
                "H1",
                "Total Word Count",
                "Content Word Count",
                "Cosine Similarity",
                "Grade Level",
                "# of Unique Entities",
                "Nav Links",
                "Total Links",
                "Schema Types",
                "Lists/Tables",
                "Images",
                "Videos",
            ]

            # Convert columns to numeric where applicable
            df["Cosine Similarity"] = pd.to_numeric(df["Cosine Similarity"], errors="coerce")
            df["Grade Level"] = pd.to_numeric(df["Grade Level"], errors="coerce")

            st.dataframe(df)

def cosine_similarity_competitor_analysis_page():
    st.title("Cosine Similarity Competitor Analysis")
    st.markdown("By: [The SEO Consultant.ai](https://theseoconsultant.ai)")
    search_term = st.text_input("Enter Search Term:", "")
    source_option = st.radio("Select content source for competitors:", options=["Extract from URL", "Paste Content"], index=0)
    if source_option == "Extract from URL":
        urls_input = st.text_area("Enter Competitor URLs (one per line):", "")
        competitor_urls = [url.strip() for url in urls_input.splitlines() if url.strip()]
    else:
        st.markdown("Paste the competitor content below. If you have multiple competitors, separate each content block with `---`.")
        pasted_content = st.text_area("Enter Competitor Content:", height=200)
        competitor_contents = [content.strip() for content in pasted_content.split('---') if content.strip()]

    if st.button("Calculate Similarity"):
        model = initialize_sentence_transformer()
        if source_option == "Extract from URL":
            if not competitor_urls:
                st.warning("Please enter at least one URL.")
                return
            with st.spinner("Calculating similarities from URLs..."):
                similarity_scores = calculate_overall_similarity(competitor_urls, search_term, model)
                # Extract content lengths
                content_lengths = []
                for url in competitor_urls:
                    text = extract_text_from_url(url)
                    content_lengths.append(len(text.split()) if text else 0)

            urls_plot = [url for url, score in similarity_scores]
            scores_plot = [score if score is not None else 0 for url, score in similarity_scores]

        else:  # Paste Content
            if not competitor_contents:
                st.warning("Please paste at least one content block.")
                return
            with st.spinner("Calculating similarities from pasted content..."):
                similarity_scores = []
                content_lengths = []  # Store content lengths
                for idx, content in enumerate(competitor_contents):
                    text_embedding = get_embedding(content, model)
                    search_embedding = get_embedding(search_term, model)
                    similarity = cosine_similarity([text_embedding], [search_embedding])[0][0]
                    similarity_scores.append((f"Competitor {idx+1}", similarity))
                    content_lengths.append(len(content.split()))  # Count words

            urls_plot = [label for label, score in similarity_scores]
            scores_plot = [score for label, score in similarity_scores]

        # --- Option 1:  2D Scatter Plot with Hover Data and Labels---
        df = pd.DataFrame({
            'Competitor': urls_plot,
            'Cosine Similarity': scores_plot,
            'Content Length (Words)': content_lengths
        })

        fig = px.scatter(df, x='Cosine Similarity', y='Content Length (Words)',
                         title='Competitor Analysis: Similarity vs. Content Length',
                         hover_data=['Competitor', 'Cosine Similarity', 'Content Length (Words)'],
                         color='Cosine Similarity',  # Color by similarity
                         color_continuous_scale=px.colors.sequential.Viridis,
                         text='Competitor') # Add this line

        fig.update_traces(textposition='top center') #And this line.

        fig.update_layout(
            xaxis_title="Cosine Similarity (Higher = More Relevant)",
            yaxis_title="Content Length (Words)",
            width=800,
            height=600
        )

        st.plotly_chart(fig)

        st.dataframe(df) #Show data

        # --- Option 2: Bar Chart with Secondary Y-Axis for Content Length and Labels---

        df = pd.DataFrame({
            'Competitor': urls_plot,
            'Cosine Similarity': scores_plot,
            'Content Length (Words)': content_lengths
        })

        # Sort by similarity for better visualization
        df = df.sort_values('Cosine Similarity', ascending=False)

        fig = go.Figure(data=[
            go.Bar(name='Cosine Similarity', x=df['Competitor'], y=df['Cosine Similarity'],
                   marker_color=df['Cosine Similarity'],  # Color by similarity
                   marker_colorscale='Viridis',
                    text=df['Competitor'],  # Add labels to the bars
                    textposition='outside'),
            go.Scatter(name='Content Length', x=df['Competitor'], y=df['Content Length (Words)'], yaxis='y2',
                       mode='lines+markers', marker=dict(color='red'))
        ])
        fig.update_traces(textfont_size=12) # Adjust text size as needed


        fig.update_layout(
            title='Competitor Analysis: Similarity and Content Length',
            xaxis_title="Competitor",
            yaxis_title="Cosine Similarity (Higher = More Relevant)",
            yaxis2=dict(title='Content Length (Words)', overlaying='y', side='right'),
            width=800,
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig)
        st.dataframe(df)

        # --- Option 3: Bubble Chart with labels---
        df = pd.DataFrame({
            'Competitor': urls_plot,
            'Cosine Similarity': scores_plot,
            'Content Length (Words)': content_lengths
        })

        fig = px.scatter(df, x='Cosine Similarity', y='Content Length (Words)',
                         size=[10] * len(df),  # Constant bubble size.  Change if you have a 3rd metric.
                         title='Competitor Analysis: Similarity, Content Length (Bubble Chart)',
                         hover_data=['Competitor', 'Cosine Similarity', 'Content Length (Words)'],
                         color='Cosine Similarity',
                         color_continuous_scale=px.colors.sequential.Viridis,
                         text='Competitor') # Add this line

        fig.update_traces(textposition='top center') #And this line.

        fig.update_layout(
            xaxis_title="Cosine Similarity (Higher = More Relevant)",
            yaxis_title="Content Length (Words)",
            width=800,
            height=600
        )
        st.plotly_chart(fig)
        st.dataframe(df)

def cosine_similarity_every_embedding_page():
    st.header("Cosine Similarity Score - Every Embedding")
    st.markdown("Calculates the cosine similarity score for each sentence in your input.")
    url = st.text_input("Enter URL (Optional):", key="every_embed_url", value="")
    use_url = st.checkbox("Use URL for Text Input", key="every_embed_use_url")
    text = st.text_area("Enter Text:", key="every_embed_text", value="", disabled=use_url)
    search_term = st.text_input("Enter Search Term:", key="every_embed_search", value="")
    if st.button("Calculate Similarity", key="every_embed_button"):
        if use_url:
            if url:
                with st.spinner(f"Extracting and analyzing text from {url}..."):
                    text = extract_text_from_url(url)
                    if not text:
                        st.error(f"Could not extract text from {url}. Please check the URL.")
                        return
            else:
                st.warning("Please enter a URL to extract the text.")
                return
        elif not text:
            st.warning("Please enter either text or a URL.")
            return
        model = initialize_sentence_transformer()
        with st.spinner("Calculating Similarities..."):
            sentences, similarities = calculate_similarity(text, search_term, model)
        st.subheader("Similarity Scores:")
        for i, (sentence, score) in enumerate(zip(sentences, similarities), 1):
            st.write(f"{i}. {sentence} (Similarity: {score:.4f})")

def cosine_similarity_content_heatmap_page():
    st.header("Cosine Similarity Content Heatmap")
    st.markdown("Green text is the most relevant to the search query. Red is the least relevant.")
    url = st.text_input("Enter URL (Optional):", key="heatmap_url", value="")
    use_url = st.checkbox("Use URL for Text Input", key="heatmap_use_url")
    input_text = st.text_area("Enter your text:", key="heatmap_input", height=300, value="", disabled=use_url)
    search_term = st.text_input("Enter your search term:", key="heatmap_search", value="")
    if st.button("Highlight", key="heatmap_button"):
        if use_url:
            if url:
                with st.spinner(f"Extracting and analyzing text from {url}..."):
                    text = extract_text_from_url(url)
                    if not text:
                        st.error(f"Could not extract text from {url}. Please check the URL.")
                        return
                input_text = text
            else:
                st.warning("Please enter a URL to extract the text.")
                return
        elif not input_text:
            st.error("Please enter either text or a URL.")
            return
        with st.spinner("Generating highlighted text..."):
            highlighted_text = highlight_text(input_text, search_term)
        st.markdown(highlighted_text, unsafe_allow_html=True)

def top_bottom_embeddings_page():
    st.header("Top 10 & Bottom 10 Embeddings")
    st.markdown("Assess and consider re-writing the bottom 10 embeddings.")
    url = st.text_input("Enter URL (Optional):", key="tb_url", value="")
    use_url = st.checkbox("Use URL for Text Input", key="tb_use_url")
    text = st.text_area("Enter your text:", key="top_bottom_text", height=300, value="", disabled=use_url)
    search_term = st.text_input("Enter your search term:", key="top_bottom_search", value="")
    top_n = st.slider("Number of results:", min_value=1, max_value=20, value=10, key="top_bottom_slider")
    if st.button("Search", key="top_bottom_button"):
        if use_url:
            if url:
                with st.spinner(f"Extracting and analyzing text from {url}..."):
                    text = extract_text_from_url(url)
                    if not text:
                        st.error(f"Could not extract text from {url}. Please check the URL.")
                        return
            else:
                st.error("Please enter either text or a URL.")
                return
        elif not text:
            st.error("Please enter either text or a URL.")
            return
        model = initialize_sentence_transformer()
        with st.spinner("Searching..."):
            top_sections, bottom_sections = rank_sections_by_similarity_bert(text, search_term, top_n)
        st.subheader("Top Sections (Highest Cosine Similarity):")
        for i, (sentence, score) in enumerate(top_sections, 1):
            st.write(f"{i}. {sentence} (Similarity: {score:.4f})")
        st.subheader("Bottom Sections (Lowest Cosine Similarity):")
        for i, (sentence, score) in enumerate(reversed(bottom_sections), 1):
            st.write(f"{i}. {sentence} (Similarity: {score:.4f})")

def entity_analysis_page():
    st.header("Entity Topic Gap Analysis")
    st.markdown("Analyze multiple sources to identify common entities missing on your site, *and* unique entities on your site.")
    
    # Get competitor content
    competitor_source_option = st.radio(
        "Select competitor content source:",
        options=["Extract from URL", "Paste Content"],
        index=0,
        key="entity_comp_source"
    )
    if competitor_source_option == "Extract from URL":
        competitor_input = st.text_area("Enter Competitor URLs (one per line):", key="entity_urls", value="")
        competitor_list = [url.strip() for url in competitor_input.splitlines() if url.strip()]
    else:
        st.markdown("Paste competitor content below. For multiple sources, separate each with `---`.")
        competitor_input = st.text_area("Enter Competitor Content:", key="entity_competitor_text", value="", height=200)
        competitor_list = [content.strip() for content in competitor_input.split('---') if content.strip()]
    
    # Get target content
    st.markdown("#### Target Site")
    target_option = st.radio(
        "Select target content source:",
        options=["Extract from URL", "Paste Content"],
        index=0,
        key="target_source"
    )
    if target_option == "Extract from URL":
        target_input = st.text_input("Enter Target URL:", key="target_url", value="")
    else:
        target_input = st.text_area("Paste target content:", key="target_text", value="", height=100)
    
    # Exclude content
    st.markdown("#### Exclude Content (Paste Only)")
    exclude_input = st.text_area("Paste content to exclude:", key="exclude_text", value="", height=100)
    exclude_types = st.multiselect(
        "Select entity types to exclude:",
        options=["CARDINAL", "DATE", "TIME", "PERCENT", "MONEY",
                 "QUANTITY", "ORDINAL", "GPE", "ORG", "PERSON", "NORP",
                 "FAC", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART",
                 "LAW", "LANGUAGE", "MISC"],
        default=["CARDINAL", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL"]
    )
    
    if st.button("Analyze", key="entity_button"):
        if not competitor_list:
            st.warning("Please provide at least one competitor content or URL.")
            return
        if not target_input:
            st.warning("Please provide target content or URL.")
            return
        
        with st.spinner("Extracting content and analyzing entities..."):
            nlp_model = load_spacy_model()
            if not nlp_model:
                return
            
            # Get target text
            if target_option == "Extract from URL":
                target_text = extract_text_from_url(target_input) if target_input else ""
            else:
                target_text = target_input
            
            target_entities = identify_entities(target_text, nlp_model) if target_text else []
            filtered_target_entities = [(entity, label) for entity, label in target_entities if label not in exclude_types]
            target_entity_counts = count_entities(filtered_target_entities, nlp_model)
            target_entities_set = set(target_entity_counts.keys())
            exclude_entities_set = {ent.text.lower() for ent in nlp_model(exclude_input).ents} if exclude_input else set()
            
            # Process competitor content
            entity_counts_per_source = {}
            for source in competitor_list:
                if competitor_source_option == "Extract from URL":
                    text = extract_text_from_url(source)
                else:
                    text = source
                if text:
                    entities = identify_entities(text, nlp_model)
                    filtered_entities = [
                        (entity, label) for entity, label in entities
                        if label not in exclude_types and entity.lower() not in exclude_entities_set
                    ]
                    source_counts = count_entities(filtered_entities, nlp_model)
                    entity_counts_per_source[source] = source_counts
            
            # Calculate aggregated gap entities (only if missing from target)
            gap_entities = Counter()
            for source, counts in entity_counts_per_source.items():
                for (entity, label), count in counts.items():
                    if (entity, label) not in target_entities_set:
                        gap_entities[(entity, label)] += count

            display_entity_barchart(gap_entities)

            # --- NEW: Build an aggregated table with unique competitor entity counts (number of sites) and Wikidata links ---
            st.markdown("### # of Sites Entities Are Present but Missing in Target")
            aggregated_gap_table = []
            aggregated_site_count = {}
            # For each competitor, count the unique occurrence of each entity (one count per site)
            for source, counts in entity_counts_per_source.items():
                for (entity, label), count in counts.items():
                    if (entity, label) not in target_entities_set:
                        # Increment by one for each site where the entity is found
                        aggregated_site_count[(entity, label)] = aggregated_site_count.get((entity, label), 0) + 1
            if aggregated_site_count:
                for (entity, label), site_count in aggregated_site_count.items():
                    wikidata_url = get_wikidata_link(entity)
                    aggregated_gap_table.append({
                        "Entity": entity,
                        "Label": label,
                        "# of Sites": site_count,
                        "Wikidata URL": wikidata_url if wikidata_url else "Not found"
                    })
                if aggregated_gap_table:
                    df_aggregated_gap = pd.DataFrame(aggregated_gap_table)
                    # Sort DataFrame by '# of Sites' in descending order
                    df_aggregated_gap = df_aggregated_gap.sort_values(by='# of Sites', ascending=False)
                    st.dataframe(df_aggregated_gap) # Use st.dataframe for sortable columns
                else:
                    st.write("No gap entities available for Wikidata linking.")
            else:
                st.write("No significant gap entities found.")
            
            
            st.markdown("### Entities Unique to Target Site")
            unique_target_entities = Counter()
            for (entity, label), count in target_entity_counts.items():
                if (entity, label) not in gap_entities:
                    unique_target_entities[(entity, label)] = count
            if unique_target_entities:
                for (entity, label), count in unique_target_entities.most_common(50):
                    st.write(f"- {entity} ({label}): {count}")
                display_entity_barchart(unique_target_entities)
            else:
                st.write("No unique entities found on the target site.")
            
            if exclude_input:
                st.markdown("### Entities from Exclude Content (Excluded from Analysis)")
                exclude_doc = nlp_model(exclude_input)
                exclude_entities_list = [(ent.text, ent.label_) for ent in exclude_doc.ents]
                exclude_entity_counts = count_entities(exclude_entities_list, nlp_model)
                for (entity, label), count in exclude_entity_counts.most_common(50):
                    st.write(f"- {entity} ({label}): {count}")
            
            st.markdown("### Entities Per Competitor Source")
            for source, entity_counts_local in entity_counts_per_source.items():
                st.markdown(f"#### Source: {source}")
                if entity_counts_local:
                    for (entity, label), count in entity_counts_local.most_common(50):
                        st.write(f"- {entity} ({label}): {count}")
                else:
                    st.write("No relevant entities found.")

def displacy_visualization_page():
    st.header("Entity Visualizer")
    st.markdown("Visualize named entities within your content.")
    url = st.text_input("Enter a URL (Optional):", key="displacy_url", value="")
    use_url = st.checkbox("Use URL for Text Input", key="displacy_use_url")
    text = st.text_area("Enter Text:", key="displacy_text", value="", disabled=use_url)
    if st.button("Visualize Entities", key="displacy_button"):
        if use_url:
            if url:
                with st.spinner("Extracting text from URL..."):
                    text = extract_text_from_url(url)
                    if not text:
                        st.error("Could not extract text from the URL.")
                        return
            else:
                st.warning("Please enter a URL or uncheck 'Use URL for Text Input'.")
                return
        elif not text:
            st.warning("Please enter text or a URL.")
            return
        nlp_model = load_spacy_model()
        if not nlp_model:
            return
        doc = nlp_model(text)
        try:
            html = spacy.displacy.render(doc, style="ent", page=True)
            st.components.v1.html(html, height=600, scrolling=True)
        except Exception as e:
            st.error(f"Error rendering visualization: {e}")

def named_entity_barchart_page():
    st.header("Entity Frequency Charts")
    st.markdown("Visualize the most frequent named entities across multiple sources. This tool counts the total number of occurrences for each entity.")
    input_method = st.radio(
        "Select content input method:",
        options=["Extract from URL", "Paste Content"],
        key="entity_barchart_input"
    )
    if input_method == "Paste Content":
        st.markdown("Please paste your content. For multiple sources, separate each block with `---`.")
        text = st.text_area("Enter Text:", key="barchart_text", height=300, value="")
    else:
        st.markdown("Enter one or more URLs (one per line). The app will fetch and combine the text from each URL.")
        urls_input = st.text_area("Enter URLs (one per line):", key="barchart_url", value="")
        urls = [url.strip() for url in urls_input.splitlines() if url.strip()]
    if st.button("Generate Visualizations", key="barchart_button"):
        all_text = ""
        entity_texts_by_url: Dict[str, str] = {}
        if input_method == "Paste Content":
            if not text:
                st.warning("Please enter the text to proceed.")
                return
            all_text = text
        else:
            if not urls:
                st.warning("Please enter at least one URL.")
                return
            with st.spinner("Extracting text from URLs..."):
                for url in urls:
                    extracted_text = extract_text_from_url(url)
                    if extracted_text:
                        entity_texts_by_url[url] = extracted_text
                        all_text += extracted_text + "\n"
                    else:
                        st.warning(f"Couldn't grab the text from {url}...")
        with st.spinner("Analyzing entities and generating visualizations..."):
            nlp_model = load_spacy_model()
            if not nlp_model:
                st.error("Could not load spaCy model. Aborting.")
                return
            entities = identify_entities(all_text, nlp_model)
            filtered_entities = [(entity, label) for entity, label in entities if label not in ["CARDINAL", "PERCENT", "MONEY"]]
            entity_counts = count_entities_total(filtered_entities, nlp_model)
            if entity_counts:
                st.subheader("Total Entity Counts")
                display_entity_barchart(entity_counts)
                st.subheader("Entity Wordcloud")
                display_entity_wordcloud(entity_counts)
                if input_method == "Extract from URL":
                    st.subheader("List of Entities from each URL (with counts):")
                    for url in urls:
                        text_from_url = entity_texts_by_url.get(url)
                        if text_from_url:
                            st.write(f"**Text from {url}:**")
                            url_entities = identify_entities(text_from_url, nlp_model)
                            url_entity_counts = count_entities_total(url_entities, nlp_model)
                            for (entity, label), count in url_entity_counts.most_common():
                                st.write(f"- {entity} ({label}): {count}")
                        else:
                            st.write(f"No text for {url}")
            else:
                st.warning("No relevant entities found. Please check your text or URL(s).")

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
        "Select content source:",
        options=["Extract from URL", "Paste Content"],
        index=0,
        key="competitor_source"
    )
    if competitor_source_option == "Extract from URL":
        competitor_input = st.text_area("Enter Competitor URLs (one per line):", key="competitor_urls", value="")
        competitor_list = [url.strip() for url in competitor_input.splitlines() if url.strip()]
    else:
        st.markdown("Paste competitor content below. Separate each competitor content block with `---`.")
        competitor_input = st.text_area("Enter Competitor Content:", key="competitor_text", value="", height=200)
        competitor_list = [content.strip() for content in competitor_input.split('---') if content.strip()]

    # Target input
    st.subheader("Your Site")
    target_source_option = st.radio(
        "Select content source:",
        options=["Extract from URL", "Paste Content"],
        index=0,
        key="target_source"
    )
    if target_source_option == "Extract from URL":
        target_url = st.text_input("Enter Your Target URL:", key="target_url", value="")
    else:
        target_text = st.text_area("Paste your target content:", key="target_text", value="", height=200)

    # Word options
    st.subheader("Word Options")
    n_value = st.selectbox("Select # of Words in Phrase:", options=[1,2,3,4,5,6,7,8,9,10], index=1)
    st.markdown("*(For example, choose 2 for bigrams)*")
    min_df = st.number_input("Minimum Frequency:", value=1, min_value=1)
    max_df = st.number_input("Maximum Frequency:", value=1.0, min_value=0.0, step=0.1)
    top_n = st.slider("Number of top results to display:", min_value=1, max_value=50, value=10)
    num_topics = st.slider("Number of topics for LDA:", min_value=2, max_value=15, value=5, key="lda_topics") # Slider for LDA topics

    if st.button("Analyze Content Gaps", key="content_gap_button"):
        if competitor_source_option == "Extract from URL" and not competitor_list:
            st.warning("Please enter at least one competitor URL.")
            return
        if target_source_option == "Extract from URL" and not target_url:
            st.warning("Please enter your target URL.")
            return

        # Extract competitor content
        competitor_texts = []
        valid_competitor_sources = []
        with st.spinner("Extracting competitor content..."):
            for source in competitor_list:
                if competitor_source_option == "Extract from URL":
                    text = extract_relevant_text_from_url(source)
                else:
                    text = source
                if text:
                    valid_competitor_sources.append(source)
                    competitor_texts.append(text)
                else:
                    st.warning(f"Could not extract content from: {source}")

        # Extract target content
        if target_source_option == "Extract from URL":
            target_content = extract_text_from_url(target_url)
            if not target_content:
                st.warning(f"Could not extract content from target URL: {target_url}")
                return
        else:
            target_content = target_text

        nlp_model = load_spacy_model()
        competitor_texts = [preprocess_text(text, nlp_model) for text in competitor_texts]
        target_content = preprocess_text(target_content, nlp_model)
        if not competitor_texts:
            st.error("No competitor content was extracted.")
            return

        # Calculate TF-IDF scores
        with st.spinner("Calculating TF-IDF scores for competitors..."):
            vectorizer = TfidfVectorizer(ngram_range=(n_value, n_value), min_df=min_df, max_df=max_df, stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(competitor_texts)
            feature_names = vectorizer.get_feature_names_out()
            df_tfidf_competitors = pd.DataFrame(tfidf_matrix.toarray(), index=valid_competitor_sources, columns=feature_names)

        # --- LDA Topic Modeling ---
        with st.spinner("Performing Latent Dirichlet Allocation (LDA)..."):
            lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
            lda_output = lda_model.fit_transform(tfidf_matrix)

            st.subheader("Identified Topics from Competitor Content (LDA)")
            topic_keywords = {}
            for i, topic in enumerate(lda_model.components_):
                top_keyword_indices = topic.argsort()[-10:][::-1]  # Top 10 keywords per topic
                keywords = [feature_names[i] for i in top_keyword_indices]
                topic_keywords[f"Topic {i+1}"] = keywords
                st.markdown(f"**Topic {i+1}:** {', '.join(keywords)}")

            # Prepare topic distribution for visualization (optional - for later)
            topic_distribution_df = pd.DataFrame(lda_output, index=valid_competitor_sources, columns=[f"Topic {i+1}" for i in range(num_topics)])
            st.dataframe(topic_distribution_df) # Display topic distribution dataframe

        with st.spinner("Calculating TF-IDF scores for target content..."):
            target_tfidf_vector = vectorizer.transform([target_content])
            df_tfidf_target = pd.DataFrame(target_tfidf_vector.toarray(), index=["Target Content"], columns=feature_names)

        # For each competitor, get its top n-grams (based on TF-IDF)
        top_ngrams_competitors = {}
        for source in valid_competitor_sources:
            row = df_tfidf_competitors.loc[source]
            sorted_row = row.sort_values(ascending=False)
            top_ngrams = sorted_row.head(top_n)
            top_ngrams_competitors[source] = list(top_ngrams.index)

        model = initialize_sentence_transformer()
        with st.spinner("Calculating SentenceTransformer embedding for target content..."):
            target_embedding = get_embedding(target_content, model)
        competitor_embeddings = []
        with st.spinner("Calculating SentenceTransformer embeddings for competitors..."):
            for text in competitor_texts:
                emb = get_embedding(text, model)
                competitor_embeddings.append(emb)

        # Compute candidate gap scores – now include competitor source in the tuple
        candidate_scores = []
        for idx, source in enumerate(valid_competitor_sources):
            for ngram in top_ngrams_competitors[source]:
                if ngram in df_tfidf_target.columns:
                    competitor_tfidf = df_tfidf_competitors.loc[source, ngram]
                    target_tfidf = df_tfidf_target.iloc[0][ngram]
                    tfidf_diff = competitor_tfidf - target_tfidf
                    if tfidf_diff <= 0:
                        continue
                    ngram_embedding = get_embedding(ngram, model)
                    competitor_similarity = cosine_similarity([ngram_embedding], [competitor_embeddings[idx]])[0][0]
                    target_similarity = cosine_similarity([ngram_embedding], [target_embedding])[0][0]
                    bert_diff = competitor_similarity - target_similarity
                    candidate_scores.append((source, ngram, tfidf_diff, bert_diff))
                else:
                    competitor_tfidf = df_tfidf_competitors.loc[source, ngram]
                    ngram_embedding = get_embedding(ngram, model)
                    competitor_similarity = cosine_similarity([ngram_embedding], [competitor_embeddings[idx]])[0][0]
                    bert_diff = competitor_similarity
                    candidate_scores.append((source, ngram, competitor_tfidf, bert_diff))

        if not candidate_scores:
            st.error("No gap n-grams were identified. Consider adjusting your TF-IDF parameters.")
            return

        tfidf_vals = [item[2] for item in candidate_scores]
        bert_vals = [item[3] for item in candidate_scores]
        min_tfidf, max_tfidf = min(tfidf_vals), max(tfidf_vals)
        min_bert, max_bert = min(bert_vals), max(bert_vals)
        epsilon = 1e-8
        tfidf_weight = 0.4  # Hard-coded TF-IDF weight

        norm_candidates = []
        for source, ngram, tfidf_diff, bert_diff in candidate_scores:
            norm_tfidf = (tfidf_diff - min_tfidf) / (max_tfidf - min_tfidf + epsilon)
            norm_bert = (bert_diff - min_bert) / (max_bert - min_bert + epsilon)
            combined_score = tfidf_weight * norm_tfidf + (1 - tfidf_weight) * norm_bert
            if combined_score > 0:
                norm_candidates.append((source, ngram, combined_score))
        norm_candidates.sort(key=lambda x: x[2], reverse=True)

        # Display consolidated gap analysis table
        st.markdown("### Consolidated Semantic Gap Analysis (All Competitors)")
        df_consolidated = pd.DataFrame(norm_candidates, columns=['Competitor', 'N-gram', 'Gap Score'])
        st.dataframe(df_consolidated)

        # Display per-competitor gap analysis tables
        st.markdown("### Per-Competitor Semantic Gap Analysis")
        for source in valid_competitor_sources:
            candidate_list = [ (s, n, score) for s, n, score in norm_candidates if s == source ]
            if candidate_list:
                df_source = pd.DataFrame(candidate_list, columns=['Competitor', 'N-gram', 'Gap Score']).sort_values(by='Gap Score', ascending=False)
                st.markdown(f"#### Competitor: {source}")
                st.dataframe(df_source)

        # --- Updated: Use Gap Scores as Weights in the Wordcloud ---
        st.subheader("Combined Semantic Gap Wordcloud")
        gap_scores = {}
        for source, ngram, score in norm_candidates:
            gap_scores[ngram] = gap_scores.get(ngram, 0) + score
        if gap_scores:
            display_entity_wordcloud(gap_scores)
        else:
            st.write("No combined gap n-grams to create a wordcloud.")

        # Display a wordcloud for each competitor using gap score weights
        st.subheader("Per-Competitor Gap Wordclouds")
        for source in valid_competitor_sources:
            comp_gap_scores = {}
            for s, ngram, score in norm_candidates:
                if s == source:
                    comp_gap_scores[ngram] = comp_gap_scores.get(ngram, 0) + score
            if comp_gap_scores:
                st.markdown(f"**Wordcloud for Competitor: {source}**")
                display_entity_wordcloud(comp_gap_scores)
            else:
                st.write(f"No gap n-grams for competitor: {source}")

def keyword_clustering_from_gap_page():
    st.header("Keyword Clusters")
    st.markdown(
        """
        This tool combines semantic gap analysis with keyword clustering.
        First, it identifies key phrases where your competitors outperform your target.
        Then, it uses machine learning for these gap phrases and clusters them based on their semantic similarity.
        The resulting clusters (and their representative keywords) are displayed below.
        """
    )
    st.subheader("Competitors")
    competitor_source_option = st.radio("Select competitor content source:", options=["Extract from URL", "Paste Content"], index=0, key="comp_source")
    if competitor_source_option == "Extract from URL":
        competitor_input = st.text_area("Enter Competitor URLs (one per line):", key="comp_urls", value="")
        competitor_list = [url.strip() for url in competitor_input.splitlines() if url.strip()]
    else:
        st.markdown("Paste competitor content below. Separate each block with `---`.")
        competitor_input = st.text_area("Enter Competitor Content:", key="competitor_content", value="", height=200)
        competitor_list = [content.strip() for content in competitor_input.split('---') if content.strip()]
    st.subheader("Your Site")
    target_source_option = st.radio("Select target content source:", options=["Extract from URL", "Paste Content"], index=0, key="target_source_cluster")
    if target_source_option == "Extract from URL":
        target_url = st.text_input("Enter Your URL:", key="target_url", value="")
    else:
        target_text = st.text_area("Paste your content:", key="target_content", value="", height=200)
    st.subheader("Word Count Settings")
    n_value = st.selectbox("Select # of Words in Phrase:", options=[1,2,3,4,5], index=1, key="ngram_n")
    min_df = st.number_input("Minimum Frequency:", value=1, min_value=1, key="min_df_gap")
    max_df = st.number_input("Maximum Frequency:", value=1.0, min_value=0.0, step=0.1, key="max_df_gap")
    top_n = st.slider("Max # of Results per Competitor:", min_value=1, max_value=50, value=10, key="top_n_gap")
    st.subheader("Clustering Settings")
    algorithm = st.selectbox("Select Clustering Type:", options=["Kindred Spirit", "Affinity Stack"], key="clustering_algo_gap")
    n_clusters = st.number_input("Number of Clusters:", min_value=1, value=5, key="clusters_num")
    
    if st.button("Analyze & Cluster Gaps", key="gap_cluster_button"):
        if not competitor_list:
            st.warning("Please enter at least one competitor URL or content.")
            return
        if (target_source_option == "Extract from URL" and not target_url) or (target_source_option == "Paste Content" and not target_text):
            st.warning("Please enter your target URL or content.")
            return
        competitor_texts = []
        valid_competitor_sources = []
        with st.spinner("Extracting competitor content..."):
            for source in competitor_list:
                if competitor_source_option == "Extract from URL":
                    text = extract_relevant_text_from_url(source)
                else:
                    text = source
                if text:
                    valid_competitor_sources.append(source)
                    competitor_texts.append(text)
                else:
                    st.warning(f"Could not extract content from: {source}")
        if target_source_option == "Extract from URL":
            target_content = extract_relevant_text_from_url(target_url)
            if not target_content:
                st.error("Could not extract content from the target URL.")
                return
        else:
            target_content = target_text
        nlp_model = load_spacy_model()
        competitor_texts = [preprocess_text(text, nlp_model) for text in competitor_texts]
        target_content = preprocess_text(target_content, nlp_model)
        if not competitor_texts:
            st.error("No competitor content was extracted.")
            return
        with st.spinner("Calculating TF-IDF scores for competitors..."):
            vectorizer = TfidfVectorizer(ngram_range=(n_value, n_value), min_df=min_df, max_df=max_df, stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(competitor_texts)
            feature_names = vectorizer.get_feature_names_out()
            df_tfidf_competitors = pd.DataFrame(tfidf_matrix.toarray(), index=valid_competitor_sources, columns=feature_names)
        with st.spinner("Calculating TF-IDF scores for target content..."):
            target_tfidf_vector = vectorizer.transform([target_content])
            df_tfidf_target = pd.DataFrame(target_tfidf_vector.toarray(), index=["Target Content"], columns=feature_names)
        # For each competitor, get its top n-grams
        top_ngrams_competitors = {}
        for source in valid_competitor_sources:
            row = df_tfidf_competitors.loc[source]
            sorted_row = row.sort_values(ascending=False)
            top_ngrams = sorted_row.head(top_n)
            top_ngrams_competitors[source] = list(top_ngrams.index)
        model = initialize_sentence_transformer()
        with st.spinner("Calculating SentenceTransformer embedding for target content..."):
            target_embedding = get_embedding(target_content, model)
        competitor_embeddings = []
        with st.spinner("Calculating SentenceTransformer embeddings for competitors..."):
            for text in competitor_texts:
                emb = get_embedding(text, model)
                competitor_embeddings.append(emb)
        candidate_scores = []
        for idx, source in enumerate(valid_competitor_sources):
            for ngram in top_ngrams_competitors[source]:
                if ngram in df_tfidf_target.columns:
                    competitor_tfidf = df_tfidf_competitors.loc[source, ngram]
                    target_tfidf = df_tfidf_target.iloc[0][ngram]
                    tfidf_diff = competitor_tfidf - target_tfidf
                    if tfidf_diff <= 0:
                        continue
                    ngram_embedding = get_embedding(ngram, model)
                    competitor_similarity = cosine_similarity([ngram_embedding], [competitor_embeddings[idx]])[0][0]
                    target_similarity = cosine_similarity([ngram_embedding], [target_embedding])[0][0]
                    bert_diff = competitor_similarity - target_similarity
                    candidate_scores.append((ngram, tfidf_diff, bert_diff))
                else:
                    competitor_tfidf = df_tfidf_competitors.loc[source, ngram]
                    ngram_embedding = get_embedding(ngram, model)
                    competitor_similarity = cosine_similarity([ngram_embedding], [competitor_embeddings[idx]])[0][0]
                    bert_diff = competitor_similarity
                    candidate_scores.append((ngram, competitor_tfidf, bert_diff))
        if not candidate_scores:
            st.error("No gap n-grams were identified. Consider adjusting your TF-IDF parameters.")
            return
        tfidf_vals = [item[1] for item in candidate_scores]
        bert_vals = [item[2] for item in candidate_scores]
        min_tfidf, max_tfidf = min(tfidf_vals), max(tfidf_vals)
        min_bert, max_bert = min(bert_vals), max(bert_vals)
        epsilon = 1e-8
        tfidf_weight = 0.4  # Hard-coded TF-IDF weight
        norm_candidates = []
        for ngram, tfidf_diff, bert_diff in candidate_scores:
            norm_tfidf = (tfidf_diff - min_tfidf) / (max_tfidf - min_tfidf + epsilon)
            norm_bert = (bert_diff - min_bert) / (max_bert - min_bert + epsilon)
            combined_score = tfidf_weight * norm_tfidf + (1 - tfidf_weight) * norm_bert
            if combined_score > 0:
                norm_candidates.append((ngram, combined_score))
        norm_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # ----- Clustering & Visualization -----
        gap_ngrams = [ngram for ngram, score in norm_candidates]
        if not gap_ngrams:
            st.error("No valid gap n-grams for clustering.")
            return
        valid_gap_ngrams = []
        gap_embeddings = []
        with st.spinner("Computing SentenceTransformer embeddings for gap n-grams..."):
            for gram in gap_ngrams:
                emb = get_embedding(gram, model)
                if emb is not None:
                    gap_embeddings.append(emb)
                    valid_gap_ngrams.append(gram)
        if len(valid_gap_ngrams) == 0:
            st.error("Could not compute embeddings for any gap n-grams.")
            return
        gap_embeddings = np.vstack(gap_embeddings)
        if algorithm == "Kindred Spirit":
            clustering_model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            cluster_labels = clustering_model.fit_predict(gap_embeddings)
            centers = clustering_model.cluster_centers_
            rep_keywords = {}
            for i in range(n_clusters):
                cluster_grams = [ng for ng, label in zip(valid_gap_ngrams, cluster_labels) if label == i]
                if not cluster_grams:
                    continue
                cluster_embeddings = gap_embeddings[cluster_labels == i]
                distances = np.linalg.norm(cluster_embeddings - centers[i], axis=1)
                rep_keyword = cluster_grams[np.argmin(distances)]
                rep_keywords[i] = rep_keyword
        elif algorithm == "Affinity Stack":
            clustering_model = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels = clustering_model.fit_predict(gap_embeddings)
            rep_keywords = {}
            for i in range(n_clusters):
                cluster_grams = [ng for ng, label in zip(valid_gap_ngrams, cluster_labels) if label == i]
                cluster_embeddings = gap_embeddings[cluster_labels == i]
                if len(cluster_embeddings) > 1:
                    sims = cosine_similarity(cluster_embeddings, cluster_embeddings)
                    rep_keyword = cluster_grams[np.argmax(np.sum(sims, axis=1))]
                else:
                    rep_keyword = cluster_grams[0]
                rep_keywords[i] = rep_keyword

        # --- Display the interactive Plotly chart at the top ---
        st.markdown("### Interactive Cluster Visualization")
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(gap_embeddings)
        df_plot = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'Keyword': valid_gap_ngrams,
            'Cluster': [f"Cluster {label}" for label in cluster_labels]
        })
        fig = px.scatter(df_plot, x='x', y='y', color='Cluster', text='Keyword', hover_data=['Keyword'],
                         title="Semantic Opportunity Clusters")
        fig.update_traces(textposition='top center')
        fig.update_layout(
            xaxis_title="Topic Focus: Broad vs. Niche",
            yaxis_title="Competitive Pressure: High vs. Low"
        )
        st.plotly_chart(fig)
        
        # --- Display the detailed clusters ---
        st.markdown("### Keyword Clusters")
        clusters = {}
        for gram, label in zip(valid_gap_ngrams, cluster_labels):
            clusters.setdefault(label, []).append(gram)
        for label, gram_list in clusters.items():
            rep = rep_keywords.get(label, "N/A")
            st.markdown(f"**Cluster {label}** (Representative: {rep}):")
            for gram in gram_list:
                st.write(f" - {gram}")

def paa_extraction_clustering_page():
    st.header("People Also Asked Recommendations")
    st.markdown(
        """
        This tool is designed to build a topic cluster around a main search query that helps address a user's search intent.
        You can either write pages to support the main page or address the intent behind People Also Asked without necessarily copying questions verbatim.
        """
    )

    search_query = st.text_input("Enter Search Query:", "")
    if st.button("Analyze"):
        if not search_query:
            st.warning("Please enter a search query.")
            return

        # user_agent definition removed from here

        def get_paa(query, max_depth=10):
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            user_agent = get_random_user_agent()  # Define user_agent HERE
            chrome_options.add_argument(f"user-agent={user_agent}")
            driver = webdriver.Chrome(options=chrome_options)
            driver.get("https://www.google.com/search?q=" + query)
            time.sleep(3)

            paa_set = set()
            def extract_paa_recursive(depth, max_depth):
                if depth > max_depth:
                    return
                try:
                    elements = driver.find_elements(By.CSS_SELECTOR, "div[jsname='Cpkphb']")
                    if not elements:
                        elements = driver.find_elements(By.XPATH, "//div[@class='related-question-pair']")
                    for el in elements:
                        question_text = el.text.strip()
                        if question_text and question_text not in paa_set:
                            paa_set.add(question_text)
                            try:
                                driver.execute_script("arguments[0].click();", el)
                                time.sleep(2)
                                extract_paa_recursive(depth + 1, max_depth)
                            except Exception:
                                continue
                except Exception as e:
                    st.error(f"Error during PAA extraction for query '{query}': {e}")
            extract_paa_recursive(1, max_depth)
            driver.quit()
            return paa_set

        st.info("I'm researching...")
        paa_questions = get_paa(search_query, max_depth=20)

        st.info("Autocomplete suggestions...")
        import requests
        autocomplete_url = "http://suggestqueries.google.com/complete/search"
        params = {"client": "chrome", "q": search_query}
        try:
            response = requests.get(autocomplete_url, params=params)
            if response.status_code == 200:
                suggestions = response.json()[1]
            else:
                suggestions = []
        except Exception as e:
            st.error(f"Error fetching autocomplete suggestions: {e}")
            suggestions = []

        st.info("Related searches...")
        related_searches = []
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            user_agent = get_random_user_agent() # Moved to get_paa
            chrome_options.add_argument(f"user-agent={user_agent}")# Moved to get_paa
            driver2 = webdriver.Chrome(options=chrome_options)
            driver2.get("https://www.google.com/search?q=" + search_query)
            time.sleep(3)
            related_elements = driver2.find_elements(By.CSS_SELECTOR, "p.nVcaUb")
            for el in related_elements:
                text = el.text.strip()
                if text:
                    related_searches.append(text)
            driver2.quit()
        except Exception as e:
            st.error(f"Error extracting related searches: {e}")

        combined_questions = list(paa_questions) + suggestions + related_searches

        st.info("Analyzing similarity...")
        model = initialize_sentence_transformer()
        query_embedding = get_embedding(search_query, model)
        question_similarities = []
        for q in combined_questions:
            q_embedding = get_embedding(q, model)
            sim = cosine_similarity([q_embedding], [query_embedding])[0][0]
            question_similarities.append((q, sim))

        if not question_similarities:
            st.warning("No questions were extracted to analyze.")
            return

        avg_sim = np.mean([sim for _, sim in question_similarities])
        st.write(f"Average Similarity Score: {avg_sim:.4f}")
        recommended = [(q, sim) for q, sim in question_similarities if sim >= avg_sim]
        recommended.sort(key=lambda x: x[1], reverse=True)

        st.subheader("Topic Tree")
        if recommended:
            rec_texts = [q for q, sim in recommended]
            dendro_labels = rec_texts
            dendro_embeddings = np.vstack([get_embedding(text, model) for text in dendro_labels])
            import plotly.figure_factory as ff
            dendro = ff.create_dendrogram(dendro_embeddings, orientation='left', labels=dendro_labels)
            dendro.update_layout(width=800, height=600)
            st.plotly_chart(dendro)
        else:
            st.info("No recommended questions to visualize.")

        st.subheader("Most Relevant Related Search Queries")
        for q, sim in recommended:
            st.write(f"{q} (Similarity: {sim:.4f})")

        st.subheader("All Related Search Queries")
        for q in combined_questions:
            st.write(f"- {q}")

# ------------------------------------
# NEW TOOL: Google Ads Search Term Analyzer (with Classifier)
# ------------------------------------
def google_ads_search_term_analyzer_page():
    st.header("Google Ads Search Term Analyzer")
    st.markdown(
        """
        Upload an Excel file (.xlsx) from your Google Ads search terms report and analyze it.
        This tool extracts n-grams which can be used to optimize your campaigns. Your paid search data can also be used to inform SEO content strategy if you have a big enough sample size.
        """
    )

    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

    if uploaded_file is not None:
        try:
            # Read the Excel file, skipping the first two rows.
            df = pd.read_excel(uploaded_file, skiprows=2)

            # Rename columns for consistency and readability.
            df = df.rename(columns={
                "Search term": "Search term",
                "Match type": "Match type",
                "Added/Excluded": "Added/Excluded",
                "Campaign": "Campaign",
                "Ad group": "Ad group",
                "Clicks": "Clicks",
                "Impr.": "Impressions",
                "Currency code": "Currency code",
                "Cost": "Cost",
                "Avg. CPC": "Avg. CPC",
                "Conv. rate": "Conversion Rate",
                "Conversions": "Conversions",
                "Cost / conv.": "Cost per Conversion"
            })

            # Input Validation (check for required columns)
            required_columns = ["Search term", "Clicks", "Impressions", "Cost", "Conversions"]
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                st.error(f"The following required columns are missing: {', '.join(missing_cols)}")
                return

            # Convert numeric columns, handling errors.
            for col in ["Clicks", "Impressions", "Cost", "Conversions"]:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(0)
                except KeyError:
                    st.error(f"Column '{col}' not found in the uploaded Excel file.")
                    return

            st.subheader("N-gram Analysis")
            # New UI option for n-gram extraction method: Contiguous vs Skip-grams
            extraction_method = st.radio("Select N-gram Extraction Method:", options=["Contiguous n-grams", "Skip-grams"], index=0)
            n_value = st.selectbox("Select N (number of words in phrase):", options=[1, 2, 3, 4], index=1)
            min_frequency = st.number_input("Minimum Frequency:", value=2, min_value=1)

            # Define extraction functions.
            def extract_ngrams(text, n):
                text = str(text).lower()
                tokens = word_tokenize(text)
                tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalnum() and t not in stop_words]
                ngrams_list = list(nltk.ngrams(tokens, n))
                return [" ".join(gram) for gram in ngrams_list]

            def extract_skipgrams(text, n):
                import itertools
                text = str(text).lower()
                tokens = word_tokenize(text)
                tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalnum() and t not in stop_words]
                if len(tokens) < n:
                    return []
                skipgrams_list = []
                for combo in itertools.combinations(range(len(tokens)), n):
                    skipgram = " ".join(tokens[i] for i in combo)
                    skipgrams_list.append(skipgram)
                return skipgrams_list

            stop_words = set(stopwords.words('english'))
            lemmatizer = WordNetLemmatizer()

            # Extract n-grams or skip-grams based on user selection.
            all_ngrams = []
            for term in df["Search term"]:
                if extraction_method == "Contiguous n-grams":
                    all_ngrams.extend(extract_ngrams(term, n_value))
                else:
                    all_ngrams.extend(extract_skipgrams(term, n_value))

            ngram_counts = Counter(all_ngrams)
            filtered_ngrams = {ngram: count for ngram, count in ngram_counts.items() if count >= min_frequency}

            if not filtered_ngrams:
                st.warning("No n-grams found with the specified minimum frequency.")
                return

            df_ngrams = pd.DataFrame(filtered_ngrams.items(), columns=["N-gram", "Frequency"])

            search_term_to_ngrams = {}
            for term in df["Search term"]:
                if extraction_method == "Contiguous n-grams":
                    search_term_to_ngrams[term] = extract_ngrams(term, n_value)
                else:
                    search_term_to_ngrams[term] = extract_skipgrams(term, n_value)

            ngram_performance = {}
            for index, row in df.iterrows():
                search_term_text = row["Search term"]
                for ngram in search_term_to_ngrams[search_term_text]:
                    if ngram in filtered_ngrams:
                        if ngram not in ngram_performance:
                            ngram_performance[ngram] = {
                                "Clicks": 0,
                                "Impressions": 0,
                                "Cost": 0,
                                "Conversions": 0
                            }
                        ngram_performance[ngram]["Clicks"] += row["Clicks"]
                        ngram_performance[ngram]["Impressions"] += row["Impressions"]
                        ngram_performance[ngram]["Cost"] += row["Cost"]
                        ngram_performance[ngram]["Conversions"] += row["Conversions"]

            df_ngram_performance = pd.DataFrame.from_dict(ngram_performance, orient='index')
            df_ngram_performance.index.name = "N-gram"
            df_ngram_performance = df_ngram_performance.reset_index()

            df_ngram_performance["CTR"] = (df_ngram_performance["Clicks"] / df_ngram_performance["Impressions"]) * 100
            df_ngram_performance["Conversion Rate"] = (df_ngram_performance["Conversions"] / df_ngram_performance["Clicks"]) * 100
            df_ngram_performance["Cost per Conversion"] = df_ngram_performance.apply(
                lambda row: "None" if row["Conversions"] == 0 else row["Cost"] / row["Conversions"], axis=1
            )
            df_ngram_performance['Cost per Conversion'] = df_ngram_performance['Cost per Conversion'].apply(lambda x: pd.NA if x == 'None' else x)
            df_ngram_performance['Cost per Conversion'] = pd.to_numeric(df_ngram_performance['Cost per Conversion'], errors='coerce')

            # --- Updated Sorting Section ---
            # Default sorting: sort by "Conversions" in descending order.
            default_sort = "Conversions" if "Conversions" in df_ngram_performance.columns else df_ngram_performance.columns[0]
            sort_column = st.selectbox("Sort by Column:", options=df_ngram_performance.columns, index=list(df_ngram_performance.columns).index(default_sort))
            sort_ascending = st.checkbox("Sort Ascending", value=False)

            if sort_ascending:
                df_ngram_performance = df_ngram_performance.sort_values(by=sort_column, ascending=True, na_position='last')
            else:
                df_ngram_performance = df_ngram_performance.sort_values(by=sort_column, ascending=False, na_position='first')

            st.dataframe(df_ngram_performance.style.format({
                "Cost": "${:,.2f}",
                "Cost per Conversion": "${:,.2f}",
                "CTR": "{:,.2f}%",
                "Conversion Rate": "{:,.2f}%",
                "Conversions": "{:,.1f}"
            }))

        except Exception as e:
            st.error(f"An error occurred while processing the Excel file: {e}")

# ------------------------------------
# NEW TOOL: GSC Analyzer
# ------------------------------------

def google_search_console_analysis_page():
    st.header("Google Search Console Data Analysis")
    st.markdown(
        """
        The goal is to identify key topics that are contributing to your SEO performance.
        This tool lets you compare GSC query data from two different time periods. I recommend limiting to the top 1,000 queries as this can take awhile to process.
        Upload CSV files (one for the 'Before' period and one for the 'After' period), and the tool will:
        - Classify queries into topics with descriptive labels using LDA.
        - Display the original merged data table with topic labels.
        - Aggregate metrics by topic, with an option to display more rows.
        - Visualize the YOY % change by topic for each metric.
        """
    )

    st.markdown("### Upload GSC Data")
    uploaded_file_before = st.file_uploader("Upload GSC CSV for 'Before' period", type=["csv"], key="gsc_before")
    uploaded_file_after = st.file_uploader("Upload GSC CSV for 'After' period", type=["csv"], key="gsc_after")

    if uploaded_file_before is not None and uploaded_file_after is not None:
        # Initialize the progress bar
        progress_bar = st.progress(0)
        try:
            # Step 1: Read the original CSV files
            df_before = pd.read_csv(uploaded_file_before)
            df_after = pd.read_csv(uploaded_file_after)
            progress_bar.progress(10)

            # Step 2: Check required columns
            if "Top queries" not in df_before.columns or "Position" not in df_before.columns:
                st.error("The 'Before' CSV must contain 'Top queries' and 'Position' columns.")
                return
            if "Top queries" not in df_after.columns or "Position" not in df_after.columns:
                st.error("The 'After' CSV must contain 'Top queries' and 'Position' columns.")
                return
            progress_bar.progress(15)

            # --- Dashboard Summary using original data ---
            st.markdown("## Dashboard Summary")
            # Rename columns in original data for consistency
            df_before.rename(columns={"Top queries": "Query", "Position": "Average Position"}, inplace=True)
            df_after.rename(columns={"Top queries": "Query", "Position": "Average Position"}, inplace=True)
            progress_bar.progress(20)

            cols = st.columns(4)
            if "Clicks" in df_before.columns and "Clicks" in df_after.columns:
                total_clicks_before = df_before["Clicks"].sum()
                total_clicks_after = df_after["Clicks"].sum()
                overall_clicks_change = total_clicks_after - total_clicks_before
                overall_clicks_change_pct = (overall_clicks_change / total_clicks_before * 100) if total_clicks_before != 0 else 0
                cols[0].metric(label="Clicks Change", value=f"{overall_clicks_change:,.0f}", delta=f"{overall_clicks_change_pct:.1f}%")
            else:
                cols[0].metric(label="Clicks Change", value="N/A")

            if "Impressions" in df_before.columns and "Impressions" in df_after.columns:
                total_impressions_before = df_before["Impressions"].sum()
                total_impressions_after = df_after["Impressions"].sum()
                overall_impressions_change = total_impressions_after - total_impressions_before
                overall_impressions_change_pct = (overall_impressions_change / total_impressions_before * 100) if total_impressions_before != 0 else 0
                cols[1].metric(label="Impressions Change", value=f"{overall_impressions_change:,.0f}", delta=f"{overall_impressions_change_pct:.1f}%")
            else:
                cols[1].metric(label="Impressions Change", value="N/A")

            overall_avg_position_before = df_before["Average Position"].mean()
            overall_avg_position_after = df_after["Average Position"].mean()
            overall_position_change = overall_avg_position_before - overall_avg_position_after
            overall_position_change_pct = (overall_position_change / overall_avg_position_before * 100) if overall_avg_position_before != 0 else 0
            cols[2].metric(label="Avg. Position Change", value=f"{overall_position_change:.1f}", delta=f"{overall_position_change_pct:.1f}%")

            if "CTR" in df_before.columns and "CTR" in df_after.columns:
                def parse_ctr(ctr):
                    try:
                        if isinstance(ctr, str) and "%" in ctr:
                            return float(ctr.replace("%", ""))
                        else:
                            return float(ctr)
                    except:
                        return None
                df_before["CTR_parsed"] = df_before["CTR"].apply(parse_ctr)
                df_after["CTR_parsed"] = df_after["CTR"].apply(parse_ctr)
                overall_ctr_before = df_before["CTR_parsed"].mean()
                overall_ctr_after = df_after["CTR_parsed"].mean()
                overall_ctr_change = overall_ctr_after - overall_ctr_before
                overall_ctr_change_pct = (overall_ctr_change / overall_ctr_before * 100) if overall_ctr_before != 0 else 0
                cols[3].metric(label="CTR Change", value=f"{overall_ctr_change:.2f}", delta=f"{overall_ctr_change_pct:.1f}%")
            else:
                cols[3].metric(label="CTR Change", value="N/A")
            progress_bar.progress(30)

            # Step 3: Merge Data for Further Analysis
            merged_df = pd.merge(df_before, df_after, on="Query", suffixes=("_before", "_after"))
            progress_bar.progress(35)

            # Calculate YOY changes from merged data
            merged_df["Position_YOY"] = merged_df["Average Position_before"] - merged_df["Average Position_after"]
            if "Clicks" in df_before.columns and "Clicks" in df_after.columns:
                merged_df["Clicks_YOY"] = merged_df["Clicks_after"] - merged_df["Clicks_before"]
            if "Impressions" in df_before.columns and "Impressions" in df_after.columns:
                merged_df["Impressions_YOY"] = merged_df["Impressions_after"] - merged_df["Impressions_before"]
            if "CTR" in df_before.columns and "CTR" in df_after.columns:
                merged_df["CTR_before"] = merged_df["CTR_before"].apply(parse_ctr)
                merged_df["CTR_after"] = df_after["CTR_parsed"].apply(parse_ctr) #Corrected to use parsed CTR
                merged_df["CTR_YOY"] = merged_df["CTR_after"] - merged_df["CTR_before"]

            # Calculate YOY percentage changes
            merged_df["Position_YOY_pct"] = merged_df.apply(lambda row: (row["Position_YOY"] / row["Average Position_before"] * 100)
                                                            if row["Average Position_before"] and row["Average Position_before"] != 0 else None, axis=1)
            if "Clicks" in df_before.columns:
                merged_df["Clicks_YOY_pct"] = merged_df.apply(lambda row: (row["Clicks_YOY"] / row["Clicks_before"] * 100)
                                                              if row["Clicks_before"] and row["Clicks_before"] != 0 else None, axis=1)
            if "Impressions" in df_before.columns:
                merged_df["Impressions_YOY_pct"] = merged_df.apply(lambda row: (row["Impressions_YOY"] / row["Impressions_before"] * 100)
                                                                   if row["Impressions_before"] and row["Impressions_before"] != 0 else None, axis=1)
            if "CTR" in df_before.columns:
                merged_df["CTR_YOY_pct"] = merged_df.apply(lambda row: (row["CTR_YOY"] / row["CTR_before"] * 100)
                                                           if row["CTR_before"] and row["CTR_before"] != 0 else None, axis=1)

            # Rearrange merged_df columns for display
            base_cols = ["Query", "Average Position_before", "Average Position_after", "Position_YOY", "Position_YOY_pct"]
            if "Clicks" in df_before.columns:
                base_cols += ["Clicks_before", "Clicks_after", "Clicks_YOY", "Clicks_YOY_pct"]
            if "Impressions" in df_before.columns:
                base_cols += ["Impressions_before", "Impressions_after", "Impressions_YOY", "Impressions_YOY_pct"]
            if "CTR" in df_before.columns:
                base_cols += ["CTR_before", "CTR_after", "CTR_YOY", "CTR_YOY_pct"]
            merged_df = merged_df[base_cols]
            progress_bar.progress(40)

            # --- Define formatting for merged data table display ---
            format_dict_merged = {}
            if "Average Position_before" in merged_df.columns:
                format_dict_merged["Average Position_before"] = "{:.1f}"
            if "Average Position_after" in merged_df.columns:
                format_dict_merged["Average Position_after"] = "{:.1f}"
            if "Position_YOY" in merged_df.columns:
                format_dict_merged["Position_YOY"] = "{:.1f}"
            if "Clicks_before" in merged_df.columns:
                format_dict_merged["Clicks_before"] = "{:,.0f}"
            if "Clicks_after" in merged_df.columns:
                format_dict_merged["Clicks_after"] = "{:,.0f}"
            if "Clicks_YOY" in merged_df.columns:
                format_dict_merged["Clicks_YOY"] = "{:,.0f}"
            if "Impressions_before" in merged_df.columns:
                format_dict_merged["Impressions_before"] = "{:,.0f}"
            if "Impressions_after" in merged_df.columns:
                format_dict_merged["Impressions_after"] = "{:,.0f}"
            if "Impressions_YOY" in merged_df.columns:
                format_dict_merged["Impressions_YOY"] = "{:,.0f}"
            if "CTR_before" in merged_df.columns:
                format_dict_merged["CTR_before"] = "{:.2f}%"
            if "CTR_after" in merged_df.columns:
                format_dict_merged["CTR_after"] = "{:.2f}%"
            if "CTR_YOY" in merged_df.columns:
                format_dict_merged["CTR_YOY"] = "{:.2f}%"
            if "Position_YOY_pct" in merged_df.columns:
                format_dict_merged["Position_YOY_pct"] = "{:.2f}%"
            if "Clicks_YOY_pct" in merged_df.columns:
                format_dict_merged["Clicks_YOY_pct"] = "{:.2f}%"
            if "Impressions_YOY_pct" in merged_df.columns:
                format_dict_merged["Impressions_YOY_pct"] = "{:.2f}%"
            if "CTR_YOY_pct" in merged_df.columns:
                format_dict_merged["CTR_YOY_pct"] = "{:.2f}%"
            # --- End define format_dict_merged ---

            # Step 4: Topic Classification using LDA
            st.markdown("### Topic Classification and Combined Data")
            st.markdown("Here is the original merged data table with added topic labels for each search query.") # Added description
            st.markdown("### Topic Modeling of Search Queries (LDA)")
            n_topics_gsc_lda = st.slider("Select number of topics for Query LDA:", min_value=2, max_value=15, value=5, key="lda_topics_gsc") # UI for number of topics

            queries = merged_df["Query"].tolist() # Get queries for LDA
            with st.spinner(f"Performing LDA Topic Modeling on search queries ({n_topics_gsc_lda} topics)..."):
                vectorizer_queries_lda = CountVectorizer(stop_words="english", max_df=0.95, min_df=2) # Vectorize queries
                query_matrix_lda = vectorizer_queries_lda.fit_transform(queries)
                feature_names_queries_lda = vectorizer_queries_lda.get_feature_names_out()

                lda_queries_model = LatentDirichletAllocation(n_components=n_topics_gsc_lda, random_state=42)
                lda_queries_model.fit(query_matrix_lda)

                query_topic_labels = lda_queries_model.transform(query_matrix_lda).argmax(axis=1) # Assign topic label to each query
                merged_df["Query_Topic_Label"] = query_topic_labels # Add topic labels to dataframe

                topic_labels_desc_queries = {} # Generate descriptive labels for query topics
                for topic in range(n_topics_gsc_lda):
                    topic_queries_lda = merged_df[merged_df["Query_Topic_Label"] == topic]["Query"].tolist()
                    topic_labels_desc_queries[topic] = generate_topic_label(topic_queries_lda) # Reuse generate_topic_label if suitable or create a new one
                merged_df["Query_Topic"] = merged_df["Query_Topic_Label"].apply(lambda x: topic_labels_desc_queries.get(x, f"Topic {x+1}")) # Apply topic descriptions

                st.write("Identified Query Topics:")
                for topic_idx, topic in enumerate(lda_queries_model.components_):
                    top_keyword_indices = topic.argsort()[-10:][::-1]
                    topic_keywords = [feature_names_queries_lda[i] for i in top_keyword_indices]
                    st.write(f"**Query Topic {topic_idx + 1}:** {', '.join(topic_keywords)}")

            progress_bar.progress(50) # Update progress bar

            # --- Display Merged Data Table with Topic Labels ---
            merged_df_display = merged_df[["Query", "Query_Topic"] + base_cols[1:]] #Reorder columns for display - Topic first
            format_dict_merged_display = format_dict_merged.copy() # Copy existing format dict
            st.dataframe(merged_df_display.style.format(format_dict_merged_display)) # Display merged_df with formatting


            # Step 5: Aggregated Metrics by Topic
            st.markdown("### Aggregated Metrics by Topic")
            agg_dict = {
                "Average Position_before": "mean",
                "Average Position_after": "mean",
                "Position_YOY": "mean"
            }
            if "Clicks_before" in merged_df.columns:
                agg_dict.update({
                    "Clicks_before": "sum",
                    "Clicks_after": "sum",
                    "Clicks_YOY": "sum"
                })
            if "Impressions_before" in merged_df.columns:
                agg_dict.update({
                    "Impressions_before": "sum",
                    "Impressions_after": "sum",
                    "Impressions_YOY": "sum"
                })
            if "CTR_before" in merged_df.columns:
                agg_dict.update({
                    "CTR_before": "mean",
                    "CTR_after": "mean",
                    "CTR_YOY": "mean"
                })
            aggregated = merged_df.groupby("Query_Topic").agg(agg_dict).reset_index() # Group by Query_Topic now
            aggregated.rename(columns={"Query_Topic": "Topic"}, inplace=True) # Rename for consistency


            # Calculate aggregated YOY percentage changes
            aggregated["Position_YOY_pct"] = aggregated.apply(
                lambda row: (row["Position_YOY"] / row["Average Position_before"] * 100)
                if row["Average Position_before"] and row["Average Position_before"] != 0 else None, axis=1)
            if "Clicks_before" in aggregated.columns:
                aggregated["Clicks_YOY_pct"] = aggregated.apply(
                    lambda row: (row["Clicks_YOY"] / row["Clicks_before"] * 100)
                    if row["Clicks_before"] and row["Clicks_before"] != 0 else None, axis=1)
            if "Impressions_before" in aggregated.columns:
                aggregated["Impressions_YOY_pct"] = aggregated.apply(
                    lambda row: (row["Impressions_YOY"] / row["Impressions_before"] * 100)
                    if row["Impressions_before"] and row["Impressions_before"] != 0 else None, axis=1)
            if "CTR_before" in aggregated.columns:
                aggregated["CTR_YOY_pct"] = aggregated.apply(
                    lambda row: (row["CTR_YOY"] / row["CTR_before"] * 100)
                    if row["CTR_before"] and row["CTR_before"] != 0 else None, axis=1)
            progress_bar.progress(75)

            # Reorder columns so that each % Change is immediately next to its related metric columns.
            new_order = ["Topic"]
            if "Average Position_before" in aggregated.columns:
                new_order.extend(["Average Position_before", "Average Position_after", "Position_YOY", "Position_YOY_pct"])
            if "Clicks_before" in aggregated.columns:
                new_order.extend(["Clicks_before", "Clicks_after", "Clicks_YOY", "Clicks_YOY_pct"])
            if "Impressions_before" in aggregated.columns:
                new_order.extend(["Impressions_before", "Impressions_after", "Impressions_YOY", "Impressions_YOY_pct"])
            if "CTR_before" in aggregated.columns:
                new_order.extend(["CTR_before", "CTR_after", "CTR_YOY", "CTR_YOY_pct"])
            aggregated = aggregated[new_order]

            # Define formatting for aggregated metrics display
            format_dict = {}
            if "Average Position_before" in aggregated.columns:
                format_dict["Average Position_before"] = "{:.1f}"
            if "Average Position_after" in aggregated.columns:
                format_dict["Average Position_after"] = "{:.1f}"
            if "Position_YOY" in aggregated.columns:
                format_dict["Position_YOY"] = "{:.1f}"
            if "Position_YOY_pct" in aggregated.columns:
                format_dict["Position_YOY_pct"] = "{:.2f}%"
            if "Clicks_before" in aggregated.columns:
                format_dict["Clicks_before"] = "{:,.0f}"
            if "Clicks_after" in aggregated.columns:
                format_dict["Clicks_after"] = "{:,.0f}"
            if "Clicks_YOY" in aggregated.columns:
                format_dict["Clicks_YOY"] = "{:,.0f}"
            if "Clicks_YOY_pct" in aggregated.columns:
                format_dict["Clicks_YOY_pct"] = "{:.2f}%"
            if "Impressions_before" in aggregated.columns:
                format_dict["Impressions_before"] = "{:,.0f}"
            if "Impressions_after" in aggregated.columns:
                format_dict["Impressions_after"] = "{:,.0f}"
            if "Impressions_YOY" in aggregated.columns:
                format_dict["Impressions_YOY"] = "{:,.0f}"
            if "Impressions_YOY_pct" in aggregated.columns:
                format_dict["Impressions_YOY_pct"] = "{:.2f}%"
            if "CTR_before" in aggregated.columns:
                format_dict["CTR_before"] = "{:.2f}%"
            if "CTR_after" in aggregated.columns:
                format_dict["CTR_after"] = "{:.2f}%"
            if "CTR_YOY" in aggregated.columns:
                format_dict["CTR_YOY"] = "{:.2f}%"
            if "CTR_YOY_pct" in aggregated.columns:
                format_dict["CTR_YOY_pct"] = "{:.2f}%"

            display_count = st.number_input("Number of aggregated topics to display:", min_value=1, value=aggregated.shape[0])
            st.dataframe(aggregated.head(display_count).style.format(format_dict))
            progress_bar.progress(80)

            # Step 6: Visualization - Grouped Bar Chart of YOY % Change by Topic for Each Metric
            st.markdown("### YOY % Change by Topic for Each Metric")
            import plotly.express as px

            # Allow user to disable specific topics from the chart
            available_topics = aggregated["Topic"].unique().tolist()
            selected_topics = st.multiselect("Select topics to display on the chart:", options=available_topics, default=available_topics)

            vis_data = []
            for idx, row in aggregated.iterrows():
                topic = row["Topic"]
                if topic not in selected_topics:
                    continue
                if "Position_YOY_pct" in aggregated.columns:
                    vis_data.append({"Topic": topic, "Metric": "Average Position", "YOY % Change": row["Position_YOY_pct"]})
                if "Clicks_YOY_pct" in aggregated.columns:
                    vis_data.append({"Topic": topic, "Metric": "Clicks", "YOY % Change": row["Clicks_YOY_pct"]})
                if "Impressions_YOY_pct" in aggregated.columns:
                    vis_data.append({"Topic": topic, "Metric": "Impressions", "YOY % Change": row["Impressions_YOY_pct"]})
                if "CTR_YOY_pct" in aggregated.columns:
                    vis_data.append({"Topic": topic, "Metric": "CTR", "YOY % Change": row["CTR_YOY_pct"]})
            vis_df = pd.DataFrame(vis_data)
            fig = px.bar(vis_df, x="Topic", y="YOY % Change", color="Metric", barmode="group",
                         title="YOY % Change by Topic for Each Metric",
                         labels={"YOY % Change": "YOY % Change (%)"})
            st.plotly_chart(fig)
            progress_bar.progress(100)

        except Exception as e:
            st.error(f"An error occurred while processing the files: {e}")
    else:
        st.info("Please upload both GSC CSV files to start the analysis.")

# ------------------------------------
# NEW TOOL: Vector Embeddings Scatterplot
# ------------------------------------
# Cache the SentenceTransformer model so it loads only once
@st.cache_resource
def load_sentence_transformer(model_name='all-MiniLM-L6-v2'):
    return SentenceTransformer(model_name)

def load_data(file):
    """Loads Screaming Frog crawl data from an uploaded CSV file."""
    df = pd.read_csv(file)
    if 'URL' not in df.columns or 'Content' not in df.columns:
        raise ValueError("CSV must contain 'URL' and 'Content' columns.")
    return df[['URL', 'Content']]

def vectorize_pages(contents, model):
    """Converts page content into vector embeddings using a transformer model."""
    embeddings = model.encode(contents, convert_to_numpy=True)
    return embeddings

# --------------------------
# Updated: Dimension Reduction Function using UMAP instead of PCA
# --------------------------
def reduce_dimensions(embeddings, n_components=2):
    """Reduces vector dimensionality using UMAP instead of PCA."""
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    reduced_embeddings = reducer.fit_transform(embeddings)
    return reduced_embeddings

def cluster_embeddings(embeddings, n_clusters=5):
    """Clusters embeddings using KMeans."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    return labels

def plot_embeddings_interactive(embeddings, labels, urls):
    """
    Creates an interactive UMAP scatter plot using Plotly Express.
    Hovering over a point displays the corresponding URL.
    """
    # Create a DataFrame with the coordinates, cluster labels, and URLs
    df_plot = pd.DataFrame({
        "x": embeddings[:, 0],
        "y": embeddings[:, 1],
        "Cluster": [f"Cluster {label}" for label in labels],
        "URL": urls
    })
    # Create an interactive scatter plot
    fig = px.scatter(
        df_plot,
        x="x",
        y="y",
        color="Cluster",
        hover_data=["URL"],
        title="Interactive UMAP Scatterplot of Website Pages"
    )
    return fig

def semantic_clustering_page():
    st.header("Site Focus Visualizer")
    st.markdown(
        """
        Upload your Screaming Frog CSV file containing your website's embeddings.
        The CSV must include **URL** and **Content** columns.
        """
    )
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            data = load_data(uploaded_file)
        except Exception as e:
            st.error(f"Error: {e}")
            return

        st.write("Data loaded successfully. Here is a preview:")
        st.dataframe(data.head())
        
        # Extract the URLs for use in the interactive plot
        urls = data['URL'].tolist()
        
        # Load the transformer model (cached)
        model = load_sentence_transformer()
        
        with st.spinner("Vectorizing page content..."):
            embeddings = vectorize_pages(data['Content'].tolist(), model)
        
        with st.spinner("Reducing dimensions using UMAP..."):
            reduced_embeddings = reduce_dimensions(embeddings)
        
        n_clusters = st.number_input("Select number of clusters:", min_value=2, max_value=20, value=5, step=1)
        with st.spinner("Clustering embeddings..."):
            labels = cluster_embeddings(reduced_embeddings, n_clusters)
        
        st.success("Clustering complete!")
        
        # Use the interactive Plotly function
        fig = plot_embeddings_interactive(reduced_embeddings, labels, urls)
        st.plotly_chart(fig)

# --------------------------
# Content Idea Generator
# --------------------------
def content_idea_generator_page():
    st.header("Content Idea Generator using SentenceTransformer")
    st.markdown(
        "This tool extracts content from one or more URLs and uses SentenceTransformer to generate new content ideas. "
        "Enter a main keyword to build off of—the tool will automatically rank candidate words from the combined content based on their similarity to the main keyword."
    )
    
    # User must enter the main keyword (no default)
    main_keyword = st.text_input("Enter the main keyword to build off of:")
    
    # Allow user to input multiple URLs (one per line) without a default URL.
    urls_input = st.text_area("Enter URLs to analyze (one per line):")
    urls = [url.strip() for url in urls_input.splitlines() if url.strip()]
    
    if st.button("Extract and Generate"):
        combined_text = ""
        with st.spinner("Extracting content from URLs..."):
            for url in urls:
                text = extract_relevant_text_from_url(url)
                if text:
                    combined_text += text + "\n"
                else:
                    st.warning(f"Content could not be extracted from {url}.")
        
        if not combined_text:
            st.error("No content extracted from the provided URLs.")
            return
        
        st.markdown("### Combined Extracted Content Preview")
        st.write(combined_text[:1000] + " ...")
        
        # Build a vocabulary from the combined content.
        import re
        tokens = re.findall(r'\b\w+\b', combined_text.lower())
        filtered_tokens = [word for word in tokens if word not in stop_words]
        vocabulary = list(set(filtered_tokens))
        
        # Compute word frequencies and extract the top 20 candidate words.
        from collections import Counter
        word_freq = Counter(filtered_tokens)
        candidate_words = [word for word, count in word_freq.most_common(20)]
        st.markdown(f"Extracted vocabulary of {len(vocabulary)} unique words. Top candidate words (by frequency): {', '.join(candidate_words)}")
        
        # Load the SentenceTransformer model.
        model = initialize_sentence_transformer()
        
        # Compute embeddings for the vocabulary.
        with st.spinner("Computing vocabulary embeddings..."):
            vocab_embeddings = model.encode(vocabulary, convert_to_numpy=True)
        
        from sklearn.metrics.pairwise import cosine_similarity
        
        # If a main keyword was provided, rank candidate words by similarity to it.
        if main_keyword:
            main_keyword_emb = model.encode(main_keyword, convert_to_numpy=True)
            candidate_embeddings = model.encode(candidate_words, convert_to_numpy=True)
            sim_scores = cosine_similarity([main_keyword_emb], candidate_embeddings)[0]
            ranked_candidates = sorted(zip(candidate_words, sim_scores), key=lambda x: x[1], reverse=True)
            ranked_candidate_words = [word for word, score in ranked_candidates]
            st.markdown(f"**Candidate words ranked by similarity to '{main_keyword}':** {', '.join(ranked_candidate_words)}")
        else:
            st.warning("Please enter a main keyword for ranking candidate words.")
            ranked_candidate_words = candidate_words
        
        # --- 1. Finding Similar Words ---
        st.markdown("## 1. Finding Similar Words")
        if ranked_candidate_words:
            input_word = st.selectbox("Select a word to find similar words:", options=ranked_candidate_words)
            input_embedding = model.encode(input_word, convert_to_numpy=True)
            similarities = cosine_similarity([input_embedding], vocab_embeddings)[0]
            similar_indices = similarities.argsort()[::-1]
            top_similar = []
            count = 0
            for idx in similar_indices:
                if vocabulary[idx] == input_word:
                    continue
                top_similar.append((vocabulary[idx], similarities[idx]))
                count += 1
                if count >= 10:
                    break
            st.markdown(f"**Words similar to '{input_word}':**")
            for word, sim in top_similar:
                st.write(f"- {word} (Similarity: {sim:.2f})")
        else:
            st.warning("No candidate words available for similar words analysis.")
        
        # --- 2. Word Vector Arithmetic ---
        st.markdown("## 2. Word Vector Arithmetic")
        if len(ranked_candidate_words) >= 2:
            word1 = st.selectbox("Select the first word for vector arithmetic:", options=ranked_candidate_words)
            default_index = 1 if ranked_candidate_words[0] == word1 and len(ranked_candidate_words) > 1 else 0
            word2 = st.selectbox("Select the second word for vector arithmetic:", options=ranked_candidate_words, index=default_index)
            emb1 = model.encode(word1, convert_to_numpy=True)
            emb2 = model.encode(word2, convert_to_numpy=True)
            arithmetic_vector = emb1 + emb2
            similarities_arith = cosine_similarity([arithmetic_vector], vocab_embeddings)[0]
            similar_indices_arith = similarities_arith.argsort()[::-1]
            top_arith = []
            count = 0
            for idx in similar_indices_arith:
                candidate_word = vocabulary[idx]
                if candidate_word in [word1, word2]:
                    continue
                top_arith.append((candidate_word, similarities_arith[idx]))
                count += 1
                if count >= 10:
                    break
            st.markdown(f"**Result of combining '{word1}' and '{word2}':**")
            for word, sim in top_arith:
                st.write(f"- {word} (Similarity: {sim:.2f})")
        else:
            st.warning("Not enough candidate words available for word vector arithmetic.")
        
        # --- 3. Phrase Analysis ---
        st.markdown("## 3. Phrase Analysis")
        selected_phrase_words = st.multiselect("Select words to form a phrase for analysis:", options=ranked_candidate_words)
        if not selected_phrase_words:
            st.error("Please select at least one word for phrase analysis.")
        else:
            phrase_input = " ".join(selected_phrase_words)
            st.markdown(f"**Analyzing phrase: '{phrase_input}'**")
            phrase_words = phrase_input.split()
            for word in phrase_words:
                word_emb = model.encode(word, convert_to_numpy=True)
                similarities_word = cosine_similarity([word_emb], vocab_embeddings)[0]
                similar_indices_word = similarities_word.argsort()[::-1]
                top_similar_word = []
                count = 0
                for idx in similar_indices_word:
                    if vocabulary[idx] == word:
                        continue
                    top_similar_word.append((vocabulary[idx], similarities_word[idx]))
                    count += 1
                    if count >= 5:
                        break
                st.markdown(f"**Similar words for '{word}':**")
                for sim_word, sim_val in top_similar_word:
                    st.write(f"- {sim_word} (Similarity: {sim_val:.2f})")



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
        [data-testid="stDecoration"] { display: none !important; }
        a[href*='streamlit.io/cloud'],
        div._profileContainer_gzau3_53 { display: none !important; }
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
        "Top/Bottom 10 Embeddings",
        "Entity Topic Gap Analysis",
        "Entity Visualizer",
        "Entity Frequency Charts",
        "Semantic Gap Analyzer",
        "Keyword Clustering",
        "People Also Asked",
        "Google Ads Search Term Analyzer",  # New tool
        "Google Search Console Analyzer",
        "Site Focus Visualizer",
        "Content Idea Generator"  # New option
    ])
    if tool == "URL Analysis Dashboard":
        url_analysis_dashboard_page()
    elif tool == "Cosine Similarity - Competitor Analysis":
        cosine_similarity_competitor_analysis_page()
    elif tool == "Cosine Similarity - Every Embedding":
        cosine_similarity_every_embedding_page()
    elif tool == "Cosine Similarity - Content Heatmap":
        cosine_similarity_content_heatmap_page()
    elif tool == "Top/Bottom 10 Embeddings":
        top_bottom_embeddings_page()
    elif tool == "Entity Topic Gap Analysis":
        entity_analysis_page()
    elif tool == "Entity Visualizer":
        displacy_visualization_page()
    elif tool == "Entity Frequency Charts":
        named_entity_barchart_page()
    elif tool == "Semantic Gap Analyzer":
        ngram_tfidf_analysis_page()
    elif tool == "Keyword Clustering":
        keyword_clustering_from_gap_page()
    elif tool == "People Also Asked":
        paa_extraction_clustering_page()
    elif tool == "Google Ads Search Term Analyzer":
        google_ads_search_term_analyzer_page()
    elif tool == "Google Search Console Analyzer":
        google_search_console_analysis_page()
    elif tool == "Site Focus Visualizer":
        semantic_clustering_page()
    elif tool == "Content Idea Generator":
    content_idea_generator_page()
    st.markdown("---")
    st.markdown("Powered by [The SEO Consultant.ai](https://theseoconsultant.ai)", unsafe_allow_html=True)

if __name__ == "__main__":
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')
    main()









