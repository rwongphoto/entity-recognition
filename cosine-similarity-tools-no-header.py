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

import networkx as nx

from urllib.parse import urlparse

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
        user_agent = get_random_user_agent()
        chrome_options.add_argument(f"user-agent={user_agent}")
        
        # Generate a truly unique user data directory using uuid
        import uuid
        unique_dir = f"/tmp/chrome_profile_{uuid.uuid4()}"
        chrome_options.add_argument(f"--user-data-dir={unique_dir}")
        
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
        
        # Generate a unique user data directory using uuid
        import uuid
        unique_dir = f"/tmp/chrome_profile_{uuid.uuid4()}"
        chrome_options.add_argument(f"--user-data-dir={unique_dir}")
        
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
                    import tempfile
                    temp_dir = tempfile.mkdtemp()
                    chrome_options.add_argument(f"--user-data-dir={temp_dir}")
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
                         text='Competitor')
        fig.update_traces(textposition='top center')
        fig.update_layout(
            xaxis_title="Cosine Similarity (Higher = More Relevant)",
            yaxis_title="Content Length (Words)",
            width=800,
            height=600
        )
        st.plotly_chart(fig)
        st.dataframe(df)

        # --- Option 2: Bar Chart with Secondary Y-Axis for Content Length and Labels---
        df = pd.DataFrame({
            'Competitor': urls_plot,
            'Cosine Similarity': scores_plot,
            'Content Length (Words)': content_lengths
        })
        df = df.sort_values('Cosine Similarity', ascending=False)
        fig = go.Figure(data=[
            go.Bar(name='Cosine Similarity', x=df['Competitor'], y=df['Cosine Similarity'],
                   marker_color=df['Cosine Similarity'],
                   marker_colorscale='Viridis',
                    text=df['Competitor'],
                    textposition='outside'),
            go.Scatter(name='Content Length', x=df['Competitor'], y=df['Content Length (Words)'], yaxis='y2',
                       mode='lines+markers', marker=dict(color='red'))
        ])
        fig.update_traces(textfont_size=12)
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
                         size=[10] * len(df),
                         title='Competitor Analysis: Similarity, Content Length (Bubble Chart)',
                         hover_data=['Competitor', 'Cosine Similarity', 'Content Length (Words)'],
                         color='Cosine Similarity',
                         color_continuous_scale=px.colors.sequential.Viridis,
                         text='Competitor')
        fig.update_traces(textposition='top center')
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
            
            if target_option == "Extract from URL":
                target_text = extract_text_from_url(target_input) if target_input else ""
            else:
                target_text = target_input
            
            target_entities = identify_entities(target_text, nlp_model) if target_text else []
            filtered_target_entities = [(entity, label) for entity, label in target_entities if label not in exclude_types]
            target_entity_counts = count_entities(filtered_target_entities, nlp_model)
            target_entities_set = set(target_entity_counts.keys())
            exclude_entities_set = {ent.text.lower() for ent in nlp_model(exclude_input).ents} if exclude_input else set()
            
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
            
            gap_entities = Counter()
            for source, counts in entity_counts_per_source.items():
                for (entity, label), count in counts.items():
                    if (entity, label) not in target_entities_set:
                        gap_entities[(entity, label)] += count

            display_entity_barchart(gap_entities)

            st.markdown("### # of Sites Entities Are Present but Missing in Target")
            aggregated_gap_table = []
            aggregated_site_count = {}
            for source, counts in entity_counts_per_source.items():
                for (entity, label), count in counts.items():
                    if (entity, label) not in target_entities_set:
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
                    df_aggregated_gap = df_aggregated_gap.sort_values(by='# of Sites', ascending=False)
                    st.dataframe(df_aggregated_gap)
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

def ngram_tfidf_analysis_page():
    st.header("Semantic Gap Analyzer")
    st.markdown("""
        Uncover hidden opportunities by comparing your website's content to your top competitors.
        Identify key phrases and topics they're covering that you might be missing, and prioritize your content creation.
    """)
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

    st.subheader("Word Options")
    n_value = st.selectbox("Select # of Words in Phrase:", options=[1,2,3,4,5,6,7,8,9,10], index=1)
    st.markdown("*(For example, choose 2 for bigrams)*")
    min_df = st.number_input("Minimum Frequency:", value=1, min_value=1)
    max_df = st.number_input("Maximum Frequency:", value=1.0, min_value=0.0, step=0.1)
    top_n = st.slider("Number of top results to display:", min_value=1, max_value=50, value=10)
    num_topics = st.slider("Number of topics for LDA:", min_value=2, max_value=15, value=5, key="lda_topics")

    if st.button("Analyze Content Gaps", key="content_gap_button"):
        if competitor_source_option == "Extract from URL" and not competitor_list:
            st.warning("Please enter at least one competitor URL.")
            return
        if target_source_option == "Extract from URL" and not target_url:
            st.warning("Please enter your target URL.")
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

        with st.spinner("Calculating TF-IDF scores for competitors..."):
            vectorizer = TfidfVectorizer(ngram_range=(n_value, n_value), min_df=min_df, max_df=max_df, stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(competitor_texts)
            feature_names = vectorizer.get_feature_names_out()
            df_tfidf_competitors = pd.DataFrame(tfidf_matrix.toarray(), index=valid_competitor_sources, columns=feature_names)

        with st.spinner("Performing Latent Dirichlet Allocation (LDA)..."):
            lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
            lda_output = lda_model.fit_transform(tfidf_matrix)

            st.subheader("Identified Topics from Competitor Content (LDA)")
            topic_keywords = {}
            for i, topic in enumerate(lda_model.components_):
                top_keyword_indices = topic.argsort()[-10:][::-1]
                keywords = [feature_names[i] for i in top_keyword_indices]
                topic_keywords[f"Topic {i+1}"] = keywords
                st.markdown(f"**Topic {i+1}:** {', '.join(keywords)}")

            topic_distribution_df = pd.DataFrame(lda_output, index=valid_competitor_sources, columns=[f"Topic {i+1}" for i in range(num_topics)])
            st.dataframe(topic_distribution_df)

        with st.spinner("Calculating TF-IDF scores for target content..."):
            target_tfidf_vector = vectorizer.transform([target_content])
            df_tfidf_target = pd.DataFrame(target_tfidf_vector.toarray(), index=["Target Content"], columns=feature_names)

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
        tfidf_weight = 0.4

        norm_candidates = []
        for source, ngram, tfidf_diff, bert_diff in candidate_scores:
            norm_tfidf = (tfidf_diff - min_tfidf) / (max_tfidf - min_tfidf + epsilon)
            norm_bert = (bert_diff - min_bert) / (max_bert - min_bert + epsilon)
            combined_score = tfidf_weight * norm_tfidf + (1 - tfidf_weight) * norm_bert
            if combined_score > 0:
                norm_candidates.append((source, ngram, combined_score))
        norm_candidates.sort(key=lambda x: x[2], reverse=True)

        st.markdown("### Consolidated Semantic Gap Analysis (All Competitors)")
        df_consolidated = pd.DataFrame(norm_candidates, columns=['Competitor', 'N-gram', 'Gap Score'])
        st.dataframe(df_consolidated)

        st.markdown("### Per-Competitor Semantic Gap Analysis")
        for source in valid_competitor_sources:
            candidate_list = [ (s, n, score) for s, n, score in norm_candidates if s == source ]
            if candidate_list:
                df_source = pd.DataFrame(candidate_list, columns=['Competitor', 'N-gram', 'Gap Score']).sort_values(by='Gap Score', ascending=False)
                st.markdown(f"#### Competitor: {source}")
                st.dataframe(df_source)

        st.subheader("Combined Semantic Gap Wordcloud")
        gap_scores = {}
        for source, ngram, score in norm_candidates:
            gap_scores[ngram] = gap_scores.get(ngram, 0) + score
        if gap_scores:
            display_entity_wordcloud(gap_scores)
        else:
            st.write("No combined gap n-grams to create a wordcloud.")

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
        tfidf_weight = 0.4
        norm_candidates = []
        for ngram, tfidf_diff, bert_diff in candidate_scores:
            norm_tfidf = (tfidf_diff - min_tfidf) / (max_tfidf - min_tfidf + epsilon)
            norm_bert = (bert_diff - min_bert) / (max_bert - min_bert + epsilon)
            combined_score = tfidf_weight * norm_tfidf + (1 - tfidf_weight) * norm_bert
            if combined_score > 0:
                norm_candidates.append((ngram, combined_score))
        norm_candidates.sort(key=lambda x: x[1], reverse=True)
        
        st.markdown("### Keyword Clusters")
        clusters = {}
        for gram, label in zip([ng for ng, score in norm_candidates], KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit_predict(
                np.vstack([get_embedding(gram, model) for gram, score in norm_candidates])
            )):
            clusters.setdefault(label, []).append(gram)
        for label, gram_list in clusters.items():
            st.markdown(f"**Cluster {label}**:")
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

        def get_paa(query, max_depth=10):
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            user_agent = get_random_user_agent()
            chrome_options.add_argument(f"user-agent={user_agent}")
            import tempfile
            temp_dir = tempfile.mkdtemp()
            chrome_options.add_argument(f"--user-data-dir={temp_dir}")
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
            user_agent = get_random_user_agent()
            chrome_options.add_argument(f"user-agent={user_agent}")
            import tempfile
            temp_dir = tempfile.mkdtemp()
            chrome_options.add_argument(f"--user-data-dir={temp_dir}")
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

@st.cache_resource
def load_sentence_transformer(model_name='all-MiniLM-L6-v2'):
    return SentenceTransformer(model_name)

def load_data(file):
    df = pd.read_csv(file)
    if 'URL' not in df.columns or 'Content' not in df.columns:
        raise ValueError("CSV must contain 'URL' and 'Content' columns.")
    return df[['URL', 'Content']]

def vectorize_pages(contents, model):
    embeddings = model.encode(contents, convert_to_numpy=True)
    return embeddings

def reduce_dimensions(embeddings, n_components=2):
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    reduced_embeddings = reducer.fit_transform(embeddings)
    return reduced_embeddings

def cluster_embeddings(embeddings, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    return labels

def plot_embeddings_interactive(embeddings, labels, urls):
    df_plot = pd.DataFrame({
        "x": embeddings[:, 0],
        "y": embeddings[:, 1],
        "Cluster": [f"Cluster {label}" for label in labels],
        "URL": urls
    })
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
        
        urls = data['URL'].tolist()
        
        model = load_sentence_transformer()
        
        with st.spinner("Vectorizing page content..."):
            embeddings = vectorize_pages(data['Content'].tolist(), model)
        
        with st.spinner("Reducing dimensions using UMAP..."):
            reduced_embeddings = reduce_dimensions(embeddings)
        
        n_clusters = st.number_input("Select number of clusters:", min_value=2, max_value=20, value=5, step=1)
        with st.spinner("Clustering embeddings..."):
            labels = cluster_embeddings(reduced_embeddings, n_clusters)
        
        st.success("Clustering complete!")
        
        fig = plot_embeddings_interactive(reduced_embeddings, labels, urls)
        st.plotly_chart(fig)

def extract_entities_and_relationships(sentences, nlp):
    entities = []
    relationships = []
    entity_counts = Counter()

    for sentence in sentences:
        doc = nlp(sentence)
        for ent in doc.ents:
          if ent.label_ not in ("DATE", "TIME", "CARDINAL", "ORDINAL", "PERCENT", "MONEY", "QUANTITY"):
              entities.append((ent.text, ent.label_))
              entity_counts[ent.text] += 1

    for sentence in sentences:
        doc = nlp(sentence)
        sentence_entities = [ent.text for ent in doc.ents if ent.label_ not in ("DATE", "TIME", "CARDINAL", "ORDINAL", "PERCENT", "MONEY", "QUANTITY")]
        for i in range(len(sentence_entities)):
            for j in range(i + 1, len(sentence_entities)):
                relationships.append((sentence_entities[i], sentence_entities[j]))

    return entities, relationships, entity_counts

def create_entity_graph(entities, relationships, entity_counts):
    G = nx.Graph()
    for entity, entity_type in entities:
        G.add_node(entity, type=entity_type, count=entity_counts[entity])
    relationship_counts = Counter(relationships)
    for (entity1, entity2), count in relationship_counts.items():
        G.add_edge(entity1, entity2, weight=count)
    return G

def visualize_graph(G, website_url):
    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(G, seed=42, k=0.5)
    node_sizes = [G.nodes[node]['count'] * 500 for node in G.nodes()]
    node_colors = []
    for node in G.nodes():
      if G.nodes[node]['type'] == 'ORG':
        node_colors.append('skyblue')
      elif G.nodes[node]['type'] == 'GPE':
        node_colors.append('lightgreen')
      elif G.nodes[node]['type'] == 'LOC':
        node_colors.append('lightcoral')
      elif G.nodes[node]['type'] == 'WORK_OF_ART':
        node_colors.append('plum')
      elif G.nodes[node]['type'] == 'PRODUCT':
        node_colors.append('palegoldenrod')
      else:
        node_colors.append('lightgray')
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)
    nx.draw_networkx_edges(G, pos, width=[data['weight'] for _, _, data in G.edges(data=True)])
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")
    title = f"Entity Relationship Graph for: {website_url}"
    plt.title(title, fontsize=16)
    plt.axis("off")
    st.pyplot(plt)

def entity_relationship_graph_page():
    st.header("Entity Relationship Graph Generator")
    url = st.text_input("Enter a website URL:", "")
    if url:
        with st.spinner(f"Scraping content from {url}..."):
            text = extract_text_from_url(url)
            if text:
                sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
                sentences = [s.strip() for s in sentences if s.strip()]
            else:
                st.warning(f"Could not retrieve content from {url}.")
                return
        if sentences:
            with st.spinner("Extracting entities and relationships..."):
                nlp_model = load_spacy_model()
                entities, relationships, entity_counts = extract_entities_and_relationships(sentences, nlp_model)
                graph = create_entity_graph(entities, relationships, entity_counts)
            with st.spinner("Visualizing graph..."):
                visualize_graph(graph, url)
        else:
            st.warning("No content was retrieved from the URL.")

def semrush_organic_pages_by_subdirectory_page():
    st.header("SEMRush Organic Pages by Subdirectory")
    st.markdown("""
    Upload your SEMRush Organic Pages report (Excel format) to see data aggregated by top-level sub-directory.  
    The file should contain a 'URL' column plus any numeric columns (e.g. 'Traffic', 'Number of Keywords', etc.).
    """)
    uploaded_file = st.file_uploader(
        "Upload SEMRush Organic Pages Excel file",
        type=["xlsx"],
        key="semrush_file"
    )
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error reading the Excel file: {e}")
            return
        if "URL" not in df.columns:
            st.error("No 'URL' column found in the file.")
            return
        numeric_cols = []
        for col in df.columns:
            if col == "URL":
                continue
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].notnull().sum() > 0:
                numeric_cols.append(col)
        def get_subdirectory(url):
            try:
                parsed = urlparse(url)
                path_segments = [seg for seg in parsed.path.split('/') if seg]
                return path_segments[0] if path_segments else "Root"
            except:
                return "Invalid URL"
        df["Subdirectory"] = df["URL"].apply(get_subdirectory)
        st.markdown("### Data Preview")
        st.dataframe(df.head())
        st.markdown("### Aggregated Metrics by Subdirectory")
        agg_dict = {col: "sum" for col in numeric_cols}
        subdir_agg = df.groupby("Subdirectory").agg(agg_dict).reset_index()
        st.dataframe(subdir_agg)
        if "Traffic" in numeric_cols:
            fig = px.bar(
                subdir_agg,
                x="Subdirectory",
                y="Traffic",
                title="Traffic by Subdirectory",
                labels={"Subdirectory": "Subdirectory", "Traffic": "Traffic"}
            )
            st.plotly_chart(fig)
        else:
            st.write("No 'Traffic' column found to plot.")
    else:
        st.info("Please upload a SEMRush Organic Pages Excel file to begin the analysis.")

def semrush_hierarchical_subdirectories_minimal_no_leaf_with_intent_filter():
    st.header("SEMRush Hierarchical Subdirectory Aggregation (Keywords & Traffic, No Leaf Nodes)")
    st.markdown("""
    **Goal:**  
    1. Keep only the **URL**, **Number of Keywords**, and **Traffic** columns along with any user intent traffic columns.  
    2. Expand each URL into **all** hierarchical subdirectories and omit any leaf nodes (subdirectories without deeper levels).  
    3. Aggregate (sum) the metrics at each non‑leaf hierarchical level.  
    4. Optionally filter the Plotly chart by user intent traffic columns.
    """)
    uploaded_file = st.file_uploader(
        "Upload an Excel file with 'URL', 'Number of Keywords', 'Traffic' and (optionally) user intent traffic columns",
        type=["xlsx"]
    )
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error reading Excel file: {e}")
            return
        required_cols = ["URL", "Number of Keywords", "Traffic"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return
        user_intent_options = [
            "Traffic with commercial intents in top 20",
            "Traffic with informational intents in top 20",
            "Traffic with navigational intents in top 20",
            "Traffic with transactional intents in top 20",
            "Traffic with unknown intents in top 20"
        ]
        available_intent_cols = [col for col in user_intent_options if col in df.columns]
        all_cols = required_cols + available_intent_cols
        df = df[all_cols]
        for col in df.columns:
            if col != "URL":
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        def get_subdirectory_levels(url):
            parsed = urlparse(str(url))
            segments = [seg for seg in parsed.path.strip("/").split("/") if seg]
            if not segments:
                return ["/"]
            paths = []
            for i in range(len(segments)):
                path = "/" + "/".join(segments[:i+1])
                paths.append(path)
            return paths
        exploded_rows = []
        for _, row in df.iterrows():
            url_levels = get_subdirectory_levels(row["URL"])
            for level in url_levels:
                new_row = row.copy()
                new_row["Hierarchical_Path"] = level
                exploded_rows.append(new_row)
        df_exploded = pd.DataFrame(exploded_rows)
        all_paths = set(df_exploded["Hierarchical_Path"].unique())
        def is_leaf(path):
            prefix = path.rstrip("/") + "/"
            return not any(candidate.startswith(prefix) for candidate in all_paths if candidate != path)
        df_exploded["IsLeaf"] = df_exploded["Hierarchical_Path"].apply(is_leaf)
        df_filtered = df_exploded[~df_exploded["IsLeaf"]]
        st.markdown("### Expanded Data (After Removing Leaf Nodes)")
        st.dataframe(df_filtered.head())
        numeric_cols = [col for col in df.columns if col != "URL"]
        df_agg = df_filtered.groupby("Hierarchical_Path")[numeric_cols].sum().reset_index()
        st.markdown("### Aggregated Data by Hierarchical Subdirectory (No Leaf Nodes)")
        st.dataframe(df_agg)
        st.markdown("### Plotly Chart")
        st.write("By default, a bar chart for overall 'Traffic' is shown. You can also filter by user intent traffic columns.")
        if available_intent_cols:
            selected_intents = st.multiselect(
                "Select User Intent Traffic Columns to plot:",
                options=available_intent_cols,
                default=[]
            )
        else:
            selected_intents = []
        if selected_intents:
            df_melt = df_agg.melt(id_vars=["Hierarchical_Path"],
                                  value_vars=selected_intents,
                                  var_name="Intent Type",
                                  value_name="Intent Traffic")
            fig = px.bar(
                df_melt,
                x="Hierarchical_Path",
                y="Intent Traffic",
                color="Intent Type",
                barmode="group",
                title="User Intent Traffic by Hierarchical Subdirectory (No Leaf Nodes)",
                labels={"Hierarchical_Path": "Subdirectory"}
            )
            fig.update_layout(height=800)
            st.plotly_chart(fig)
        else:
            if "Traffic" in df_agg.columns:
                fig = px.bar(
                    df_agg,
                    x="Hierarchical_Path",
                    y="Traffic",
                    title="Overall Traffic by Hierarchical Subdirectory (No Leaf Nodes)",
                    labels={"Hierarchical_Path": "Subdirectory", "Traffic": "Traffic"}
                )
                fig.update_layout(height=800)
                st.plotly_chart(fig)
            else:
                st.write("No 'Traffic' column found for plotting.")
    else:
        st.info("Please upload an Excel file to begin the analysis.")

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
        "Google Ads Search Term Analyzer",
        "Google Search Console Analyzer",
        "Site Focus Visualizer",
        "Entity Relationship Graph",
        "SEMRush Organic Pages by Top Sub-Directory",
        "SEMRush - Sub-Directories (No Leaf Nodes)"
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
        # (Google Ads Search Term Analyzer code here)
        st.write("Google Ads Search Term Analyzer functionality goes here.")
    elif tool == "Google Search Console Analyzer":
        # (Google Search Console Analyzer code here)
        st.write("Google Search Console Analyzer functionality goes here.")
    elif tool == "Site Focus Visualizer":
        semantic_clustering_page()
    elif tool == "Entity Relationship Graph":
        entity_relationship_graph_page()
    elif tool == "SEMRush Organic Pages by Top Sub-Directory":
        semrush_organic_pages_by_subdirectory_page()
    elif tool == "SEMRush - Sub-Directories (No Leaf Nodes)":
        semrush_hierarchical_subdirectories_minimal_no_leaf_with_intent_filter()
    st.markdown("---")
    st.markdown("Powered by [The SEO Consultant.ai](https://theseoconsultant.ai)", unsafe_allow_html=True)

if __name__ == "__main__":
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')
    main()

