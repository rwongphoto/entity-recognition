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
        user_agent = ("Mozilla/5.0 (iPhone; CPU iPhone OS 14_7_1 like Mac OS X) "
                      "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.7.1 Mobile/15E148 Safari/604.1")
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
        user_agent = ("Mozilla/5.0 (iPhone; CPU iPhone OS 14_7_1 like Mac OS X) "
                      "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.7.1 Mobile/15E148 Safari/604.1")
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

def identify_entities(text, nlp_model):
    doc = nlp_model(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
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
                    user_agent = ("Mozilla/5.0 (iPhone; CPU iPhone OS 14_7_1 like Mac OS X) "
                                  "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.7.1 Mobile/15E148 Safari/604.1")
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
            urls_plot = [url for url, score in similarity_scores]
            scores_plot = [score if score is not None else 0 for url, score in similarity_scores]
        else:
            if not competitor_contents:
                st.warning("Please paste at least one content block.")
                return
            with st.spinner("Calculating similarities from pasted content..."):
                similarity_scores = []
                for idx, content in enumerate(competitor_contents):
                    text_embedding = get_embedding(content, model)
                    search_embedding = get_embedding(search_term, model)
                    similarity = cosine_similarity([text_embedding], [search_embedding])[0][0]
                    similarity_scores.append((f"Competitor {idx+1}", similarity))
            urls_plot = [label for label, score in similarity_scores]
            scores_plot = [score for label, score in similarity_scores]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(urls_plot, scores_plot)
        ax.set_xlabel("Competitors")
        ax.set_ylabel("Similarity Score")
        ax.set_title("Cosine Similarity of Competitor Content to Search Term")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        data = {'Competitor': urls_plot, 'Similarity Score': scores_plot}
        df = pd.DataFrame(data)
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
    exclude_types = st.multiselect("Select entity types to exclude:",
                                   options=["CARDINAL", "DATE", "TIME", "PERCENT", "MONEY",
                                            "QUANTITY", "ORDINAL", "GPE", "ORG", "PERSON", "NORP",
                                            "FAC", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART",
                                            "LAW", "LANGUAGE", "MISC"],
                                   default=["CARDINAL", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL"])
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
            all_competitor_entities = []
            entity_counts_per_source = {}
            for source in competitor_list:
                if competitor_source_option == "Extract from URL":
                    text = extract_text_from_url(source)
                else:
                    text = source
                if text:
                    entities = identify_entities(text, nlp_model)
                    filtered_entities = [(entity, label) for entity, label in entities
                                         if label not in exclude_types and entity.lower() not in exclude_entities_set]
                    source_counts = count_entities(filtered_entities, nlp_model)
                    entity_counts_per_source[source] = source_counts
                    for (lemma, label) in source_counts:
                        all_competitor_entities.append((lemma, label))
            competitor_entity_counts = Counter(all_competitor_entities)
            gap_entities = Counter()
            for (entity, label), count in competitor_entity_counts.items():
                if (entity, label) not in target_entities_set:
                    gap_entities[(entity, label)] = count
            unique_target_entities = Counter()
            for (entity, label), count in target_entity_counts.items():
                if (entity, label) not in competitor_entity_counts:
                    unique_target_entities[(entity, label)] = count
            st.markdown("### # of Sites Entities Are Present but Missing in Target")
            if gap_entities:
                for (entity, label), count in gap_entities.most_common(50):
                    st.write(f"- {entity} ({label}): {count}")
                display_entity_barchart(gap_entities)
            else:
                st.write("No significant gap entities found.")
            st.markdown("### Entities Unique to Target Site")
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

        user_agent = ("Mozilla/5.0 (iPhone; CPU iPhone OS 14_7_1 like Mac OS X) "
                      "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.7.1 Mobile/15E148 Safari/604.1")
        
        def get_paa(query, max_depth=10):
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
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
            chrome_options.add_argument(f"user-agent={user_agent}")
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
        - Classify queries into topics with descriptive labels.
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
                merged_df["CTR_after"] = merged_df["CTR_after"].apply(parse_ctr)
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
            
            # Step 4: Topic Classification
            st.markdown("### Topic Classification and Combined Data")
            model = initialize_sentence_transformer()
            queries = merged_df["Query"].tolist()
            num_topics = st.slider("Select number of topics:", min_value=2, max_value=25, value=5, key="num_topics")
            # Compute embeddings and cluster queries using KMeans
            from sklearn.cluster import KMeans
            embeddings = [get_embedding(query, model) for query in queries]
            kmeans = KMeans(n_clusters=num_topics, random_state=42, n_init='auto')
            topic_labels = kmeans.fit_predict(embeddings)
            merged_df["Topic_Label"] = topic_labels
            
            # Generate descriptive topic labels by extracting common keywords per topic
            import collections
            import nltk
            nltk.download('stopwords')
            stop_words = set(nltk.corpus.stopwords.words('english'))
            
            def generate_topic_label(queries_in_topic):
                words = []
                for query in queries_in_topic:
                    tokens = query.lower().split()
                    filtered = [t for t in tokens if t not in stop_words]
                    words.extend(filtered)
                if words:
                    freq = collections.Counter(words)
                    common = freq.most_common(2)
                    label = ", ".join([word for word, count in common])
                    return label.capitalize()
                else:
                    return "N/A"
            
            topic_labels_desc = {}
            for topic in range(num_topics):
                topic_queries = merged_df[merged_df["Topic_Label"] == topic]["Query"].tolist()
                topic_labels_desc[topic] = generate_topic_label(topic_queries)
            merged_df["Topic"] = merged_df["Topic_Label"].apply(lambda x: topic_labels_desc.get(x, f"Topic {x+1}"))
            progress_bar.progress(55)
            
            # Define formatting for the "Topic Classification and Combined Data" table
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
            
            st.dataframe(merged_df.style.format(format_dict_merged))
            progress_bar.progress(60)
            
            # --- Hidden Initial Apriori Analysis ---
            with st.expander("Show Initial Apriori Analysis on Query Terms", expanded=False):
                st.markdown("### Initial Apriori Analysis on Query Terms")
                transactions = merged_df["Query"].apply(lambda x: str(x).lower().split()).tolist()
                from mlxtend.preprocessing import TransactionEncoder
                te = TransactionEncoder()
                te_ary = te.fit(transactions).transform(transactions)
                df_transactions = pd.DataFrame(te_ary, columns=te.columns_)
                from mlxtend.frequent_patterns import apriori
                freq_items = apriori(df_transactions, min_support=0.1, use_colnames=True)
                st.dataframe(freq_items.sort_values(by="support", ascending=False))
            progress_bar.progress(65)
            
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
            aggregated = merged_df.groupby("Topic").agg(agg_dict).reset_index()
            
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
        "Google Search Console Analyzer" # Add this line
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
    st.markdown("---")
    st.markdown("Powered by [The SEO Consultant.ai](https://theseoconsultant.ai)", unsafe_allow_html=True)

if __name__ == "__main__":
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')
    main()







