import streamlit as st
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from PIL import Image
import cairosvg
import json

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, WebDriverException
import matplotlib.pyplot as plt
import pandas as pd
import spacy
from collections import Counter
from typing import List, Tuple, Dict

# For Topic Modeling with Gensim's LDA Mallet
from gensim import corpora
from gensim.models.ldamallet import LdaMallet
from gensim.utils import simple_preprocess
import nltk
from nltk.corpus import stopwords

# ------------------------------------
# Global Variables & Utility Functions
# ------------------------------------

# Download NLTK stopwords if necessary
nltk.download('stopwords')
stop_words = stopwords.words('english')

logo_url = "https://theseoconsultant.ai/wp-content/uploads/2024/12/cropped-theseoconsultant-logo-2.jpg"

# Global spaCy model variable
nlp = None

@st.cache_resource
def load_spacy_model():
    """Loads the spaCy model (only once)."""
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
def initialize_bert_model():
    """Initializes the BERT tokenizer and model."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    return tokenizer, model

@st.cache_data(ttl=3600*24)  # Cache for 24 hours
def extract_text_from_url(url):
    """Extracts text from a URL using Selenium, handling JavaScript rendering,
    and excluding header and footer content. Returns the body text."""
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

        # Wait longer for JavaScript to load
        wait = WebDriverWait(driver, 20)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "body")))

        page_source = driver.page_source
        driver.quit()
        soup = BeautifulSoup(page_source, "html.parser")

        # Find the body and remove header and footer tags
        body = soup.find('body')
        if not body:
            return None
        for tag in body.find_all(['header', 'footer']):
            tag.decompose()

        text = body.get_text(separator='\n', strip=True)
        return text

    except (TimeoutException, WebDriverException) as e:
        st.error(f"Selenium error fetching {url}: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error fetching {url}: {e}")
        return None

@st.cache_data
def count_videos(_soup):
    """Counts the number of video elements and embedded videos on the page."""
    # Count HTML5 <video> tags
    video_count = len(_soup.find_all("video"))
    
    # Count <iframe> tags that have YouTube or Vimeo sources
    iframe_videos = len([
        iframe for iframe in _soup.find_all("iframe")
        if any(domain in (iframe.get("src") or "") for domain in ["youtube.com", "youtube-nocookie.com", "vimeo.com"])
    ])
    
    return video_count + iframe_videos

def get_embedding(text, model, tokenizer):
    """Generates a BERT embedding for the given text."""
    tokenizer.pad_token = tokenizer.unk_token
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def create_navigation_menu(logo_url):
    """Creates a top navigation menu."""
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
    """Identifies named entities in the text."""
    doc = nlp_model(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def count_entities(entities: List[Tuple[str, str]]) -> Counter:
    """Counts named entities."""
    entity_counts = Counter()
    for entity, label in entities:
        entity = entity.replace('\n', ' ').replace('\r', '')
        if len(entity) > 2 and label != "CARDINAL":
            entity_counts[(entity, label)] += 1
    return entity_counts

def display_entity_barchart(entity_counts, top_n=30):
    """Displays a bar chart of the top N most frequent entities."""
    entity_data = pd.DataFrame.from_dict(entity_counts, orient='index', columns=['count'])
    entity_data.index.names = ['entity']
    entity_data = entity_data.sort_values('count', ascending=False).head(top_n)
    entity_data = entity_data.reset_index()
    entity_names = [e[0] for e in entity_data['entity']]
    counts = entity_data['count']
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(entity_names, counts)
    ax.set_xlabel("Entities")
    ax.set_ylabel("Frequency")
    ax.set_title("Entity Frequency Bar Chart")
    plt.xticks(rotation=45, ha="right")
    for bar, count in zip(bars, counts):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, int(count), ha='center', va='bottom')
    plt.tight_layout()
    st.pyplot(fig)

# ------------------------------
# Helper Functions for Topic Planner
# ------------------------------

def preprocess_text(text):
    """
    Tokenizes and preprocesses input text using gensim's simple_preprocess.
    Removes punctuation and stop words.
    """
    tokens = simple_preprocess(text, deacc=True)
    return [token for token in tokens if token not in stop_words and len(token) > 3]

def run_lda_mallet(tokenized_texts, num_topics=5, iterations=1000, mallet_path='/path/to/mallet'):
    """
    Builds a dictionary and corpus from the tokenized texts, then runs Gensim's LDA Mallet.
    Replace '/path/to/mallet' with the actual path to your Mallet installation.
    Returns the list of discovered topics.
    """
    dictionary = corpora.Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    lda_model = LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=dictionary, iterations=iterations)
    topics = lda_model.show_topics(formatted=False)
    return topics

# ------------------------------------
# Cosine Similarity Functions
# ------------------------------------

def calculate_overall_similarity(urls, search_term, model, tokenizer):
    """Calculates the overall cosine similarity score for a list of URLs against a search term."""
    search_term_embedding = get_embedding(search_term, model, tokenizer)
    results = []
    for url in urls:
        text = extract_text_from_url(url)
        if text:
            text_embedding = get_embedding(text, model, tokenizer)
            similarity = cosine_similarity(text_embedding, search_term_embedding)[0][0]
            results.append((url, similarity))
        else:
            results.append((url, None))
    return results

def calculate_similarity(text, search_term, tokenizer, model):
    """Calculates similarity scores for each sentence in the text against the search term."""
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_embeddings = [get_embedding(sentence, model, tokenizer) for sentence in sentences]
    search_term_embedding = get_embedding(search_term, model, tokenizer)
    similarities = []
    for sentence_embedding in sentence_embeddings:
        similarity = cosine_similarity(sentence_embedding, search_term_embedding)[0][0]
        similarities.append(similarity)
    return sentences, similarities

def rank_sentences_by_similarity(text, search_term):
    """Calculates cosine similarity between sentences and a search term using BERT."""
    tokenizer, model = initialize_bert_model()
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_embeddings = [get_embedding(sentence, model, tokenizer) for sentence in sentences]
    search_term_embedding = get_embedding(search_term, model, tokenizer)
    similarities = [cosine_similarity(sentence_embedding, search_term_embedding)[0][0]
                    for sentence_embedding in sentence_embeddings]
    min_similarity = min(similarities)
    max_similarity = max(similarities)
    normalized_similarities = ([0.0] * len(similarities) if max_similarity == min_similarity 
                               else [(s - min_similarity) / (max_similarity - min_similarity) for s in similarities])
    return list(zip(sentences, normalized_similarities))

def highlight_text(text, search_term):
    """Highlights text based on similarity to the search term using HTML/CSS."""
    sentences_with_similarity = rank_sentences_by_similarity(text, search_term)
    highlighted_text = ""
    for sentence, similarity in sentences_with_similarity:
        if similarity < 0.35:
            color = "red"
        elif similarity < 0.65:
            color = "black"
        else:
            color = "green"
        highlighted_text += f'<p style="color:{color};">{sentence}</p>'
    return highlighted_text

def rank_sections_by_similarity_bert(text, search_term, top_n=10):
    """Ranks content sections by cosine similarity to a search term using BERT embeddings."""
    tokenizer, model = initialize_bert_model()
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_embeddings = [get_embedding(sentence, model, tokenizer) for sentence in sentences]
    search_term_embedding = get_embedding(search_term, model, tokenizer)
    similarities = []
    for sentence_embedding in sentence_embeddings:
        similarity = cosine_similarity(sentence_embedding, search_term_embedding)[0][0]
        similarities.append(similarity)
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
            tokenizer, model = initialize_bert_model()
            data = []
            similarity_results = calculate_overall_similarity(urls, search_term, model, tokenizer)
            
            for i, url in enumerate(urls):
                try:
                    # Use Selenium to get full page source
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
                    meta_title = driver.title
                    driver.quit()
                    
                    # Create cleaned content by removing header and footer from <body>
                    content_soup = BeautifulSoup(page_source, "html.parser")
                    if content_soup.find("body"):
                        body = content_soup.find("body")
                        for tag in body.find_all(["header", "footer"]):
                            tag.decompose()
                        total_text = body.get_text(separator="\n", strip=True)
                    else:
                        total_text = ""
                    total_word_count = len(total_text.split())
                    
                    # Custom (content) word count: from <p>, <li>, header tags, and tables (from cleaned body)
                    custom_elements = body.find_all(["p", "li", "h1", "h2", "h3", "h4", "h5", "h6"]) if body else []
                    custom_words = []
                    for el in custom_elements:
                        custom_words.extend(el.get_text().split())
                    for table in body.find_all("table"):
                        for row in table.find_all("tr"):
                            for cell in row.find_all(["td", "th"]):
                                custom_words.extend(cell.get_text().split())
                    custom_word_count = len(custom_words)
                    
                    # Extract H1 tag
                    h1_tag = soup.find("h1").get_text(strip=True) if soup.find("h1") else "None"
                    
                    # Count links in header & footer navigation
                    header = soup.find("header")
                    footer = soup.find("footer")
                    header_links = len(header.find_all("a", href=True)) if header else 0
                    footer_links = len(footer.find_all("a", href=True)) if footer else 0
                    total_nav_links = header_links + footer_links
                    
                    # Count total links on the page
                    total_links = len(soup.find_all("a", href=True))
                    
                    # Schema Markup: Find JSON‑LD scripts and extract types
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
                            continue
                    schema_markup = ", ".join(schema_types) if schema_types else "None"
                    
                    # Lists/Tables Present: Extract from the cleaned body only
                    lists_tables = (
                        f"OL: {'Yes' if body.find('ol') else 'No'} | "
                        f"UL: {'Yes' if body.find('ul') else 'No'} | "
                        f"Table: {'Yes' if body.find('table') else 'No'}"
                    )
                    
                    # Count the number of images (from full soup)
                    num_images = len(soup.find_all("img"))
                    
                    # Count videos using the helper function
                    num_videos = count_videos(soup)
                    
                    # Cosine similarity score from earlier calculation
                    similarity_val = similarity_results[i][1] if similarity_results[i][1] is not None else np.nan
                    
                    # Count Unique Entities from the cleaned body text
                    entities = identify_entities(total_text, nlp_model) if total_text and nlp_model else []
                    unique_entity_count = len(set([ent[0] for ent in entities]))
                    
                    # Append data in order:
                    data.append([
                        url,               # URL
                        meta_title,        # Meta Title
                        h1_tag,            # H1
                        total_word_count,  # Total Word Count
                        custom_word_count, # Content Word Count
                        similarity_val,    # Cosine Similarity
                        unique_entity_count,  # # of Unique Entities
                        total_nav_links,   # Nav Links
                        total_links,       # Total Links
                        schema_markup,     # Schema Types
                        lists_tables,      # Lists/Tables
                        num_images,        # Images
                        num_videos         # Videos
                    ])
                    
                except Exception as e:
                    st.error(f"Error processing URL {url}: {e}")
                    data.append([url] + ["Error"] * 12)
            
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
                "# of Videos"
            ])
            
            # Reorder and rename columns as required:
            df = df[[
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
                "# of Videos"
            ]]
            df.columns = [
                "URL",
                "Meta Title",
                "H1",
                "Total Word Count",
                "Content Word Count",
                "Cosine Similarity",
                "# of Unique Entities",
                "Nav Links",
                "Total Links",
                "Schema Types",
                "Lists/Tables",
                "Images",
                "Videos"
            ]
            
            # Ensure Cosine Similarity is numeric.
            df["Cosine Similarity"] = pd.to_numeric(df["Cosine Similarity"], errors="coerce")
            
            st.dataframe(df)


def cosine_similarity_competitor_analysis_page():
    st.title("Cosine Similarity Competitor Analysis")
    st.markdown("By: [The SEO Consultant.ai](https://theseoconsultant.ai)")
    search_term = st.text_input("Enter Search Term:", "")
    urls_input = st.text_area("Enter URLs (one per line):", "")
    urls = [url.strip() for url in urls_input.splitlines() if url.strip()]
    if st.button("Calculate Similarity"):
        if not urls:
            st.warning("Please enter at least one URL.")
        else:
            tokenizer, model = initialize_bert_model()
            with st.spinner("Calculating similarities..."):
                similarity_scores = calculate_overall_similarity(urls, search_term, model, tokenizer)
            urls_plot = [url for url, score in similarity_scores]
            scores_plot = [score if score is not None else 0 for url, score in similarity_scores]
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(urls_plot, scores_plot)
            ax.set_xlabel("URLs")
            ax.set_ylabel("Similarity Score")
            ax.set_title("Cosine Similarity of URLs to Search Term")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            data = {'URL': urls_plot, 'Similarity Score': scores_plot}
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
        tokenizer, model = initialize_bert_model()
        with st.spinner("Calculating Similarities..."):
            sentences, similarities = calculate_similarity(text, search_term, tokenizer, model)
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
        tokenizer, model = initialize_bert_model()
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
    st.markdown("Analyze multiple URLs to identify common entities missing on your site.")
    urls_input = st.text_area("Enter URLs (one per line):", key="entity_urls", value="")
    urls = [url.strip() for url in urls_input.splitlines() if url.strip()]
    exclude_url = st.text_input("Enter URL to exclude:", key="exclude_url", value="")
    if st.button("Analyze", key="entity_button"):
        if not urls:
            st.warning("Please enter at least one URL.")
            return
        with st.spinner("Extracting content and analyzing entities..."):
            nlp_model = load_spacy_model()
            if not nlp_model:
                return
            exclude_text = extract_text_from_url(exclude_url)
            exclude_entities_set = {ent.text.lower() for ent in nlp_model(exclude_text).ents} if exclude_text else set()
            all_entities = []
            entity_counts_per_url: Dict[str, Counter] = {}
            url_entity_counts: Counter = Counter()
            for url in urls:
                text = extract_text_from_url(url)
                if text:
                    entities = identify_entities(text, nlp_model)
                    entities = [(entity, label) for entity, label in entities if label != "CARDINAL"]
                    filtered_entities = [(entity, label) for entity, label in entities if entity.lower() not in exclude_entities_set]
                    entity_counts_per_url[url] = count_entities(filtered_entities)
                    all_entities.extend(filtered_entities)
                    for entity, label in set(filtered_entities):
                        url_entity_counts[(entity, label)] += 1
            filtered_url_entity_counts = Counter({k: v for k, v in url_entity_counts.items() if v >= 2})
            if url_entity_counts:
                st.markdown("### Overall Entity Counts (Found in more than one URL)")
                for (entity, label), count in filtered_url_entity_counts.most_common(50):
                    st.write(f"- {entity} ({label}): {count}")
                display_entity_barchart(filtered_url_entity_counts)
                st.markdown("### Entities from Exclude URL")
                if exclude_text:
                    exclude_doc = nlp_model(exclude_text)
                    exclude_entities_list = [(ent.text, ent.label_) for ent in exclude_doc.ents]
                    exclude_entity_counts = count_entities(exclude_entities_list)
                    for (entity, label), count in exclude_entity_counts.most_common(50):
                        st.write(f"- {entity} ({label}): {count}")
                else:
                    st.write("No entities found in the exclude URL.")
                st.markdown("### Entities Per URL")
                for url, entity_counts_local in entity_counts_per_url.items():
                    st.markdown(f"#### URL: {url}")
                    if entity_counts_local:
                        for (entity, label), count in entity_counts_local.most_common(50):
                            st.write(f"- {entity} ({label}): {count}")
                    else:
                        st.write("No relevant entities found.")
            else:
                st.warning("No relevant entities found.")

def displacy_visualization_page():
    st.header("Entity Visualizer")
    st.markdown("Visualize named entities within your content using displacy.")
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
    st.header("Entity Frequency Bar Chart")
    st.markdown("Generate a bar chart from the most frequent named entities across multiple sites.")
    text_source = st.radio("Select text source:", ('Enter Text', 'Enter URLs'), key="barchart_text_source")
    text = None
    urls = None
    if text_source == 'Enter Text':
        text = st.text_area("Enter Text:", key="barchart_text", height=300, value="")
    else:
        urls_input = st.text_area("Enter URLs (one per line):", key="barchart_url", value="")
        urls = [url.strip() for url in urls_input.splitlines() if url.strip()]
    if st.button("Generate Bar Chart", key="barchart_button"):
        all_text = ""
        entity_texts_by_url: Dict[str, str] = {}
        entity_counts_per_url: Dict[str, Counter] = {}
        if text_source == 'Enter Text':
            if not text:
                st.warning("Please enter the text to proceed.")
                return
            all_text = text
        else:
            if not urls:
                st.warning("Please enter at least one URL.")
                return
            url_texts = {}
            with st.spinner("Extracting text from URLs..."):
                for url in urls:
                    extracted_text = extract_text_from_url(url)
                    if extracted_text:
                        url_texts[url] = extracted_text
                        all_text += extracted_text + "\n"
                    else:
                        st.warning(f"Couldn't grab the text from {url}...")
                        return
            entity_texts_by_url = url_texts
        with st.spinner("Analyzing entities and generating bar chart..."):
            nlp_model = load_spacy_model()
            if not nlp_model:
                st.error("Could not load spaCy model. Aborting.")
                return
            entities = identify_entities(all_text, nlp_model)
            entity_counts = Counter((entity[0], entity[1]) for entity in entities)
            if len(entity_counts) > 0:
                display_entity_barchart(entity_counts)
                if text_source == 'Enter URLs':
                    st.subheader("List of Entities from each URL:")
                    for url in urls:
                        text = entity_texts_by_url.get(url)
                        if text:
                            st.write(f"Text from {url}:")
                            url_entities = identify_entities(text, nlp_model)
                            for entity, label in url_entities:
                                st.write(f"- {entity} ({label})")
                        else:
                            st.write(f"No text for {url}")
            else:
                st.warning("No relevant entities found. Please check your text or URL(s).")

# ------------------------------------
# New Tool: N-gram TF-IDF Analysis with Comparison Table
# ------------------------------------

from sklearn.feature_extraction.text import TfidfVectorizer

def ngram_tfidf_analysis_page():
    st.header("Semantic Gap Analyzer")
    st.markdown("""
        Uncover hidden opportunities by comparing your website's content to your top competitors. Identify key phrases and topics they're covering that you might be missing, and prioritize your content creation based on what works best in your industry.
    """)

    # --- Input Section ---
    st.subheader("Input URLs")
    competitor_urls_input = st.text_area("Enter Competitor URLs (one per line):", key="competitor_urls", value="")
    target_url = st.text_input("Enter Your Target URL:", key="target_url", value="")

    competitor_urls = [url.strip() for url in competitor_urls_input.splitlines() if url.strip()]

    # --- N-gram and TF-IDF Options ---
    st.subheader("Word Options")
    n_value = st.selectbox("Select # of Words in Phrase:", options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=1)
    st.markdown("*(For example, choose 2 for bigrams)*")
    min_df = st.number_input("Minimum Frequency:", value=1, min_value=1)
    max_df = st.number_input("Maximum Frequency:", value=1.0, min_value=0.0, step=0.1)
    top_n = st.slider("Number of top results to display:", min_value=1, max_value=50, value=10)  # Increased default and max

    if st.button("Analyze Content Gaps", key="content_gap_button"):
        if not competitor_urls:
            st.warning("Please enter at least one competitor URL.")
            return
        if not target_url:
            st.warning("Please enter your target URL.")
            return

        # --- 1. Extract Text from URLs ---
        texts = []
        valid_urls = []  # Competitor URLs + Target URL (if text extraction successful)
        url_text_dict = {}  # {url: text}

        with st.spinner("Extracting text from URLs..."):
            # Competitor URLs
            for url in competitor_urls:
                text = extract_text_from_url(url)  # Assuming you have this function
                if text:
                    texts.append(text)
                    url_text_dict[url] = text
                    valid_urls.append(url)
                else:
                    st.warning(f"Could not extract text from {url}")

            # Target URL
            target_text = extract_text_from_url(target_url)
            if target_text:
                url_text_dict[target_url] = target_text
                # Don't add target_text to texts yet; we'll handle it separately
            else:
                st.warning(f"Could not extract text from {target_url}")
                return  # Exit if we can't get text from the target URL

        if not texts:
            st.error("No text was extracted from the competitor URLs.")
            return
        
        # --- 2. Calculate TF-IDF for Competitors ---
        with st.spinner("Calculating TF-IDF scores for competitors..."):
            vectorizer = TfidfVectorizer(ngram_range=(n_value, n_value), min_df=min_df, max_df=max_df)
            tfidf_matrix = vectorizer.fit_transform(texts)  # Only competitor texts
            feature_names = vectorizer.get_feature_names_out()
            df_tfidf_competitors = pd.DataFrame(tfidf_matrix.toarray(), index=valid_urls, columns=feature_names)
        
        # --- 3. Calculate TF-IDF for Target URL ---
        with st.spinner("Calculating TF-IDF scores for target URL..."):
            # Use the *same* vectorizer (fitted on competitors) to transform the target text
            target_tfidf_vector = vectorizer.transform([target_text])
            df_tfidf_target = pd.DataFrame(target_tfidf_vector.toarray(), index=[target_url], columns=feature_names)
        
        # --- 4. Identify Top N-grams for Competitors ---
        top_ngrams_competitors = {}
        for url in valid_urls:  # Only competitor URLs
            row = df_tfidf_competitors.loc[url]
            sorted_row = row.sort_values(ascending=False)
            top_ngrams = sorted_row.head(top_n)
            top_ngrams_competitors[url] = list(top_ngrams.index)  # Store just the n-gram strings
        
        # --- 5. Content Gap Analysis ---
        content_gaps = {}  # {competitor_url: [list of gap n-grams]}

        for competitor_url, competitor_ngrams in top_ngrams_competitors.items():
            gap_ngrams = []
            for ngram in competitor_ngrams:
                # Check if the n-gram exists in the target URL's TF-IDF matrix
                if ngram in df_tfidf_target.columns:
                    # Compare TF-IDF scores
                    competitor_score = df_tfidf_competitors.loc[competitor_url, ngram]
                    target_score = df_tfidf_target.loc[target_url, ngram]

                    if competitor_score > target_score:
                        gap_ngrams.append(f"{ngram} (Competitor: {competitor_score:.3f}, Target: {target_score:.3f})")
                else:
                    # N-gram is completely missing from the target URL
                     competitor_score = df_tfidf_competitors.loc[competitor_url, ngram]
                     gap_ngrams.append(f"{ngram} (Competitor: {competitor_score:.3f}, Target: 0.000)")

            content_gaps[competitor_url] = gap_ngrams

        # --- 6. Display Results ---
        st.markdown("### Content Gap Analysis")

        # Display competitor top n-grams and gaps in a single DataFrame
        st.markdown(f"**Target URL:** {target_url}")
        all_data = {}
        for competitor_url, gap_ngrams in content_gaps.items():
           all_data[competitor_url] = gap_ngrams
           #Pad to ensure all are the same length.
           while len(all_data[competitor_url]) < top_n:
               all_data[competitor_url].append("")
        df_display = pd.DataFrame(all_data)
        st.dataframe(df_display)

# ------------------------------------
# New Tool: Keyword Clustering
# ------------------------------------

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA

def keyword_clustering_from_gap_page():
    st.header("Keyword Clustering from Content Gap Analysis")
    st.markdown(
        """
        This tool combines content gap analysis with keyword clustering.
        First, it extracts key phrases (n‑grams) where your competitors outperform your target.
        Then, it computes BERT embeddings for these gap phrases and clusters them based on their semantic similarity.
        The resulting clusters (and their representative keywords) are displayed below.
        """
    )
    
    # --- Input Section for Semantic Gap Analysis ---
    st.subheader("Input URLs")
    competitor_urls_input = st.text_area("Enter Competitor URLs (one per line):", key="comp_urls", value="")
    target_url = st.text_input("Enter Your Target URL:", key="target_url", value="")
    
    competitor_urls = [url.strip() for url in competitor_urls_input.splitlines() if url.strip()]
    if not competitor_urls or not target_url:
        st.warning("Please enter at least one competitor URL and your target URL.")
        return
    
    # --- N-gram Options for Gap Analysis ---
    st.subheader("N‑gram Settings")
    n_value = st.selectbox("Select # of Words in Phrase:", options=[1, 2, 3, 4, 5], index=1, key="ngram_n")
    min_df = st.number_input("Minimum Frequency:", value=1, min_value=1, key="min_df_gap")
    max_df = st.number_input("Maximum Frequency:", value=1.0, min_value=0.0, step=0.1, key="max_df_gap")
    top_n = st.slider("Number of Top n‑grams to Consider per Competitor:", min_value=1, max_value=50, value=10, key="top_n_gap")
    
    # --- Clustering Settings (Select Algorithm and Parameters) ---
    st.subheader("Clustering Settings")
    algorithm = st.selectbox("Select Clustering Algorithm:", 
                             options=["K-Means", "DBSCAN", "Agglomerative Clustering"],
                             key="clustering_algo_gap")
    if algorithm == "K-Means":
        n_clusters = st.number_input("Number of Clusters:", min_value=1, value=5, key="kmeans_clusters_gap")
    elif algorithm == "DBSCAN":
        eps = st.number_input("Epsilon (eps):", min_value=0.1, value=0.5, step=0.1, key="dbscan_eps_gap")
        min_samples = st.number_input("Minimum Samples:", min_value=1, value=2, key="dbscan_min_samples_gap")
    elif algorithm == "Agglomerative Clustering":
        n_clusters = st.number_input("Number of Clusters:", min_value=1, value=5, key="agg_clusters_gap")
    
    # --- Analyze & Cluster Button ---
    if st.button("Analyze & Cluster Gaps", key="gap_cluster_button"):
        # 1. Extract Text from Competitors and Target
        competitor_texts = []
        competitor_valid_urls = []
        url_text_dict = {}
        with st.spinner("Extracting text from competitor URLs..."):
            for url in competitor_urls:
                text = extract_text_from_url(url)
                if text:
                    competitor_texts.append(text)
                    url_text_dict[url] = text
                    competitor_valid_urls.append(url)
                else:
                    st.warning(f"Could not extract text from {url}")
        
        target_text = extract_text_from_url(target_url)
        if not target_text:
            st.error("Could not extract text from the target URL.")
            return
        
        # 2. Calculate TF‑IDF for Competitors and Target
        with st.spinner("Calculating TF‑IDF scores for competitors..."):
            vectorizer = TfidfVectorizer(ngram_range=(n_value, n_value), min_df=min_df, max_df=max_df)
            tfidf_matrix = vectorizer.fit_transform(competitor_texts)
            feature_names = vectorizer.get_feature_names_out()
            df_tfidf_competitors = pd.DataFrame(tfidf_matrix.toarray(), index=competitor_valid_urls, columns=feature_names)
        
        with st.spinner("Calculating TF‑IDF scores for target URL..."):
            target_tfidf_vector = vectorizer.transform([target_text])
            df_tfidf_target = pd.DataFrame(target_tfidf_vector.toarray(), index=[target_url], columns=feature_names)
        
        # 3. Identify Gap n‑grams
        gap_ngrams = set()
        for url in competitor_valid_urls:
            row = df_tfidf_competitors.loc[url]
            sorted_row = row.sort_values(ascending=False)
            top_ngrams = sorted_row.head(top_n)
            for ngram in top_ngrams.index:
                comp_score = df_tfidf_competitors.loc[url, ngram]
                target_score = df_tfidf_target.loc[target_url, ngram] if ngram in df_tfidf_target.columns else 0.0
                if comp_score > target_score:
                    gap_ngrams.add(ngram)
        
        gap_ngrams = list(gap_ngrams)
        if not gap_ngrams:
            st.error("No gap n‑grams were identified. Consider adjusting your TF‑IDF parameters.")
            return
        
        st.markdown("### Identified Gap n‑grams:")
        st.write(gap_ngrams)
        
        # 4. Compute BERT Embeddings for Each Gap n‑gram
        tokenizer, model = initialize_bert_model()
        embeddings = []
        valid_gap_ngrams = []
        with st.spinner("Computing BERT embeddings for gap n‑grams..."):
            for gram in gap_ngrams:
                emb = get_embedding(gram, model, tokenizer)
                if emb is not None:
                    embeddings.append(emb.squeeze())
                    valid_gap_ngrams.append(gram)
        
        if len(valid_gap_ngrams) == 0:
            st.error("Could not compute embeddings for any gap n‑grams.")
            return
        
        embeddings = np.vstack(embeddings)
        
        # 5. Perform Clustering Using Selected Algorithm and Parameters
        if algorithm == "K-Means":
            from sklearn.cluster import KMeans
            clustering_model = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = clustering_model.fit_predict(embeddings)
            centers = clustering_model.cluster_centers_
            rep_keywords = {}
            for i in range(n_clusters):
                cluster_grams = [ng for ng, label in zip(valid_gap_ngrams, cluster_labels) if label == i]
                if not cluster_grams:
                    continue
                cluster_embeddings = embeddings[cluster_labels == i]
                distances = np.linalg.norm(cluster_embeddings - centers[i], axis=1)
                rep_keyword = cluster_grams[np.argmin(distances)]
                rep_keywords[i] = rep_keyword
        
        elif algorithm == "DBSCAN":
            from sklearn.cluster import DBSCAN
            clustering_model = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = clustering_model.fit_predict(embeddings)
            rep_keywords = {}
            unique_labels = set(cluster_labels)
            for label in unique_labels:
                if label == -1:
                    continue
                cluster_grams = [ng for ng, l in zip(valid_gap_ngrams, cluster_labels) if l == label]
                cluster_embeddings = embeddings[cluster_labels == label]
                sims = cosine_similarity(cluster_embeddings, cluster_embeddings)
                rep_keyword = cluster_grams[np.argmax(np.sum(sims, axis=1))]
                rep_keywords[label] = rep_keyword
        
        elif algorithm == "Agglomerative Clustering":
            from sklearn.cluster import AgglomerativeClustering
            clustering_model = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels = clustering_model.fit_predict(embeddings)
            rep_keywords = {}
            for i in range(n_clusters):
                cluster_grams = [ng for ng, label in zip(valid_gap_ngrams, cluster_labels) if label == i]
                cluster_embeddings = embeddings[cluster_labels == i]
                if len(cluster_embeddings) > 1:
                    sims = cosine_similarity(cluster_embeddings, cluster_embeddings)
                    rep_keyword = cluster_grams[np.argmax(np.sum(sims, axis=1))]
                else:
                    rep_keyword = cluster_grams[0]
                rep_keywords[i] = rep_keyword
        
        # 6. Display the Clusters
        clusters = {}
        for gram, label in zip(valid_gap_ngrams, cluster_labels):
            clusters.setdefault(label, []).append(gram)
        
        st.markdown("### Keyword Clusters:")
        for label, gram_list in clusters.items():
            if label == -1:
                st.markdown("**Noise:**")
            else:
                rep = rep_keywords.get(label, "N/A")
                st.markdown(f"**Cluster {label}** (Representative: {rep}):")
            for gram in gram_list:
                st.write(f" - {gram}")
        
        # Optional: Visualize Clusters using PCA
        with st.spinner("Generating cluster visualization..."):
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings)
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=cluster_labels, cmap='viridis', s=100)
            for i, txt in enumerate(valid_gap_ngrams):
                ax.annotate(txt, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=8)
            ax.set_title("Cluster Visualization (PCA Reduced)")
            ax.set_xlabel("PCA Component 1")
            ax.set_ylabel("PCA Component 2")
            st.pyplot(fig)

def topic_planner_page():
    st.header("Topic Planner")
    st.markdown("Input a list of URLs. The app will fetch each URL's text content and perform topic modeling using Gensim's LDA Mallet.")
    
    urls_input = st.text_area("Enter URLs (one per line):", value="")
    num_topics = st.number_input("Number of topics to extract:", min_value=2, max_value=20, value=5)
    mallet_path = st.text_input("Mallet path (e.g., /path/to/mallet):", value="/path/to/mallet")
    
    if st.button("Run Topic Planner"):
        urls = [url.strip() for url in urls_input.splitlines() if url.strip()]
        if not urls:
            st.warning("Please enter at least one URL.")
            return
        
        all_tokens = []
        st.write("Fetching and processing URLs...")
        for url in urls:
            st.write(f"Processing: {url}")
            text = extract_text_from_url(url)
            if text:
                tokens = preprocess_text(text)
                if tokens:
                    all_tokens.append(tokens)
            else:
                st.warning(f"Could not extract text from: {url}")
        
        if all_tokens:
            topics = run_lda_mallet(all_tokens, num_topics=num_topics, mallet_path=mallet_path)
            st.write("### Discovered Topics")
            for idx, topic in topics:
                topic_words = ", ".join([word for word, weight in topic])
                st.write(f"Topic {idx}: {topic_words}")
        else:
            st.write("No valid text was extracted from the provided URLs.")

# ------------------------------------
# Main Streamlit App
# ------------------------------------

def main():
    st.set_page_config(
        page_title="Semantic Search SEO Analysis Tools | The SEO Consultant.ai",
        page_icon="✏️",
        layout="wide"
    )

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
        "Entity Frequency Bar Chart",
        "Semantic Gap Analyzer",
        "Keyword Clustering",  # New tool added here
        "Topic Planner" # New tool added here
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
    elif tool == "Entity Frequency Bar Chart":
        named_entity_barchart_page()
    elif tool == "Semantic Gap Analyzer":
        ngram_tfidf_analysis_page()
    elif tool == "Keyword Clustering":
        keyword_clustering_from_gap_page()
    elif tool == "Topic Planner":
        topic_planner_page()

    st.markdown("---")
    st.markdown(
        "Powered by [The SEO Consultant.ai](https://theseoconsultant.ai)",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()




