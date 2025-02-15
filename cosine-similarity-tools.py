import streamlit as st
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from PIL import Image
import cairosvg

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

# ------------------------------------
# Global Variables & Utility Functions
# ------------------------------------

logo_url = "https://theseoconsultant.ai/wp-content/uploads/2024/12/cropped-theseoconsultant-logo-2.jpg"

# Global spacy model variable
nlp = None

@st.cache_resource
def load_spacy_model():
    """Loads the spaCy model (only once)."""
    global nlp
    if nlp is None:
        try:
            nlp = spacy.load("en_core_web_sm")
            print("spaCy model loaded successfully")
        except OSError:
            print("Downloading en_core_web_sm model...")
            spacy.cli.download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
            print("en_core_web_sm downloaded and loaded")
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

def extract_text_from_url(url):
    """Extracts text from a URL using Selenium, handling JavaScript rendering,
    excluding header and footer content.  Returns all text content from the
    <body> except for the header and footer.
    """
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        user_agent = "Mozilla/5.0 (iPhone; CPU iPhone OS 14_7_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.7.1 Mobile/15E148 Safari/604.1"
        chrome_options.add_argument(f"user-agent={user_agent}")

        driver = webdriver.Chrome(options=chrome_options)

        driver.get(url)

        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "body")))

        page_source = driver.page_source
        driver.quit()
        soup = BeautifulSoup(page_source, "html.parser")

        # Find the body
        body = soup.find('body')
        if not body:
            return None

        # Remove header and footer tags
        for tag in body.find_all(['header', 'footer']):
            tag.decompose()

        # Extract all text from the remaining elements in the body
        text = body.get_text(separator='\n', strip=True)

        return text

    except (TimeoutException, WebDriverException) as e:
        st.error(f"Selenium error fetching {url}: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error fetching {url}: {e}")
        return None


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

def display_entity_barchart(entity_counts, top_n=50):
    """Displays a bar chart of the top N most frequent entities."""
    # Convert entity counts to a pandas DataFrame for easier plotting
    entity_data = pd.DataFrame.from_dict(entity_counts, orient='index', columns=['count'])
    entity_data.index.names = ['entity']
    entity_data = entity_data.sort_values('count', ascending=False).head(top_n)
    entity_data = entity_data.reset_index()  # Make 'entity' a regular column

    # Separate entities and labels for plotting
    entity_names = [e[0] for e in entity_data['entity']]
    labels = [e[1] for e in entity_data['entity']]
    counts = entity_data['count']

    # Create the bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(entity_names, counts)
    ax.set_xlabel("Entities")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Entity Topic Gap Analysis")
    plt.xticks(rotation=45, ha="right")

    # Annotate bars with counts
    for bar, count in zip(bars, counts):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, int(count), ha='center', va='bottom')

    plt.tight_layout()  # Adjust layout to prevent labels from overlapping
    st.pyplot(fig)

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

    # Normalize the similarity scores to be between 0 and 1
    min_similarity = min(similarities)
    max_similarity = max(similarities)
    if max_similarity == min_similarity:
        normalized_similarities = [0.0] * len(similarities)  # Avoid division by zero
    else:
        normalized_similarities = [(s - min_similarity) / (max_similarity - min_similarity) for s in similarities]

    return list(zip(sentences, normalized_similarities))

def highlight_text(text, search_term):
    """Highlights text based on similarity to the search term using HTML/CSS, adding paragraph breaks."""
    sentences_with_similarity = rank_sentences_by_similarity(text, search_term)

    highlighted_text = ""
    for sentence, similarity in sentences_with_similarity:
        print(f"Sentence: {sentence}, Similarity: {similarity}")  # Debugging print

        if similarity < 0.35:
            color = "red"
        elif similarity < 0.65:
            color = "black"
        else:
            color = "green"

        # Enclose each sentence in a <p> tag for paragraph breaks
        highlighted_text += f'<p style="color:{color};">{sentence}</p>'
    return highlighted_text

def rank_sections_by_similarity_bert(text, search_term, top_n=10):
    """Ranks content sections by cosine similarity to a search term using BERT embeddings."""
    tokenizer, model = initialize_bert_model()

    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    sentences = [s.strip() for s in sentences if s.strip()]  # Remove empty strings

    sentence_embeddings = [get_embedding(sentence, model, tokenizer) for sentence in sentences]

    # Generate embedding for the search term
    search_term_embedding = get_embedding(search_term, model, tokenizer)

    # Calculate cosine similarity between each sentence and the search term
    similarities = []
    for sentence_embedding in sentence_embeddings:
        similarity = cosine_similarity(sentence_embedding, search_term_embedding)[0][0] # Access scalar similarity value
        similarities.append(similarity)

    # Create a list of (sentence, similarity score) tuples
    section_scores = list(zip(sentences, similarities))

    # Sort by similarity score
    sorted_sections = sorted(section_scores, key=lambda item: item[1], reverse=True)

    # Get top and bottom sections
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
    search_term = st.text_input("Enter Search Term (for Cosine Similarity):", key="dashboard_search_term", value="Enter Your SEO Keyword Here")

    if st.button("Analyze URLs", key="dashboard_button"):
        if not urls:
            st.warning("Please enter at least one URL.")
            return

        with st.spinner("Analyzing URLs..."):
            # Initialize models
            nlp_model = load_spacy_model()
            tokenizer, model = initialize_bert_model()

            data = []
            similarity_results = calculate_overall_similarity(urls, search_term, model, tokenizer)

            for i, url in enumerate(urls):
                try:
                    # 1. Meta Title
                    chrome_options = Options()
                    chrome_options.add_argument("--headless")
                    chrome_options.add_argument("--no-sandbox")
                    chrome_options.add_argument("--disable-dev-shm-usage")
                    user_agent = "Mozilla/5.0 (iPhone; CPU iPhone OS 14_7_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.7.1 Mobile/15E148 Safari/604.1"
                    chrome_options.add_argument(f"user-agent={user_agent}")
                    driver = webdriver.Chrome(options=chrome_options)
                    driver.get(url)
                    meta_title = driver.title
                    page_source = driver.page_source
                    driver.quit()

                    # 2. Content Word Count
                    text = extract_text_from_url(url)
                    word_count = len(text.split()) if text else 0

                    # 3. Unique Entities
                    entities = set()
                    if nlp_model and text:
                        doc = nlp_model(text)
                        for ent in doc.ents:
                            entities.add(ent.text)
                    unique_entity_count = len(entities)

                    # 4. Overall Cosine Similarity Score
                    similarity_score = similarity_results[i][1] if similarity_results[i][1] is not None else "N/A"
                    if(similarity_score != "N/A"):
                        st.write(f"Cosine similarity for {url}: {similarity_score}")  # Keep the output within the function
                    else:
                        st.write(f"Could not extract text from {url}")

                    data.append([url, meta_title, word_count, unique_entity_count, similarity_score])

                except Exception as e:
                    st.error(f"Error processing URL {url}: {e}")
                    data.append([url, "Error", "Error", "Error", "Error"])

            df = pd.DataFrame(data, columns=["URL", "Meta Title", "Content Word Count", "# of Unique Entities", "Overall Cosine Similarity Score"])
            st.dataframe(df)

def cosine_similarity_competitor_analysis_page():
    """Cosine Similarity Score - Competitor Analysis Page."""
    st.title("URL Similarity App")
    st.markdown("By: [The SEO Consultant.ai](https://theseoconsultant.ai)")

    # Input fields
    search_term = st.text_input("Enter Search Term:", "")
    urls_input = st.text_area("Enter URLs (one per line):",
                            """""")
    urls = [url.strip() for url in urls_input.splitlines() if url.strip()]


    if st.button("Calculate Similarity"):
        if not urls:
            st.warning("Please enter at least one URL.")
        else:
            # Initialize model (only once)
            tokenizer, model = initialize_bert_model()

            with st.spinner("Calculating similarities..."):
                similarity_scores = calculate_overall_similarity(urls, search_term, model, tokenizer)

            # Prepare data for plotting
            urls_plot = [url for url, score in similarity_scores]
            scores_plot = [score if score is not None else 0 for url, score in similarity_scores]  # Replace None with 0 for plotting

            # Create the bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(urls_plot, scores_plot)
            ax.set_xlabel("URLs")
            ax.set_ylabel("Similarity Score")
            ax.set_title("Cosine Similarity of URLs to Search Term")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)

            # Display results in a table
            data = {'URL': urls_plot, 'Similarity Score': scores_plot}
            df = pd.DataFrame(data)
            st.dataframe(df)

def cosine_similarity_every_embedding_page():
    """Cosine Similarity Score - Every Embedding Page."""
    st.header("Cosine Similarity Score - Every Embedding")
    st.markdown("Calculates the cosine similarity score for each sentence in your input.")

    # URL Input
    url = st.text_input("Enter URL (Optional):", key="every_embed_url", value="")
    use_url = st.checkbox("Use URL for Text Input", key="every_embed_use_url")

    # Text Area - Disable if URL is used
    text = st.text_area("Enter Text:", key="every_embed_text", value="Put Your Content Here.", disabled=use_url)

    # Search term input
    search_term = st.text_input("Enter Search Term:", key="every_embed_search", value="Enter Your SEO Keyword Here")

    if st.button("Calculate Similarity", key="every_embed_button"):
        # Prioritize URL if it's used
        if use_url:
            if url:
                with st.spinner(f"Extracting and analyzing text from {url}..."):
                    text = extract_text_from_url(url)
                    if not text:
                        st.error(f"Could not extract text from {url}. Please check the URL.")
                        return  # Stop execution
            else:
                st.warning("Please enter a URL to extract the text.")
                return
        elif not text:
            st.warning("Please enter either text or a URL.")
            return  # Stop execution

        tokenizer, model = initialize_bert_model()
        with st.spinner("Calculating Similarities..."):
            # Calculate similarities
            sentences, similarities = calculate_similarity(text, search_term, tokenizer, model)

        st.subheader("Similarity Scores:")
        for i, (sentence, score) in enumerate(zip(sentences, similarities), 1):
            st.write(f"{i}. {sentence} (Similarity: {score:.4f})")

def cosine_similarity_content_heatmap_page():
    st.header("Cosine Similarity Content Heatmap")
    st.markdown("Green text is the most relevant to the search query. Red is the least relevant content to search query.")

    # URL Input
    url = st.text_input("Enter URL (Optional):", key="heatmap_url", value="")
    use_url = st.checkbox("Use URL for Text Input", key="heatmap_use_url")

    # Input text area, disable if URL is used
    input_text = st.text_area("Enter your text:", key="heatmap_input", height=300, value="Paste your text here.", disabled=use_url)

    search_term = st.text_input("Enter your search term:", key="heatmap_search", value="Enter Your SEO Keyword Here")

    if st.button("Highlight", key="heatmap_button"):
         # Prioritize URL if it's used
        if use_url:
            if url:
                with st.spinner(f"Extracting and analyzing text from {url}..."):
                    text = extract_text_from_url(url)
                    if not text:
                        st.error(f"Could not extract text from {url}. Please check the URL.")
                        return  # Stop execution
                input_text = text  # Set the input_text to extracted text
            else:
                st.warning("Please enter a URL to extract the text.")
                return
        elif not input_text:
            st.error("Please enter either text or a URL.")
            return  # Stop execution

        # Now input_text (or extracted text) is valid
        with st.spinner("Generating highlighted text..."):
            highlighted_text = highlight_text(input_text, search_term)

        st.markdown(highlighted_text, unsafe_allow_html=True)

def top_bottom_embeddings_page():
    """Top 10 and Bottom 10 Embeddings based on Cosine Similarity."""
    st.header("Top 10 & Bottom 10 Embeddings")

     # URL Input
    url = st.text_input("Enter URL (Optional):", key="tb_url", value="")
    use_url = st.checkbox("Use URL for Text Input", key="tb_use_url")

    # Text Area - Disable if URL is used
    text = st.text_area("Enter your text:", key="top_bottom_text", height=300, value="Put Your Content Here.", disabled=use_url)

    search_term = st.text_input("Enter your search term:", key="top_bottom_search", value="Enter Your SEO Keyword Here")
    top_n = st.slider("Number of results:", min_value=1, max_value=20, value=10, key="top_bottom_slider")

    if st.button("Search", key="top_bottom_button"):
         # Prioritize URL if it's used
        if use_url:
            if url:
                with st.spinner(f"Extracting and analyzing text from {url}..."):
                    text = extract_text_from_url(url)
                    if not text:
                        st.error(f"Could not extract text from {url}. Please check the URL.")
                        return  # Stop execution
                input_text = text #Set the input_text to extracted text
            else:
                st.error("Please enter either text or a URL.")
                return  # Stop execution
        elif not text:
            st.error("Please enter either text or a URL.")
            return  # Stop execution

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
    """Entity Analysis Page with a bar chart."""
    st.header("Entity Topic Gap Analysis")
    st.markdown("Analyze content from multiple URLs to identify common entities not found on your site.  Consider adding these named entities to your content to improve search relevancy & topic coverage.")

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
            exclude_entities_set = set()
            if exclude_text:
                exclude_doc = nlp_model(exclude_text)
                exclude_entities_set = {ent.text.lower() for ent in exclude_doc.ents}

            all_entities = []  # Collect all entities across URLs
            entity_counts_per_url: Dict[str, Counter] = {}  # Counts per URL
            url_entity_counts: Counter = Counter()  # Overall counts

            for url in urls:
                text = extract_text_from_url(url)
                if text:
                    entities = identify_entities(text, nlp_model)
                    entities = [(entity, label) for entity, label in entities if label != "CARDINAL"]  # Exclude CARDINAL

                    filtered_entities = [(entity, label) for entity, label in entities
                                         if entity.lower() not in exclude_entities_set]  # Exclude entities from exclude_url

                    entity_counts_per_url[url] = count_entities(filtered_entities)
                    all_entities.extend(filtered_entities)  # Add to the overall list

                    for entity, label in set(filtered_entities):
                        url_entity_counts[(entity, label)] += 1

            # Overall entity counts after filtering
            filtered_url_entity_counts = Counter({k: v for k, v in url_entity_counts.items() if v >= 2})  # Found in > 1 URL

            if url_entity_counts:
                st.markdown("### Overall Entity Counts (Excluding Entities from Exclude URL and CARDINAL Entities, Found in More Than One URL)")
                for (entity, label), count in filtered_url_entity_counts.most_common(50):
                    st.write(f"- {entity} ({label}): {count}")

                # *ADD THE BAR CHART HERE*
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

# ------------------------------------
# App 8: Displacy Entity Visualizer Page
# ------------------------------------

def displacy_visualization_page():
    st.header("Entity Visualizer")
    st.markdown("Visualize named entities within your content using displacy.")

    url = st.text_input("Enter a URL (Optional):", key="displacy_url", value="")
    use_url = st.checkbox("Use URL for Text Input", key="displacy_use_url")
    text = st.text_area("Enter Text:", key="displacy_text", value="Paste your text here.", disabled=use_url)

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

# ------------------------------------
# App 9: Entity Frequency Bar Chart Page
# ------------------------------------

def named_entity_barchart_page():
    st.header("Entity Frequency Bar Chart")
    st.markdown("Generate a bar chart from the most frequent named entities across multiple sites.")

    text_source = st.radio("Select text source:", ('Enter Text', 'Enter URLs'), key="barchart_text_source")
    text = None
    urls = None

    if text_source == 'Enter Text':
        text = st.text_area("Enter Text:", key="barchart_text", height=300, value="Paste your text here.")
    else:
        urls_input = st.text_area("Enter URLs (one per line):", key="barchart_url", value="")
        urls = [url.strip() for url in urls_input.splitlines() if url.strip()]

    if st.button("Generate Bar Chart", key="barchart_button"):
        all_text = ""
        entity_texts_by_url: Dict[str, str] = {}
        entity_counts_per_url: Dict[str, Counter] = {}

        # Validation and text retrieval
        if text_source == 'Enter Text':
            if not text:
                st.warning("Please enter the text to proceed.")
                return
            all_text = text
        else:  # Using URLs
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

            # Identify entities for the combined text
            entities = identify_entities(all_text, nlp_model)

            # Count the entities
            entity_counts = Counter((entity[0], entity[1]) for entity in entities)

            # Display
            if len(entity_counts) > 0:
                display_entity_barchart(entity_counts)

                # Display URLs and entities only if the URLs option was chosen
                if text_source == 'Enter URLs':
                    st.subheader("List of Entities from each URLs:")
                    for url in urls:
                        text = entity_texts_by_url.get(url)
                        if text:
                            st.write(f"Text from {url}:")
                            url_entities = identify_entities(text, nlp_model)
                            for entity, label in url_entities:
                                st.write(f"- {entity} ({label})")
                        else:
                            st.write(f"No Text for the {url}")
            else:
                st.warning("No relevant entities found. Please check your text or URL(s).")

# ------------------------------------
# Main Streamlit App
# ------------------------------------

def main():
    st.set_page_config(
        page_title="SEO Analysis Tools | The SEO Consultant.ai",
        page_icon="✏️",
        layout="wide"
    )

    create_navigation_menu(logo_url)
    st.sidebar.header("SEO Analysis Tools")
    tool = st.sidebar.selectbox("Select Tool:", [
        "URL Analysis Dashboard",
        "Cosine Similarity - Competitor Analysis",
        "Cosine Similarity - Every Embedding",
        "Cosine Similarity - Content Heatmap",
        "Top/Bottom 10 Embeddings",
        "Entity Topic Gap Analysis",
        "Entity Visualizer",
        "Entity Frequency Bar Chart"
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

    st.markdown("---")
    st.markdown(
        "Powered by [The SEO Consultant.ai](https://theseoconsultant.ai)",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
