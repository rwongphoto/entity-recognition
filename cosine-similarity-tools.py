import streamlit as st
import torch
from transformers import AutoTokenizer, BertModel, pipeline
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

from wordcloud import WordCloud

# ------------------------------------
# Global Variables & Utility Functions
# ------------------------------------

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
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
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
        user_agent = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36")
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
    video_count = len(_soup.find_all("video"))
    iframe_videos = len([
        iframe for iframe in _soup.find_all("iframe")
        if any(domain in (iframe.get("src") or "") for domain in ["youtube.com", "youtube-nocookie.com", "vimeo.com"])
    ])
    return video_count + iframe_videos

def get_embedding(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
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
    """Displays a bar chart of the top N most frequent entities (excluding CARDINAL)."""
    filtered_entity_counts = {
        (entity, label): count
        for (entity, label), count in entity_counts.items()
        if label != "CARDINAL"
    }
    entity_data = pd.DataFrame.from_dict(filtered_entity_counts, orient='index', columns=['count'])
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

def display_entity_wordcloud(entity_counts):
    """
    Generate and display a wordcloud from the given frequency counts.
    The keys can be tuples (e.g., (entity, label)) or simple strings.
    """
    aggregated = {}
    for key, count in entity_counts.items():
        if isinstance(key, tuple):
            k = key[0]
        else:
            k = key
        aggregated[k] = aggregated.get(k, 0) + count
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(aggregated)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout()
    st.pyplot(fig)

# ------------------------------------
# Summarization Functionality
# ------------------------------------
@st.cache_resource
def load_summarizer():
    """
    Load the summarization pipeline.
    This model uses the facebook/bart-large-cnn model.
    """
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer

def summarization_page():
    st.header("Text Summarization")
    st.markdown("Enter your text below and click **Summarize** to generate a summary.")
    text = st.text_area("Enter text to summarize:", height=300)
    max_length = st.slider("Maximum summary length:", min_value=50, max_value=300, value=150)
    min_length = st.slider("Minimum summary length:", min_value=20, max_value=100, value=30)
    if st.button("Summarize"):
        if not text:
            st.warning("Please enter some text to summarize.")
        else:
            with st.spinner("Generating summary..."):
                summarizer = load_summarizer()
                summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
            st.subheader("Summary:")
            st.write(summary[0]['summary_text'])

def summarization_comparison_page():
    st.header("Summarization & Comparison")
    st.markdown(
        """
        This feature extracts content from competitor URLs and your URL, generates summaries for each,
        and then produces a final comparison summary highlighting key differences between your site and your competitors.
        """
    )
    competitor_input = st.text_area("Enter Competitor URLs (one per line):", height=150)
    my_url = st.text_input("Enter Your URL:")
    max_length = st.slider("Maximum summary length:", min_value=50, max_value=300, value=150)
    min_length = st.slider("Minimum summary length:", min_value=20, max_value=100, value=40)
    if st.button("Analyze and Compare Summaries"):
        if not competitor_input or not my_url:
            st.warning("Please enter both competitor URLs and your URL.")
            return
        competitor_urls = [url.strip() for url in competitor_input.splitlines() if url.strip()]
        summarizer = load_summarizer()
        competitor_summaries = {}
        for url in competitor_urls:
            with st.spinner(f"Processing competitor URL: {url}"):
                text = extract_text_from_url(url)
                if text:
                    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
                    competitor_summaries[url] = summary[0]['summary_text']
                else:
                    competitor_summaries[url] = "No content extracted."
        with st.spinner("Processing your URL..."):
            my_text = extract_text_from_url(my_url)
            if not my_text:
                st.error("Could not extract content from your URL.")
                return
            my_summary = summarizer(my_text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
        st.subheader("Competitor Summaries:")
        for url, summary in competitor_summaries.items():
            st.markdown(f"**{url}:**")
            st.write(summary)
        st.subheader("Your Site Summary:")
        st.write(my_summary)
        competitor_combined = " ".join(competitor_summaries.values())
        comparison_prompt = (
            f"Competitor summaries: {competitor_combined}\n\n"
            f"My site summary: {my_summary}\n\n"
            "Based on the above, summarize the key differences between the competitor content and my site content "
            "in terms of topics, tone, and coverage."
        )
        with st.spinner("Generating comparison summary..."):
            comparison_summary = summarizer(comparison_prompt, max_length=max_length, min_length=50, do_sample=False)[0]['summary_text']
        st.subheader("Comparison Summary:")
        st.write(comparison_summary)

# ------------------------------------
# Cosine Similarity Functions
# ------------------------------------
def calculate_overall_similarity(urls, search_term, model, tokenizer):
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
                    for table in body.find_all("table"):
                        for row in table.find_all("tr"):
                            for cell in row.find_all(["td", "th"]):
                                custom_words.extend(cell.get_text().split())
                    custom_word_count = len(custom_words)
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
                            continue
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
                        num_videos
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
            df["Cosine Similarity"] = pd.to_numeric(df["Cosine Similarity"], errors="coerce")
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
        st.markdown("Paste the competitor content below. If you have multiple competitors, separate each with `---`.")
        pasted_content = st.text_area("Enter Competitor Content:", height=200)
        competitor_contents = [content.strip() for content in pasted_content.split('---') if content.strip()]
    if st.button("Calculate Similarity"):
        tokenizer, model = initialize_bert_model()
        if source_option == "Extract from URL":
            if not competitor_urls:
                st.warning("Please enter at least one URL.")
                return
            with st.spinner("Calculating similarities from URLs..."):
                similarity_scores = calculate_overall_similarity(competitor_urls, search_term, model, tokenizer)
            urls_plot = [url for url, score in similarity_scores]
            scores_plot = [score if score is not None else 0 for url, score in similarity_scores]
        else:
            if not competitor_contents:
                st.warning("Please paste at least one content block.")
                return
            with st.spinner("Calculating similarities from pasted content..."):
                similarity_scores = []
                for idx, content in enumerate(competitor_contents):
                    text_embedding = get_embedding(content, model, tokenizer)
                    search_embedding = get_embedding(search_term, model, tokenizer)
                    similarity = cosine_similarity(text_embedding, search_embedding)[0][0]
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
    st.markdown("Analyze multiple sources to identify common entities missing on your site.")
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
    st.markdown("#### Exclude Source")
    exclude_option = st.radio(
        "Select exclude content source:",
        options=["Extract from URL", "Paste Content"],
        index=0,
        key="exclude_source"
    )
    if exclude_option == "Extract from URL":
        exclude_input = st.text_input("Enter URL to exclude:", key="exclude_url", value="")
    else:
        exclude_input = st.text_area("Paste content to exclude:", key="exclude_text", value="", height=100)
    if st.button("Analyze", key="entity_button"):
        if not competitor_list:
            st.warning("Please provide at least one competitor content or URL.")
            return
        with st.spinner("Extracting content and analyzing entities..."):
            nlp_model = load_spacy_model()
            if not nlp_model:
                return
            if exclude_option == "Extract from URL":
                exclude_text = extract_text_from_url(exclude_input) if exclude_input else ""
            else:
                exclude_text = exclude_input
            exclude_entities_set = {ent.text.lower() for ent in nlp_model(exclude_text).ents} if exclude_text else set()
            all_entities = []
            entity_counts_per_source = {}
            url_entity_counts = Counter()
            for source in competitor_list:
                if competitor_source_option == "Extract from URL":
                    text = extract_text_from_url(source)
                else:
                    text = source
                if text:
                    entities = identify_entities(text, nlp_model)
                    entities = [(entity, label) for entity, label in entities if label != "CARDINAL"]
                    filtered_entities = [(entity, label) for entity, label in entities if entity.lower() not in exclude_entities_set]
                    entity_counts_per_source[source] = count_entities(filtered_entities)
                    all_entities.extend(filtered_entities)
                    for entity, label in set(filtered_entities):
                        url_entity_counts[(entity, label)] += 1
            filtered_url_entity_counts = Counter({k: v for k, v in url_entity_counts.items() if v >= 2})
            if url_entity_counts:
                st.markdown("### Overall Entity Counts (Found in more than one source)")
                for (entity, label), count in filtered_url_entity_counts.most_common(50):
                    st.write(f"- {entity} ({label}): {count}")
                display_entity_barchart(filtered_url_entity_counts)
                st.subheader("Entity Wordcloud")
                display_entity_wordcloud(filtered_url_entity_counts)
                st.markdown("### Entities from Exclude Content")
                if exclude_text:
                    exclude_doc = nlp_model(exclude_text)
                    exclude_entities_list = [(ent.text, ent.label_) for ent in exclude_doc.ents]
                    exclude_entity_counts = count_entities(exclude_entities_list)
                    for (entity, label), count in exclude_entity_counts.most_common(50):
                        st.write(f"- {entity} ({label}): {count}")
                else:
                    st.write("No entities found in the exclude content.")
                st.markdown("### Entities Per Source")
                for source, entity_counts_local in entity_counts_per_source.items():
                    st.markdown(f"#### Source: {source}")
                    if entity_counts_local:
                        for (entity, label), count in entity_counts_local.most_common(50):
                            st.write(f"- {entity} ({label}): {count}")
                    else:
                        st.write("No relevant entities found.")
            else:
                st.warning("No relevant entities found.")

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
    st.header("Entity Frequency Bar Chart")
    st.markdown("Generate a bar chart and wordcloud from the most frequent named entities across multiple sources.")
    input_method = st.radio(
        "Select content input method:",
        options=["Extract from URL", "Paste Content"],
        key="entity_barchart_input"
    )
    if input_method == "Paste Content":
        st.markdown(
            "Please paste your content in the text area below. If you have multiple sources, separate each content block with the delimiter `---`."
        )
        text = st.text_area("Enter Text:", key="barchart_text", height=300, value="")
    else:
        st.markdown(
            "Please enter one or more URLs (one per line) from which to extract content. The app will fetch and combine the text from each URL."
        )
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
                        return
        with st.spinner("Analyzing entities and generating visualizations..."):
            nlp_model = load_spacy_model()
            if not nlp_model:
                st.error("Could not load spaCy model. Aborting.")
                return
            entities = identify_entities(all_text, nlp_model)
            filtered_entities = [
                (entity, label)
                for entity, label in entities
                if label not in ["CARDINAL", "PERCENT", "MONEY"]
            ]
            entity_counts = count_entities(filtered_entities)
            if entity_counts:
                st.subheader("Entity Frequency Bar Chart")
                display_entity_barchart(entity_counts)
                st.subheader("Entity Wordcloud")
                display_entity_wordcloud(entity_counts)
                if input_method == "Extract from URL":
                    st.subheader("List of Entities from each URL:")
                    for url in urls:
                        text_from_url = entity_texts_by_url.get(url)
                        if text_from_url:
                            st.write(f"Text from {url}:")
                            url_entities = identify_entities(text_from_url, nlp_model)
                            for entity, label in url_entities:
                                st.write(f"- {entity} ({label})")
                        else:
                            st.write(f"No text for {url}")
            else:
                st.warning("No relevant entities found. Please check your text or URL(s).")

def ngram_tfidf_analysis_page():
    st.header("Semantic Gap Analyzer")
    st.markdown(
        """
        Uncover hidden opportunities by comparing your website's content to your top competitors. 
        Identify key phrases and topics they're covering that you might be missing, and prioritize your content creation.
        """
    )
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
    n_value = st.selectbox("Select # of Words in Phrase:", options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=1)
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
        competitor_texts = []
        valid_competitor_sources = []
        with st.spinner("Extracting competitor content..."):
            for source in competitor_list:
                if competitor_source_option == "Extract from URL":
                    text = extract_text_from_url(source)
                else:
                    text = source
                if text:
                    competitor_texts.append(text)
                    valid_competitor_sources.append(source)
                else:
                    st.warning(f"Could not extract content from: {source}")
        if target_source_option == "Extract from URL":
            target_content = extract_text_from_url(target_url)
            if not target_content:
                st.warning(f"Could not extract content from target URL: {target_url}")
                return
        else:
            target_content = target_text
        if not competitor_texts:
            st.error("No competitor content was extracted.")
            return
        with st.spinner("Calculating TF-IDF scores for competitors..."):
            vectorizer = TfidfVectorizer(ngram_range=(n_value, n_value), min_df=min_df, max_df=max_df)
            tfidf_matrix = vectorizer.fit_transform(competitor_texts)
            feature_names = vectorizer.get_feature_names_out()
            df_tfidf_competitors = pd.DataFrame(tfidf_matrix.toarray(), index=valid_competitor_sources, columns=feature_names)
        with st.spinner("Calculating TF-IDF scores for target content..."):
            target_tfidf_vector = vectorizer.transform([target_content])
            df_tfidf_target = pd.DataFrame(
                target_tfidf_vector.toarray(),
                index=[target_url if target_source_option=="Extract from URL" else "Target Content"],
                columns=feature_names
            )
        top_ngrams_competitors = {}
        for source in valid_competitor_sources:
            row = df_tfidf_competitors.loc[source]
            sorted_row = row.sort_values(ascending=False)
            top_ngrams = sorted_row.head(top_n)
            top_ngrams_competitors[source] = list(top_ngrams.index)
        content_gaps = {}
        for source, competitor_ngrams in top_ngrams_competitors.items():
            gap_ngrams = []
            for ngram in competitor_ngrams:
                if ngram in df_tfidf_target.columns:
                    competitor_score = df_tfidf_competitors.loc[source, ngram]
                    target_score = df_tfidf_target.iloc[0][ngram]
                    if competitor_score > target_score:
                        gap_ngrams.append(f"{ngram} (Competitor: {competitor_score:.3f}, Target: {target_score:.3f})")
                else:
                    competitor_score = df_tfidf_competitors.loc[source, ngram]
                    gap_ngrams.append(f"{ngram} (Competitor: {competitor_score:.3f}, Target: 0.000)")
            content_gaps[source] = gap_ngrams
        st.markdown("### Semantic Gap Analysis")
        st.markdown(f"**Target:** {target_url if target_source_option=='Extract from URL' else 'Pasted Target Content'}")
        all_data = {}
        for source, gap_ngrams in content_gaps.items():
            all_data[source] = gap_ngrams
            while len(all_data[source]) < top_n:
                all_data[source].append("")
        df_display = pd.DataFrame(all_data)
        st.dataframe(df_display)
        st.subheader("Semantic Gap Wordclouds")
        for source, gap_ngrams in content_gaps.items():
            site_gap_counts = {}
            for gap in gap_ngrams:
                if gap:
                    ngram_text = gap.split(" (")[0]
                    site_gap_counts[ngram_text] = site_gap_counts.get(ngram_text, 0) + 1
            if site_gap_counts:
                st.markdown(f"**Wordcloud for {source}:**")
                display_entity_wordcloud(site_gap_counts)
            else:
                st.write(f"No gap n‑grams for {source} to create a wordcloud.")
        combined_gap_counts = {}
        for gap_list in content_gaps.values():
            for gap in gap_list:
                if gap:
                    ngram_text = gap.split(" (")[0]
                    combined_gap_counts[ngram_text] = combined_gap_counts.get(ngram_text, 0) + 1
        if combined_gap_counts:
            st.subheader("Combined Semantic Gap Wordcloud for All Sites")
            display_entity_wordcloud(combined_gap_counts)
        else:
            st.write("No combined gap n‑grams to create a wordcloud.")

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
    competitor_source_option = st.radio(
        "Select competitor content source:",
        options=["Extract from URL", "Paste Content"],
        index=0,
        key="comp_source"
    )
    if competitor_source_option == "Extract from URL":
        competitor_input = st.text_area("Enter Competitor URLs (one per line):", key="comp_urls", value="")
        competitor_list = [url.strip() for url in competitor_input.splitlines() if url.strip()]
    else:
        st.markdown("Paste competitor content below. Separate each block with `---`.")
        competitor_input = st.text_area("Enter Competitor Content:", key="competitor_content", value="", height=200)
        competitor_list = [content.strip() for content in competitor_input.split('---') if content.strip()]
    st.subheader("Your Site")
    target_source_option = st.radio(
        "Select target content source:",
        options=["Extract from URL", "Paste Content"],
        index=0,
        key="target_source_cluster"
    )
    if target_source_option == "Extract from URL":
        target_url = st.text_input("Enter Your URL:", key="target_url", value="")
    else:
        target_text = st.text_area("Paste your content:", key="target_content", value="", height=200)
    st.subheader("Word Count Settings")
    n_value = st.selectbox("Select # of Words in Phrase:", options=[1, 2, 3, 4, 5], index=1, key="ngram_n")
    min_df = st.number_input("Minimum Frequency:", value=1, min_value=1, key="min_df_gap")
    max_df = st.number_input("Maximum Frequency:", value=1.0, min_value=0.0, step=0.1, key="max_df_gap")
    top_n = st.slider("Max # of Results per Competitor:", min_value=1, max_value=50, value=10, key="top_n_gap")
    st.subheader("Clustering Settings")
    algorithm = st.selectbox(
        "Select Clustering Type:", 
        options=["Kindred Spirit", "Affinity Stack"],
        key="clustering_algo_gap"
    )
    if algorithm == "Kindred Spirit":
        n_clusters = st.number_input("Number of Clusters:", min_value=1, value=5, key="kmeans_clusters_gap")
    elif algorithm == "Affinity Stack":
        n_clusters = st.number_input("Number of Clusters:", min_value=1, value=5, key="agg_clusters_gap")
    if st.button("Analyze & Cluster Gaps", key="gap_cluster_button"):
        competitor_texts = []
        valid_competitor_sources = []
        with st.spinner("Extracting competitor content..."):
            for source in competitor_list:
                if competitor_source_option == "Extract from URL":
                    text = extract_text_from_url(source)
                else:
                    text = source
                if text:
                    competitor_texts.append(text)
                    valid_competitor_sources.append(source)
                else:
                    st.warning(f"Could not extract content from: {source}")
        if target_source_option == "Extract from URL":
            target_content = extract_text_from_url(target_url)
            if not target_content:
                st.error("Could not extract content from the target URL.")
                return
        else:
            target_content = target_text
        if not competitor_texts:
            st.error("No competitor content was extracted.")
            return
        with st.spinner("Calculating TF‑IDF scores for competitors..."):
            vectorizer = TfidfVectorizer(ngram_range=(n_value, n_value), min_df=min_df, max_df=max_df)
            tfidf_matrix = vectorizer.fit_transform(competitor_texts)
            feature_names = vectorizer.get_feature_names_out()
            df_tfidf_competitors = pd.DataFrame(tfidf_matrix.toarray(), index=valid_competitor_sources, columns=feature_names)
        with st.spinner("Calculating TF‑IDF scores for target content..."):
            target_tfidf_vector = vectorizer.transform([target_content])
            df_tfidf_target = pd.DataFrame(
                target_tfidf_vector.toarray(),
                index=[target_url if target_source_option=="Extract from URL" else "Target Content"],
                columns=feature_names
            )
        gap_ngrams = set()
        for source in valid_competitor_sources:
            row = df_tfidf_competitors.loc[source]
            sorted_row = row.sort_values(ascending=False)
            top_ngrams = sorted_row.head(top_n)
            for ngram in top_ngrams.index:
                comp_score = df_tfidf_competitors.loc[source, ngram]
                target_score = df_tfidf_target.iloc[0][ngram] if ngram in df_tfidf_target.columns else 0.0
                if comp_score > target_score:
                    gap_ngrams.add(ngram)
        gap_ngrams = list(gap_ngrams)
        if not gap_ngrams:
            st.error("No gap n‑grams were identified. Consider adjusting your TF‑IDF parameters.")
            return
        st.markdown("### Top Phrases:")
        st.write(gap_ngrams)
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
        if algorithm == "Kindred Spirit":
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
        elif algorithm == "Affinity Stack":
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
        with st.spinner("Generating cluster visualization..."):
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings)
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=cluster_labels, cmap='viridis', s=100)
            for i, txt in enumerate(valid_gap_ngrams):
                ax.annotate(txt, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=8)
            ax.set_title("Semantic Opportunity")
            ax.set_xlabel("Topic Focus: Broad vs. Niche")
            ax.set_ylabel("Competitive Pressure: High vs. Low")
            st.pyplot(fig)

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
        a[href*='streamlit.io/cloud'], div._profileContainer_gzau3_53 { display: none !important; }
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
        "Entity Frequency Bar Chart",
        "Semantic Gap Analyzer",
        "Keyword Clustering",
        "Text Summarization",
        "Summarization & Comparison"
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
    elif tool == "Text Summarization":
        summarization_page()
    elif tool == "Summarization & Comparison":
        summarization_comparison_page()
    st.markdown("---")
    st.markdown("Powered by [The SEO Consultant.ai](https://theseoconsultant.ai)", unsafe_allow_html=True)

if __name__ == "__main__":
    main()







