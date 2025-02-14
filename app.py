import streamlit as st
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import spacy
import requests
from collections import Counter
import numpy as np
from typing import List, Tuple, Dict
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import io
from spacy import displacy
import os

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
            return None  # or raise the exception
    return nlp


def extract_text_from_url(url):
    """Extracts text from a URL using Selenium, handling JavaScript rendering."""
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
        wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "p, h1, h2, h3, li"))
        )

        page_source = driver.page_source
        driver.quit()
        soup = BeautifulSoup(page_source, "html.parser")

        for tag in soup.find_all(['header', 'footer', 'nav']):
            tag.decompose()

        all_relevant_tags = soup.find_all(
            ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'p'])
        text = ""
        for tag in all_relevant_tags:
            text += tag.get_text(separator=" ", strip=True) + "\n"
        return text

    except Exception as e:
        st.error(f"Error fetching or processing URL {url}: {e}")
        return None


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


def plot_entity_counts(entity_counts, top_n=50, title_suffix="", min_urls=2):
    """Plots the top N entity counts as a bar chart."""
    filtered_entity_counts = Counter({k: v for k, v in entity_counts.items() if v >= min_urls})

    most_common_entities = filtered_entity_counts.most_common(top_n)
    entity_labels = [f"{entity} ({label})" for (entity, label), count in most_common_entities]
    counts = [count for (entity, label), count in most_common_entities]

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.bar(entity_labels, counts, color=plt.cm.viridis(np.linspace(0, 1, len(entity_labels))))
    ax.set_xlabel("Entities (with Labels)", fontsize=12)
    ax.set_ylabel("Counts (Number of URLs)", fontsize=12)
    ax.set_title(f"Entity Topic Gap Analysis", fontsize=14)
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    return fig


def create_navigation_menu(logo_url):
    """Creates a top navigation menu."""
    menu_options = {
        "Home": "https://theseoconsultant.ai/",
        "About": "https://theseoconsultant.ai/about/",
        "Services": "https://theseoconsultant.ai/seo-consulting/",
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


def entity_analysis_page():
    """Original Entity Analysis Page."""
    st.header("Entity Topic Gap Analysis")
    st.markdown("Analyze content from multiple URLs to identify common entities.")

    urls_input = st.text_area("Enter URLs (one per line):",
                              """""")
    urls = [url.strip() for url in urls_input.splitlines() if url.strip()]

    exclude_url = st.text_input("Enter URL to exclude:",
                                 "")

    if st.button("Analyze"):
        if not urls:
            st.warning("Please enter at least one URL.")
            return

        with st.spinner("Extracting content and analyzing entities..."):
            nlp_model = load_spacy_model()
            if not nlp_model:
                return  # Stop if model loading failed

            exclude_text = extract_text_from_url(exclude_url)
            exclude_entities_set = set()
            if exclude_text:
                exclude_doc = nlp_model(exclude_text)
                exclude_entities_set = {ent.text.lower() for ent in exclude_doc.ents}

            all_entities = []
            entity_counts_per_url: Dict[str, Counter] = {}
            url_entity_counts: Counter = Counter()

            for url in urls:
                text = extract_text_from_url(url)
                if text:
                    entities = identify_entities(text, nlp_model)
                    entities = [(entity, label) for entity, label in entities if label != "CARDINAL"]

                    filtered_entities = [(entity, label) for entity, label in entities
                                         if entity.lower() not in exclude_entities_set]
                    entity_counts_per_url[url] = count_entities(filtered_entities)
                    all_entities.extend(filtered_entities)

                    for entity, label in set(filtered_entities):
                        url_entity_counts[(entity, label)] += 1

            filtered_url_entity_counts = Counter({k: v for k, v in url_entity_counts.items() if v >= 2})

            if url_entity_counts:
                fig = plot_entity_counts(url_entity_counts, top_n=50, title_suffix=" - Overall", min_urls=2)
                st.pyplot(fig)

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
                for url, entity_counts in entity_counts_per_url.items():
                    st.markdown(f"#### URL: {url}")
                    if entity_counts:
                        for (entity, label), count in entity_counts.most_common(50):
                            st.write(f"- {entity} ({label}): {count}")
                    else:
                        st.write("No relevant entities found.")

                st.markdown("### Overall Entity Counts (Excluding Entities from Exclude URL and CARDINAL Entities, Found in More Than One URL)")
                for (entity, label), count in filtered_url_entity_counts.most_common(50):
                    st.write(f"- {entity} ({label}): {count}")

            else:
                st.warning("No relevant entities found.")


def displacy_visualization_page():
    """Displacy Entity Visualization Page."""
    st.header("Named Entity Visualizer")
    st.markdown("Visualize named entities within your content.")

    url = st.text_input("Enter a URL to visualize entities:")
    if st.button("Visualize Entities"):
        if not url:
            st.warning("Please enter a URL.")
            return

        with st.spinner("Extracting text and visualizing entities..."):
            nlp_model = load_spacy_model()
            if not nlp_model:
                return

            text = extract_text_from_url(url)
            if text:
                doc = nlp_model(text)
                html = displacy.render(doc, style="ent", page=True)
                st.components.v1.html(html, height=600, scrolling=True) # Render HTML

            else:
                st.error("Could not extract text from the URL.")


def main():
    st.set_page_config(
        page_title="Named Entity Topic Analysis | The SEO Consultant.ai",
        page_icon=":bar_chart:",
        layout="wide"
    )
    logo_url = "https://theseoconsultant.ai/wp-content/uploads/2024/12/cropped-theseoconsultant-logo-2.jpg"
    create_navigation_menu(logo_url)

    nlp = load_spacy_model()
    if nlp is None:
        st.error("Failed to load spaCy model.  Check the logs for details.")
        return

    # Navigation
    page = st.sidebar.selectbox("Choose a Page:",
                                ("Entity Topic Gap Analysis", "DisplaCy Entity Visualization"))

    # Page routing
    if page == "Entity Topic Gap Analysis":
        entity_analysis_page()
    elif page == "DisplaCy Entity Visualization":
        displacy_visualization_page()

    st.markdown("---")
    st.markdown("Powered by [The SEO Consultant.ai](https://theseoconsultant.ai)",
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()
