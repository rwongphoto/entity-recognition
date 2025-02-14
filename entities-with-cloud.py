import streamlit as st
import spacy
import networkx as nx
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from typing import List, Tuple, Dict
from collections import Counter
import os  # Import os module
from selenium.webdriver.support.ui import WebDriverWait #Added the import to use webDriverWait
from selenium.webdriver.support import expected_conditions as EC #Added the import to use webDriverWait
from selenium.webdriver.common.by import By #Added the import to use webDriverWait
import matplotlib.pyplot as plt # matplotlib needed for barcharts
import pandas as pd # pandas needed for tables

# ------------------------------------
# Global Variables & Utility Functions
# ------------------------------------

logo_url = "https://theseoconsultant.ai/wp-content/uploads/2024/12/cropped-theseoconsultant-logo-2.jpg"

# Global spacy model variable
nlp = None

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

def display_entity_barchart(entity_counts, top_n=20):
    """Displays a bar chart of the top N most frequent entities."""
    most_common_entities = entity_counts.most_common(top_n)
    entity_names = [entity for entity, count in most_common_entities]
    counts = [count for entity, count in most_common_entities]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(entity_names, counts)
    ax.set_xlabel("Entities")
    ax.set_ylabel("Frequency")
    ax.set_title("Top {} Named Entities".format(top_n))
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()  # Adjust layout to prevent labels from overlapping
    st.pyplot(fig)

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
                    entity_counts_per_url = count_entities(filtered_entities)
                    all_entities.extend(filtered_entities)

                    for entity, label in set(filtered_entities):
                        url_entity_counts[(entity, label)] += 1

            filtered_url_entity_counts = Counter({k: v for k, v in url_entity_counts.items() if v >= 2})

            if url_entity_counts:
                st.markdown("### Overall Entity Counts (Excluding Entities from Exclude URL and CARDINAL Entities, Found in More Than One URL)")
                for (entity, label), count in filtered_url_entity_counts.most_common(50):
                    st.write(f"- {entity} ({label}): {count}")

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

                fig = plot_entity_counts(url_entity_counts, top_n=50, title_suffix=" - Overall", min_urls=2)
                st.pyplot(fig)

            else:
                st.warning("No relevant entities found.")


def displacy_visualization_page():
    """Displacy Entity Visualization Page."""
    st.header("Entity Visualizer")
    st.markdown("Visualize named entities within your content.")

    # URL Input
    url = st.text_input("Enter a URL (Optional):", key="displacy_url", value="")
    use_url = st.checkbox("Use URL for Text Input", key="displacy_use_url")

    # Text Area - Disable if URL is used
    text = st.text_area("Enter Text:", key="displacy_text", height=300, value="Paste your text here.", disabled=use_url)

    if st.button("Visualize Entities", key="displacy_button"):
        # Check if using URL or Text
        if use_url:
            if url:
                with st.spinner(f"Extracting text from {url}..."):
                    text = extract_text_from_url(url)
                    if not text:
                        st.error(f"Could not extract text from {url}. Please check the URL and make sure it is reachable.")
                        return  # Stop execution
            else:
                st.warning("Please enter a URL to extract the text or uncheck `Use URL for Text Input` to paste text directly.")
                return  # Stop execution

        elif not text:
            st.warning("Please enter either text or a URL.")
            return  # Stop execution

        with st.spinner("Visualizing entities..."):
            nlp_model = load_spacy_model()
            if not nlp_model:
                return

            doc = nlp_model(text)
            html = displacy.render(doc, style="ent", page=True)
            st.components.v1.html(html, height=600, scrolling=True) # Render HTML


def named_entity_barchart_page():
    """Page to generate a bar chart of named entities and list them by URL."""
    st.header("Named Entity Frequency Bar Chart")
    st.markdown("Generate a bar chart from the most frequent named entities and list them by URL.")

    # Select between Text and URL
    text_source = st.radio(
        "Select text source:",
        ('Enter Text', 'Enter URLs'),
        key="barchart_text_source"
    )

    text = None
    urls = None

    if text_source == 'Enter Text':
        text = st.text_area("Enter Text:", key="barchart_text", height=300, value="Paste your text here.")
    else:  # Using URLs:
        urls_input = st.text_area("Enter URLs (one per line):", key="barchart_url", value="")
        urls = [url.strip() for url in urls_input.splitlines() if url.strip()]

    if st.button("Generate Bar Chart", key="barchart_button"):
        all_text = ""
        entity_texts_by_url = {}  # To store entities for each URL
        # Validation and text retrieval
        if text_source == 'Enter Text':
            if not text:
                st.warning("Please enter the text to proceed.")
                return
            all_text = text
           # entity_texts_by_url["Input Text"] = text #No need to set URL in code block
        else:  # Using URLs
            if not urls:
                st.warning("Please enter at least one URL.")
                return

            for url in urls:
                extracted_text = extract_text_from_url(url)
                if not extracted_text:
                    st.warning(f"Couldn't grab the text from {url}...")
                    return

               
                entity_texts_by_url[url] = extracted_text #stores URL, extracted texts for that url and prints to screen when generated

        with st.spinner("Analyzing entities and generating bar chart..."):
            nlp_model = load_spacy_model() #Load the Spacy model

            if not nlp_model:
                st.error("Failed to load spaCy model. Aborting.")
                return

            all_entities = []
            entity_texts_list= [] #Create a list that will hold all texts
            if text_source == 'Enter Text':
                entities = identify_entities(all_text, nlp_model)
                entity_texts = [entity[0] for entity in entities]

                # Update entity_texts_by_url with just what the user put in the text, and count for overall graph
                entity_texts_by_url["Input Text"] = entity_texts
                entity_counts= Counter([item for item in entity_texts])
                # print(entity_texts_by_url["Input Text"])

                fig, ax = plt.subplots(figsize=(10, 6))  # Create a figure and an axes.

                ax.bar(entity_counts.keys(), entity_counts.values())
                ax.set_xlabel("Entities")
                ax.set_ylabel("Frequency")
                ax.set_title("Frequency of Entities (from Inputted Text) ")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)

            else: #If source are URLs
                
                entity_counts_per_url: Dict[str, Counter] = {}  # Per-URL entity counts
                #All entities for URL
                for url in urls:
                    text = extract_text_from_url(url)
                    entities = identify_entities(text, nlp_model)
                    entity_texts_temp = [entity[0] for entity in entities]
                    entity_counts_per_url[url] = Counter(entity_texts_temp)
                    entity_texts_list.extend(entity_texts_temp)

                #Combined totals:
                entity_counts = Counter(entity_texts_list)
                #Now create the graph and store the data:

                 # Plot the graph - combined
                fig, ax = plt.subplots(figsize=(10, 6))
                entity_labels = list(entity_counts.keys())
                entity_values = list(entity_counts.values())
                ax.bar(entity_labels, entity_values)
                ax.set_xlabel("Entities")
                ax.set_ylabel("Frequency")
                ax.set_title("Frequency of Entities (from Multiple URLs)")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)

                # Display entities with counts for each URL
                st.subheader("Entities per URL")
                for url, counts in entity_counts_per_url.items(): #Loop
                  if not counts: #If there wasn't a relevant entity to be found, notificate
                    st.write(f"No entities to load")
                    return
                  st.write(f"The entities for {url}:") #Print URL header
                  for entity, count in counts.most_common(): #Get 50 most common
                    st.write(f"  - {entity}: {count}") #Print to screen

def main():
    st.set_page_config(
        page_title="Named Entity Analysis | The SEO Consultant.ai",
        page_icon=":pencil:",
        layout="wide"
    )

    logo_url = "https://theseoconsultant.ai/wp-content/uploads/2024/12/cropped-theseoconsultant-logo-2.jpg"
    create_navigation_menu(logo_url)

    nlp = load_spacy_model()
    if nlp is None:
        st.error("Failed to load spaCy model. Check the logs for details.")
        return

    # Navigation
    st.sidebar.header("Named Entity Recognition")
    page = st.sidebar.selectbox("Switch Tool:",
                                ("Entity Topic Gap Analysis", "Entity Visualizer", "Entity Frequency Bar Chart"))

    # Page routing
    if page == "Entity Topic Gap Analysis":
        entity_analysis_page()
    elif page == "Entity Visualizer":
        displacy_visualization_page()
    elif page == "Entity Frequency Bar Chart":
        named_entity_barchart_page()

    st.markdown("---")
    st.markdown(
        "Powered by [The SEO Consultant.ai](https://theseoconsultant.ai)",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
