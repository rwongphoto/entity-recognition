import streamlit as st
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import spacy
import requests
from collections import Counter
import numpy as np  # Import numpy
from typing import List, Tuple, Dict
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import io

# Move set_page_config to the top
st.set_page_config(
    page_title="Entity Topic Gap Analysis | The SEO Consultant.ai",
    page_icon=":bar_chart:"
)

# Load the spaCy model
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

def extract_text_from_url(url):
    """Extracts text from a URL using Selenium and BeautifulSoup, rendering JavaScript."""
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        # Add User-Agent for iPhone
        user_agent = "Mozilla/5.0 (iPhone; CPU iPhone OS 14_7_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.7.1 Mobile/15E148 Safari/604.1"
        chrome_options.add_argument(f"user-agent={user_agent}")

        # Ensure ChromeDriver is in your PATH or specify executable_path
        driver = webdriver.Chrome(options=chrome_options)

        driver.get(url)

        # Explicit wait for the presence of at least one of the desired elements
        wait = WebDriverWait(driver, 10)
        wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "p, h1, h2, h3, li"))
        )

        page_source = driver.page_source
        driver.quit()
        soup = BeautifulSoup(page_source, "html.parser")
       # Exclude header and footer divs
        for tag in soup.find_all(['header', 'footer']):
            tag.decompose()

        # Extract text from the remaining elements
        text = " ".join([p.get_text() for p in soup.find_all(["p", "h1", "h2", "h3", "li"])])
        return text
    except Exception as e:
        print(f"Error fetching or processing URL {url}: {e}")
        return None


def identify_entities(text, nlp_model):
    """Identifies named entities in the text."""
    doc = nlp_model(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities


def count_entities(entities: List[Tuple[str, str]]) -> Counter:
    """Counts named entities."""
    entity_counts = Counter()

    for entity, label in entities:  # Iterate through entities, including the label
        entity = entity.replace('\n', ' ').replace('\r', '')
        if len(entity) > 2 and label != "CARDINAL":  # Filter out short entities and CARDINAL entities
            entity_counts[(entity, label)] += 1  # Count based on (entity, label) tuple

    return entity_counts


def plot_entity_counts(entity_counts, top_n=50, title_suffix="", min_urls=2):
    """Plots the top N entity counts as a bar chart, including entity labels.
       Only includes entities found in at least `min_urls` URLs.
    """
    filtered_entity_counts = Counter({k: v for k, v in entity_counts.items() if v >= min_urls})

    most_common_entities = filtered_entity_counts.most_common(top_n)
    entity_labels = [f"{entity} ({label})" for (entity, label), count in most_common_entities] # Concatenate entity text and label for display
    counts = [count for (entity, label), count in most_common_entities]

    # Create a visually appealing color palette
    colors = plt.cm.viridis(np.linspace(0, 1, len(entity_labels))) # Use viridis colormap

    fig, ax = plt.subplots(figsize=(16, 12))  # Adjust figure size for better readability - increased size
    ax.bar(entity_labels, counts, color=colors)
    ax.set_xlabel("Entities (with Labels)", fontsize=12)
    ax.set_ylabel("Counts (Number of URLs)", fontsize=12)  # Updated ylabel
    ax.set_title(f"Entity Topic Gap Analysis", fontsize=14)
    ax.tick_params(axis='x', rotation=45, labelsize=10)  # Set fontsize for x axis labels

    plt.tight_layout()  # Adjust layout to prevent labels from overlapping
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add a subtle grid for better visualization

    return fig



def create_navigation_menu(logo_url):
    """Creates a top navigation menu for the Streamlit app with a logo above and centered."""

    menu_options = {
        "Home": "https://theseoconsultant.ai/",
        "About": "https://theseoconsultant.ai/about/",
        "Services": "https://theseoconsultant.ai/seo-consulting/",  # Flattened for top nav
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
          background-color: #f1f1f1; /* Adjust color as needed */
          display: flex;              /* Use flexbox */
          justify-content: center;    /* Horizontally center items */
          margin-bottom: 35px;        /* Add space below the menu */
        }

        .topnav a {
          float: left;
          display: block;
          color: black; /* Adjust color as needed */
          text-align: center;
          padding: 14px 16px;
          text-decoration: none;
        }

        .topnav a:hover {
          background-color: #ddd; /* Adjust color as needed */
          color: black; /* Adjust color as needed */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Create the top navigation menu
    menu_html = "<div class='topnav'>"
    for key, value in menu_options.items():
        menu_html += f"<a href='{value}' target='_blank'>{key}</a>"
    menu_html += "</div>"

    st.markdown(menu_html, unsafe_allow_html=True)


def main():

    logo_url = "https://theseoconsultant.ai/wp-content/uploads/2024/12/cropped-theseoconsultant-logo-2.jpg"
    create_navigation_menu(logo_url)

    st.title("Entity Topic Gap Analysis")
    st.markdown("Analyze content from multiple URLs to identify common entities and potential topic gaps.")

    # URL Input
    urls_input = st.text_area("Enter URLs (one per line):",
                              """https://www.zachnicholz.com/landscape-prints
https://aimeedanielson.com/
https://www.lijahhanley.com/collections/pacific-northwest-fine-art-landscape-photography
https://www.lucascometto.com/pnw-photography-guide
https://www.peterjamesphotogallery.com/large-pacific-northwest-photography
https://www.jessleephotos.com/gallery/pacific-northwest-photos/?srsltid=AfmBOorW3eqp8GyrNKpp9dK7maiwft1hOmeaMOJf1mJUiCQmfSj4zZC5
https://www.artofgregmartin.com/viewsofthepnw""")
    urls = [url.strip() for url in urls_input.splitlines() if url.strip()]

    exclude_url = st.text_input("Enter URL to exclude:",
                                 "https://www.rwongphoto.com/gallery/pacific-northwest-pictures/")

    if st.button("Analyze"):
        if not urls:
            st.warning("Please enter at least one URL.")
            return

        with st.spinner("Extracting content and analyzing entities..."):
            # 1. Extract entities from the exclude URL
            exclude_text = extract_text_from_url(exclude_url)
            exclude_entities_set = set()  # Use a set for efficient lookup
            if exclude_text:
                exclude_doc = nlp(exclude_text)
                exclude_entities_set = {ent.text.lower() for ent in exclude_doc.ents}


            all_entities = []
            entity_counts_per_url: Dict[str, Counter] = {}  # Store entity counts for each URL
            url_entity_counts: Counter = Counter() # NEW: Counter to hold the counts across *all* URLs.



            for url in urls:
                text = extract_text_from_url(url)
                if text:
                    entities = identify_entities(text, nlp)
                    # Filter out CARDINAL entities before counting
                    entities = [(entity, label) for entity, label in entities if label != "CARDINAL"]

                    # Count entities, excluding those in the exclude set
                    filtered_entities = [(entity, label) for entity, label in entities
                                         if entity.lower() not in exclude_entities_set]
                    entity_counts_per_url[url] = count_entities(filtered_entities)
                    all_entities.extend(filtered_entities) # Extend with the filtered list

                    # Update the *overall* counts correctly, even if the same entity appears multiple times in a single URL.
                    for entity, label in set(filtered_entities):  # Iterate through the *unique* entities in this URL
                        url_entity_counts[(entity, label)] += 1

            # Overall entity counts and plot
            # Now `url_entity_counts` holds the number of URLs each entity appears in.
            filtered_url_entity_counts = Counter({k: v for k, v in url_entity_counts.items() if v >= 2})

            if url_entity_counts: # Use the new Counter variable

                fig = plot_entity_counts(url_entity_counts, top_n=50, title_suffix=" - Overall", min_urls=2)  # Generate and display the overall bar chart - set top_n to 50
                st.pyplot(fig)
            else:
                st.warning("No relevant entities found.")



    st.markdown("---")
    st.markdown(
        "Powered by [The SEO Consultant.ai](https://theseoconsultant.ai)",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
