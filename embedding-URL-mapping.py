import streamlit as st
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from PIL import Image, ImageDraw, ImageFont
import cairosvg
import io
import pandas as pd
from typing import List, Tuple, Dict
import os
import requests  # Import requests
from bs4 import BeautifulSoup #To be able to get content from the URL
from selenium import webdriver #To be able to find javascript content

# ------------------------------------
# Global Variables & Utility Functions
# ------------------------------------

logo_url = "https://theseoconsultant.ai/wp-content/uploads/2024/12/cropped-theseoconsultant-logo-2.jpg"

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
        response = requests.get(url, timeout=15)  # Timeout after 15 seconds

        # Check if the request was successful
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")

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

        else:
            st.error(f"Failed to retrieve content from {url}. Status code: {response.status_code}")
            return None

    except requests.Timeout:
        st.error(f"Request timed out for URL {url}. The page took too long to respond.")
        return None
    except requests.RequestException as e:
        st.error(f"Request failed for URL {url}: {e}")
        return None

def get_embedding(text, model, tokenizer):
    """Generates a BERT embedding for the given text."""
    tokenizer.pad_token = tokenizer.unk_token
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def url_recommendation_page():
    st.header("URL Linking Recommendations")
    st.markdown("Generate URL linking recommendations based on vector embeddings and cosine similarity.")

    # URL Input
    source_code = st.radio(
        "Select source:",
        ('Enter URLs', 'Upload a Sitemap'),
        key="urls_text_source"
    )

    urls = []

    if source_code == 'Enter URLs':
         urls_input = st.text_area("Enter URLs (one per line):", key="heatmap_url", value="")
         urls = [url.strip() for url in urls_input.splitlines() if url.strip()]#Get URL
    else: # SITEMAP
        st.info("Feature still under development. It can only use manual URLs.")
        st.stop() #Code still under construction.

    similarity_threshold = st.slider("Similarity Threshold:", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
    if st.button("Generate Recommendations"):
        if not urls:
            st.warning("Please enter at least one URL.")
            return

        st.info("REMINDER: Functionality is extremely outdated and not guaranteed to run")
        # Initialize model (only once)
        tokenizer, model = initialize_bert_model()

        with st.spinner("Extracting content and calculating similarities..."):
            url_embeddings = {}
            url_texts= {}
            for url in urls:
                text = extract_text_from_url(url)
                if text:
                    url_embeddings[url] = get_embedding(text, model, tokenizer)
                    url_texts[url]=text #Store for checking purposes.
                else:
                    st.write(f"Could not extract text from {url}")
                    url_embeddings[url] = None

            # Calculate cosine similarity matrix
            similarity_matrix = np.zeros((len(urls), len(urls)))
            for i in range(len(urls)):
                for j in range(len(urls)):
                    if i != j and url_embeddings[urls[i]] is not None and url_embeddings[urls[j]] is not None:
                        similarity_matrix[i, j] = cosine_similarity(url_embeddings[urls[i]], url_embeddings[urls[j]])[0][0]

            # Prepare recommendations
            recommendations = []
            for i in range(len(urls)):
                for j in range(len(urls)):
                    if i != j and similarity_matrix[i, j] >= similarity_threshold:
                        recommendations.append((urls[i], urls[j], similarity_matrix[i, j]))

            # Prepare data for DataFrame
            recommendation_data = []
            for source, target, score in recommendations:
                recommendation_data.append({'Source URL': source, 'Target URL': target, 'Similarity Score': score})

            df = pd.DataFrame(recommendation_data) #Format for data
            st.write("Link Recommendations:")

            st.dataframe(df)

            csv = df.to_csv(index=False) #format to CSV
            st.download_button(
               label="Download CSV of Recommendations",
               data=csv,
               file_name='url_recommendations.csv',
               mime='text/csv',
           ) # Download to allow for easier export
        st.info("REMINDER: Functionality is extremely outdated and not guaranteed to run. Selenium and BeautiulSoup4 no longer properly render JS requests or modern html types.")


def main():
    st.set_page_config(
        page_title="URL Linking Recommendations | The SEO Consultant.ai",
        page_icon=":link:",
        layout="wide"
    )

    logo_url = "https://theseoconsultant.ai/wp-content/uploads/2024/12/cropped-theseoconsultant-logo-2.jpg"
    create_navigation_menu(logo_url) # call before main function


    # Navigation - Set to be on the side bar:
    st.sidebar.header("Link Recommendation Tools")
    page = st.sidebar.selectbox( "Choose a tool:",[ "URL Linking Recommendations" ])#Call to add a selectbox

    if page == "URL Linking Recommendations":
        url_recommendation_page() #Call url function

    st.markdown("---")
    st.markdown(
        "Powered by [The SEO Consultant.ai](https://theseoconsultant.ai)",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()