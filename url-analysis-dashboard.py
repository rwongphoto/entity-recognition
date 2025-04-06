import random
import streamlit as st
st.set_page_config(layout="wide", page_title="URL Analysis Dashboard")
import pandas as pd
from bs4 import BeautifulSoup
import requests
import time
import spacy
import textstat
from sentence_transformers import SentenceTransformer, util
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException, TimeoutException
import extruct
import google.generativeai as genai
import re
from urllib.parse import urlparse

# -------------------- User Agent List --------------------
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.2151.97",  # Edge
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1",  # Safari iPhone
    "Mozilla/5.0 (iPad; CPU OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1",  # Safari iPad
    "Mozilla/5.0 (Android 14; Mobile; rv:121.0) Gecko/121.0 Firefox/121.0",  # Firefox Android
    "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",  # Chrome Android
]

def get_random_user_agent():
    """Returns a randomly selected user agent from the list."""
    return random.choice(USER_AGENTS)

# -------------------- Model and Helper Functions --------------------
@st.cache_resource
def load_spacy_model(model_name="en_core_web_sm"):
    try:
        return spacy.load(model_name)
    except OSError:
        st.error(f"Spacy model '{model_name}' not found. Please download it: python -m spacy download {model_name}")
        st.stop()

nlp = load_spacy_model()

@st.cache_resource(show_spinner="Loading SentenceTransformer model...")
def load_sentence_transformer(model_name='sentence-transformers/all-MiniLM-L6-v2'):
    try:
        return SentenceTransformer(model_name)
    except Exception as e:
        st.error(f"Error loading Sentence Transformer model '{model_name}': {e}")
        st.stop()

similarity_model = load_sentence_transformer()

@st.cache_resource
def setup_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920x1080")
    options.binary_location = "/usr/bin/chromium-browser"  # Use the system-installed Chromium
    
    # Use a random user agent from your expanded list
    user_agent = get_random_user_agent()
    options.add_argument(f'user-agent={user_agent}')
    
    try:
        # Use the system's chromedriver which should be in the PATH after installing from packages.txt
        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(30)
        return driver
    except Exception as e:
        st.error(f"Error setting up Selenium WebDriver: {e}")
        st.stop()


    import logging
    logging.getLogger('WDM').setLevel(logging.NOTSET)
    options.add_argument('--log-level=3')
    options.add_experimental_option('excludeSwitches', ['enable-logging'])

    try:
        service = ChromeService(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        driver.set_page_load_timeout(30)
        return driver
    except ValueError as ve:
        st.error(f"WebDriver Manager Error: {ve}")
        st.stop()
    except Exception as e:
        st.error(f"Error setting up Selenium WebDriver: {e}")
        st.stop()


def get_page_content_selenium(url, driver):
    try:
        driver.get(url)
        time.sleep(random.uniform(2, 5))
        html_content = driver.page_source
        if not html_content or "challenge-platform" in html_content:
            st.warning(f"Potentially blocked or empty page for {url}.")
        soup = BeautifulSoup(html_content, 'lxml')
        return soup, html_content, "Success"
    except TimeoutException:
        st.warning(f"Timeout loading {url}.")
        return None, None, "Timeout"
    except WebDriverException as e:
        st.warning(f"WebDriver error fetching {url}: {str(e)[:100]}...")
        return None, None, "WebDriver Error"
    except Exception as e:
        st.warning(f"Error fetching {url} with Selenium: {e}.")
        return None, None, "Fetch Error"

def extract_meta_title(soup):
    if soup and soup.title and soup.title.string:
        return soup.title.string.strip()
    og_title = soup.find('meta', property='og:title')
    if og_title and og_title.get('content'):
        return og_title['content'].strip()
    return "Not Found"

def extract_h1(soup):
    if soup:
        h1 = soup.find('h1')
        if h1:
            return h1.get_text(strip=True)
    return "Not Found"

def extract_word_counts(soup):
    if not soup:
        return 0, 0
    all_text = soup.get_text(separator=' ', strip=True)
    total_words = len(re.findall(r'\b\w+\b', all_text))
    content_text = []
    content_tags = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td', 'th', 'a'])
    for tag in content_tags:
        content_text.append(tag.get_text(separator=' ', strip=True))
    for img in soup.find_all('img', alt=True):
        if img['alt']:
            content_text.append(img['alt'])
    content_string = ' '.join(content_text)
    content_words = len(re.findall(r'\b\w+\b', content_string))
    return total_words, content_words

def calculate_readability(soup):
    if not soup:
        return "N/A"
    main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(r'(content|main|body|post)', re.I)) or soup.body
    if not main_content:
        main_content = soup.body
        if not main_content:
            return "N/A"
    text = main_content.get_text(separator=' ', strip=True)
    if not text or text.isspace():
        return "N/A"
    try:
        if textstat.lexicon_count(text, removepunct=True) < 100:
            return f"Low ({textstat.flesch_kincaid_grade(text):.1f})"
        return f"{textstat.flesch_kincaid_grade(text):.1f}"
    except Exception:
        return "N/A"

def calculate_cosine_similarity(soup, keyword, model):
    if not soup or not keyword:
        return "N/A"
    content_text = []
    content_tags = soup.find_all(['p', 'h1', 'h2', 'h3', 'li', 'th', 'td'])
    for tag in content_tags:
        content_text.append(tag.get_text(separator=' ', strip=True))
    title = extract_meta_title(soup)
    h1 = extract_h1(soup)
    if title != "Not Found":
        content_text.insert(0, title)
    if h1 != "Not Found":
        content_text.insert(0, h1)
    page_text = ' '.join(content_text)
    if not page_text or page_text.isspace():
        return 0.0
    try:
        page_embedding = model.encode(page_text, convert_to_tensor=True)
        keyword_embedding = model.encode(keyword, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(page_embedding, keyword_embedding)
        return f"{cosine_scores.item():.4f}"
    except Exception as e:
        st.warning(f"Error calculating similarity: {e}")
        return "Error"

def extract_entities(soup, spacy_nlp):
    if not soup:
        return 0
    main_content = soup.find('main') or soup.find('article') or soup.body
    if not main_content:
        return 0
    text = main_content.get_text(separator=' ', strip=True)
    if not text or text.isspace():
        return 0
    doc = spacy_nlp(text[:spacy_nlp.max_length])
    unique_entities = set([ent.text.lower() for ent in doc.ents])
    return len(unique_entities)

def check_list_table_presence(soup):
    if not soup:
        return "No", "No", "No"
    has_ol = "Yes" if soup.find('ol') else "No"
    has_ul = "Yes" if soup.find('ul') else "No"
    has_table = "Yes" if soup.find('table') else "No"
    return has_ol, has_ul, has_table

def count_media(soup):
    if not soup:
        return 0, 0
    img_count = len(soup.find_all('img'))
    video_count = len(soup.find_all('video')) + len(soup.find_all('iframe', src=re.compile(r'(youtube\.com|vimeo\.com)', re.I)))
    return img_count, video_count

def extract_schema_types(html_content):
    if not html_content:
        return "N/A"
    try:
        data = extruct.extract(html_content, syntaxes=['json-ld', 'microdata', 'rdfa'])
        schema_types = set()
        for syntax in data:
            if data[syntax]:
                for item in data[syntax]:
                    type_info = item.get('@type')
                    if type_info:
                        if isinstance(type_info, list):
                            schema_types.update(type_info)
                        else:
                            schema_types.add(type_info)
        return ', '.join(sorted(list(schema_types))) if schema_types else "None Found"
    except Exception as e:
        st.warning(f"Error extracting schema: {e}")
        return "Extraction Error"

@st.cache_data(ttl=3600)
def get_gemini_analysis(_analysis_data, _keyword, _target_url=None):
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt_parts = [
            f"Analyze the following SEO and content data for the URL: {_analysis_data['URL']}",
            f"Target Keyword: '{_keyword}'"
        ]
        if _target_url:
            prompt_parts.append(f"Compare against Target URL (if provided data allows): {_target_url}")
        prompt_parts.append("\nData:")
        for key, value in _analysis_data.items():
            prompt_parts.append(f"- {key}: {value}")
        prompt_parts.append(
            "\nBased ONLY on the provided data, assess the URL's semantic SEO relevance for the target keyword. "
            "Provide actionable insights for improvement. Focus on content structure, keyword alignment (based on similarity score and entities), readability, and technical elements like schema. Be concise and specific."
            "\n\nAnalysis and Recommendations:"
        )
        prompt = "\n".join(prompt_parts)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        if "API key not valid" in str(e):
            st.error("Gemini API Key is invalid. Please check your Streamlit secrets.")
            return "Error: Invalid API Key."
        else:
            st.error(f"Error calling Gemini API: {e}")
            return f"Error during Gemini analysis: {e}"

# -------------------- Streamlit App UI --------------------
st.title("📊 URL Analysis Dashboard")
st.markdown("Enter multiple URLs (one per line) and a target keyword to analyze their semantic SEO potential.")

with st.sidebar:
    st.header("Configuration")
    url_input = st.text_area("Enter URLs (one per line)", height=150, placeholder="https://example.com/page1\nhttps://example.com/page2")
    keyword = st.text_input("Target Keyword", placeholder="e.g., best semantic seo tools")
    target_url = st.text_input("Target URL (Optional, for comparison context)", placeholder="e.g., https://your-target-page.com")
    analyze_button = st.button("🚀 Analyze URLs", type="primary")
    st.markdown("---")
    st.markdown("Powered by Streamlit, Selenium, Sentence Transformers, SpaCy, and Gemini.")

if analyze_button:
    urls = [url.strip() for url in url_input.splitlines() if url.strip()]
    if not urls:
        st.warning("Please enter at least one URL.")
    elif not keyword:
        st.warning("Please enter a target keyword.")
    else:
        gemini_enabled = 'GEMINI_API_KEY' in st.secrets
        if not gemini_enabled:
            st.error("Gemini API Key not found in Streamlit Secrets (`.streamlit/secrets.toml`). Cannot perform Gemini analysis.")

        results = []
        total_urls = len(urls)
        progress_bar = st.progress(0)
        status_text = st.empty()
        driver = None
        try:
            with st.spinner("Setting up web driver..."):
                driver = setup_driver()
                if driver is None:
                    st.error("Failed to initialize WebDriver. Analysis cannot proceed.")
                    st.stop()
            for i, url in enumerate(urls):
                status_text.text(f"Analyzing URL {i+1}/{total_urls}: {url}")
                progress_bar.progress((i + 1) / total_urls)
                parsed_url = urlparse(url)
                if not all([parsed_url.scheme, parsed_url.netloc]):
                    st.warning(f"Invalid URL format: {url}. Skipping.")
                    results.append({
                        "URL": url,
                        "Status": "Invalid URL",
                        "Title": "N/A",
                        "H1": "N/A",
                        "Total Words": "N/A",
                        "Content Words": "N/A",
                        "Cosine Sim.": "N/A",
                        "Readability (FKG)": "N/A",
                        "Unique Entities": "N/A",
                        "OL": "N/A",
                        "UL": "N/A",
                        "Table": "N/A",
                        "Images": "N/A",
                        "Videos": "N/A",
                        "Schema Types": "N/A"
                    })
                    continue
                soup, html_content, fetch_status = get_page_content_selenium(url, driver)
                if soup:
                    title = extract_meta_title(soup)
                    h1 = extract_h1(soup)
                    total_words, content_words = extract_word_counts(soup)
                    readability = calculate_readability(soup)
                    similarity = calculate_cosine_similarity(soup, keyword, similarity_model)
                    entities = extract_entities(soup, nlp)
                    ol, ul, table = check_list_table_presence(soup)
                    img_count, vid_count = count_media(soup)
                    schema = extract_schema_types(html_content)
                    results.append({
                        "URL": url,
                        "Status": fetch_status,
                        "Title": title,
                        "H1": h1,
                        "Total Words": total_words,
                        "Content Words": content_words,
                        "Cosine Sim.": similarity,
                        "Readability (FKG)": readability,
                        "Unique Entities": entities,
                        "OL": ol,
                        "UL": ul,
                        "Table": table,
                        "Images": img_count,
                        "Videos": vid_count,
                        "Schema Types": schema
                    })
                else:
                    results.append({
                        "URL": url,
                        "Status": fetch_status,
                        "Title": "N/A",
                        "H1": "N/A",
                        "Total Words": "N/A",
                        "Content Words": "N/A",
                        "Cosine Sim.": "N/A",
                        "Readability (FKG)": "N/A",
                        "Unique Entities": "N/A",
                        "OL": "N/A",
                        "UL": "N/A",
                        "Table": "N/A",
                        "Images": "N/A",
                        "Videos": "N/A",
                        "Schema Types": "N/A"
                    })
            status_text.text("Analysis Complete!")
            if driver:
                driver.quit()

            if results:
                df = pd.DataFrame(results)
                cols_order = ["URL", "Status", "Cosine Sim.", "Title", "H1", "Content Words", "Total Words", "Readability (FKG)", "Unique Entities", "Schema Types", "OL", "UL", "Table", "Images", "Videos"]
                df_cols = [col for col in cols_order if col in df.columns]
                df = df[df_cols]
                st.subheader("📊 Analysis Results")
                st.dataframe(df, use_container_width=True)

                if gemini_enabled:
                    st.subheader("🤖 Gemini Semantic SEO Analysis")
                    st.markdown("Select a URL from the table above to get AI-powered insights.")
                    results_dict = {res['URL']: res for res in results if res.get('Status') == 'Success'}
                    successful_urls = list(results_dict.keys())
                    if successful_urls:
                        selected_url_gemini = st.selectbox("Select URL for Gemini Analysis", options=successful_urls)
                        if selected_url_gemini:
                            data_for_gemini = results_dict[selected_url_gemini]
                            with st.spinner(f"🤖 Calling Gemini for {selected_url_gemini}..."):
                                gemini_insights = get_gemini_analysis(data_for_gemini, keyword, target_url)
                                st.markdown(f"**Analysis for:** `{selected_url_gemini}`")
                                st.markdown(gemini_insights)
                    else:
                        st.info("No URLs were successfully analyzed to provide Gemini insights.")
                else:
                    st.info("Gemini analysis is disabled. Please configure your API key in Streamlit Secrets.")
            else:
                st.info("No results to display.")
        except Exception as e:
            st.error(f"An unexpected error occurred during analysis: {e}")
            if driver:
                driver.quit()

st.sidebar.markdown("---")
st.sidebar.caption("Note: Analysis time depends on the number of URLs and website loading speeds. Ensure Chrome is installed.")

