def named_entity_barchart_page():
    """Page to generate a bar chart of named entities and list them by URL."""
    st.header("Named Entity Frequency Bar Chart")
    st.markdown("Generate a bar chart from the most frequent named entities and list them by URL.")

    # Select between Text and URL input
    text_source = st.radio(
        "Select text source:",
        ('Enter Text', 'Enter URLs'),
        key="barchart_text_source"
    )

    text = None
    urls = None

    if text_source == 'Enter Text':
        text = st.text_area("Enter Text:", key="barchart_text", height=300, value="Paste your text here.")
    else:  # Using URLs
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
            entity_texts_by_url["Input Text"] = text
        else:  # Using URLs
            if not urls:
                st.warning("Please enter at least one URL.")
                return

            with st.spinner("Extracting text from URLs..."):
                for url in urls:
                    extracted_text = extract_text_from_url(url)
                    if extracted_text:
                        all_text += extracted_text + "\n"
                        entity_texts_by_url[url] = extracted_text
                    else:
                        st.warning(f"Could not extract from {url}...")
        
        with st.spinner("Analyzing entities and generating bar chart..."):
            nlp_model = load_spacy_model()
            if not nlp_model:
                st.error("Could not load spaCy model.  Aborting.")
                return

            all_entities = []
            if text_source == 'Enter Text':
                 entities = identify_entities(all_text, nlp_model)
                 entity_texts = [entity[0] for entity in entities] #Extract just the text from tuples
            else:
              entity_texts = [ ]#Initalized for scope
              all_entities = {}
              entity_counts_per_url: Dict[str, Counter] = {}  # Store entity counts for each URL
              url_entity_counts: Counter = Counter() # Counter to hold the counts across *all* URLs.
              for url in urls:
                    text = extract_text_from_url(url) # extract text from url
                    entities = identify_entities(text, nlp_model) # identify the named entity
                  #  entities = [(entity, label) for entity, label in entities if label != "CARDINAL"]
                    filtered_entities = [(entity, label) for entity, label in entities] #Create a list of the extracted labels
                    entity_counts_per_url[url] = count_entities(filtered_entities) #counts per Urls
                    url_entity_counts.update(entity_counts_per_url[url]) #overall entities
                    all_entities.extend(filtered_entities)
              entity_counts_no_label = Counter([entity for entity, label in all_entities])
              entity_texts = [entity for entity in entity_counts_no_label ]#Extract just the text from tuples
              if not entity_texts:
                st.warning("No relevant entities found for the current URL(s)")
                return

            # Count entities to display data and frequency
            entity_counts = Counter(entity_texts)

            if len(entity_counts) > 0:
                display_entity_barchart(entity_counts)  # Display the chart


            else:
               st.warning("No relevant entities found. Please check your text or URL(s).")

           # Now Display entities extracted for URLs
            st.markdown("### Entities Per URL")
            if text_source != 'Enter Text': # check if the source is for text or not
               for url, entity_counts in entity_counts_per_url.items():
                    st.markdown(f"#### URL: {url}")
                    if entity_counts:
                        for (entity, label), count in entity_counts.most_common(50):
                            st.write(f"- {entity} ({label}): {count}")
                    else:
                        st.write("No relevant entities found.")

def main():
    st.set_page_config(
        page_title="Named Entity Recognition | The SEO Consultant.ai",
        page_icon=":pencil:",  # Use a pencil emoji here
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
