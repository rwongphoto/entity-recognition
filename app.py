# Core Pkgs
import streamlit as st 

# NLP Pkgs
import spacy_streamlit
import spacy

import os
#from PIL import Image

path = os.getcwd()
# Print the current working directory
# print("Current working directory: {0}".format(cwd))


def main():
    """A Simple NLP app with Spacy-Streamlit"""
    st.title("PII QUEST with spaCy NLP")
    st.markdown('**spaCy is an open-source natural language processing (NLP) library**')

    menu = ["Named Entity Recognision", "Tokenization"]
    choice = st.sidebar.selectbox("Menu", menu)

    raw_text = st.text_area("Sample Text", "Your default text here...")

    # Load the spaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        st.error(f"Error loading spaCy model: {e}")
        return  # exit early if model can't be loaded

    if choice == "Named Entity Recognision":
        st.subheader("Named Entity Recognision")
        docx = nlp(raw_text)
        if st.button("Analyse"):
            spacy_streamlit.visualize_ner(docx, labels=nlp.get_pipe('ner').labels)

    elif choice == "Tokenization":
        st.subheader("Tokenization")
        docx = nlp(raw_text)
        if st.button("Analyse"):
            spacy_streamlit.visualize_tokens(docx, attrs=['text', 'pos_', 'dep_', 'ent_type_'])



if __name__ == '__main__':
	main()
