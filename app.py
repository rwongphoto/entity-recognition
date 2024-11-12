# Core Pkgs
import streamlit as st 

# NLP Pkgs
import spacy_streamlit
import spacy
nlp = spacy.load("en_core_web_sm")

import os
#from PIL import Image

path = os.getcwd()
# Print the current working directory
# print("Current working directory: {0}".format(cwd))


def main():
    """A Simple NLP app with Spacy-Streamlit"""
    #nweh_logo = Image.open(os.path.join('nweh_logo_sm.jpg')) 
    #st.image(nweh_logo)
    st.title("PII QUEST **with** _spaCy_ **NLP**")
    
    


if __name__ == '__main__':
	main()
