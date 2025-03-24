import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import Qdrant  # Corrected import
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains import create_retrieval_chain  # Corrected import
import os

def setup_retrieval_qa_chain(qdrant_url, qdrant_api_key, openai_api_key, collection_name):
    """Sets up and returns the RetrievalQA chain."""
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        vectorstore = Qdrant(client=client, collection_name=collection_name, embeddings=embeddings)
        retriever = vectorstore.as_retriever()

        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=openai_api_key)

        # Create a chain that remembers the chat history
        retriever_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
        ])
        history_aware_retriever = create_history_aware_retriever(llm, retriever, retriever_prompt)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the user's questions based on the below context:\n\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])
        document_chain = create_stuff_documents_chain(llm, prompt)

        # Use create_retrieval_chain
        qa_chain = create_retrieval_chain(history_aware_retriever, document_chain)

        st.success("RetrievalQA chain created successfully.")
        return qa_chain

    except Exception as e:
        st.error(f"Failed to create RetrievalQA chain: {e}")
        return None

def main():
    st.title("How can I answer your SEO or marketing questions?")

    # --- Load Secrets ---
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    qdrant_url = st.secrets["QDRANT_CLOUD_URL"]
    qdrant_api_key = st.secrets["QDRANT_CLOUD_API_KEY"]
    # Retrieve the collection name from secrets to avoid public display
    collection_name = st.secrets["QDRANT_COLLECTION_NAME"]

    # --- Chat Interface ---
    # Initialize chat history and chain outside the input block
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "qa_chain" not in st.session_state:
        # Set environment variable.
        os.environ["OPENAI_API_KEY"] = openai_api_key
        st.session_state.qa_chain = setup_retrieval_qa_chain(qdrant_url, qdrant_api_key, openai_api_key, collection_name)

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            if st.session_state.qa_chain:  # Check if qa_chain is initialized
                try:
                    result = st.session_state.qa_chain.invoke({
                        "input": prompt, 
                        "chat_history": [(message["role"], message["content"]) for message in st.session_state.messages[:-1]]
                    })
                    full_response = result["answer"]

                    # --- Optional: Display Source Documents ---
                    if "source_documents" in result:
                        full_response += "\n\n**Source Documents:**\n"
                        for doc in result["source_documents"]:
                            # Check if 'source' exists in metadata, default to a placeholder if not.
                            source = doc.metadata.get('source', 'Unknown Source')
                            full_response += f"- *{source}*\n"

                    message_placeholder.markdown(full_response)

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    full_response = "Sorry, I encountered an error."
            else:
                full_response = "The RetrievalQA chain could not be initialized. Please check your Qdrant and OpenAI settings."
                st.error(full_response)  # Show error on screen.

            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
