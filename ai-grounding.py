import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

# --- Configuration ---
st.set_page_config(page_title="AI Groundedness Checker", layout="wide")
st.title("🔎 AI Response Groundedness Checker")
st.caption("Evaluate if an AI response is grounded in the provided context using Google AI.")

# --- Load API Key ---
# Try loading from .env file first (for development/local use)
load_dotenv()
api_key_env = os.getenv("GOOGLE_API_KEY")

# Get API key from user input if not found in environment variables
api_key_input = st.sidebar.text_input(
    "Enter your Google AI API Key:",
    type="password",
    value=api_key_env if api_key_env else "", # Pre-fill if found in .env
    help="Get your key from https://aistudio.google.com/app/apikey"
)

# --- API Configuration & Model Selection ---
api_key = api_key_input if api_key_input else api_key_env
model = None
if api_key:
    try:
        genai.configure(api_key=api_key)
        # Allow model selection
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        # Filter for common chat/text models if possible, or just show all compatible
        # Simple filtering for common models:
        preferred_models = [m for m in available_models if 'gemini' in m and 'vision' not in m and 'embed' not in m]
        if not preferred_models:
            preferred_models = available_models # Fallback if no 'gemini' text models found

        selected_model_name = st.sidebar.selectbox(
            "Select Google AI Model:",
            options=preferred_models,
            index=preferred_models.index('models/gemini-1.5-flash-latest') if 'models/gemini-1.5-flash-latest' in preferred_models else 0, # Default preference
            help="Choose the model for evaluation. Newer models might be better at reasoning."
        )
        model = genai.GenerativeModel(selected_model_name)
        st.sidebar.success(f"API Key configured. Using model: `{selected_model_name}`")
    except Exception as e:
        st.sidebar.error(f"Error configuring API key or listing models: {e}")
        api_key = None # Reset api_key to prevent further attempts if configuration failed
else:
    st.sidebar.warning("Please enter your Google AI API Key to proceed.")
    st.info("Enter your Google AI API Key in the sidebar to begin.")

# --- User Inputs ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Context / Source Material")
    context = st.text_area(
        "Paste the text the AI response should be based on:",
        height=300,
        placeholder="Example: The Eiffel Tower is located in Paris, France. It was completed in 1889."
    )

with col2:
    st.subheader("AI Response to Evaluate")
    response = st.text_area(
        "Paste the AI-generated response here:",
        height=300,
        placeholder="Example: The Eiffel Tower is in France and was built in the late 19th century."
        # Example Ungrounded: "The Eiffel Tower, located in Berlin, was finished in 1901."
    )

# --- Evaluation Logic ---
evaluate_button = st.button("Evaluate Groundedness", disabled=not api_key or not context or not response)

if evaluate_button and model:
    # Construct the prompt for the LLM
    prompt = f"""
    **Task:** Evaluate if the provided "Response" is factually supported **only** by the information given in the "Context". Do not use any external knowledge.

    **Definitions:**
    *   **Grounded:** All factual claims made in the Response can be directly and explicitly verified within the provided Context. The response does not introduce external information or contradict the context.
    *   **Ungrounded:** The Response contains factual information not present in the Context, contradicts the Context, or makes claims that cannot be verified *solely* by the Context. Minor phrasing differences are acceptable if the core facts align.

    **Context:**
    ---
    {context}
    ---

    **Response:**
    ---
    {response}
    ---

    **Evaluation:**
    Based **strictly** on the provided Context:
    1.  Is the Response Grounded or Ungrounded? Answer with a single word: "Grounded" or "Ungrounded".
    2.  Provide a brief explanation for your reasoning, highlighting specific parts of the response and context if necessary.

    **Output Format:**
    Likelihood: [Grounded/Ungrounded]
    Explanation: [Your reasoning here]
    """

    try:
        with st.spinner(f"Asking `{selected_model_name}` to evaluate groundedness..."):
            # Set safety settings to be less restrictive for this kind of analysis if needed
            # Be cautious with this in production environments
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                },
            ]
            # Note: Sometimes the API might throw errors if BLOCK_NONE isn't supported directly
            # If you get errors, try removing safety_settings or using BLOCK_MEDIUM_AND_ABOVE etc.

            generation_config = genai.types.GenerationConfig(
                # temperature=0.1 # Lower temperature for more factual/deterministic assessment
            )

            ai_response = model.generate_content(
                prompt,
                generation_config=generation_config,
                # safety_settings=safety_settings # Uncomment carefully if needed
                )

        st.subheader("Evaluation Result")
        evaluation_text = ai_response.text

        # --- Parse the result ---
        likelihood = "Could not parse"
        explanation = evaluation_text # Default to full text if parsing fails

        lines = evaluation_text.strip().split('\n', 1) # Split into max 2 parts at the first newline
        if len(lines) > 0:
            first_line = lines[0].lower()
            if "likelihood:" in first_line:
                potential_likelihood = first_line.split("likelihood:", 1)[1].strip()
                if "grounded" in potential_likelihood:
                    likelihood = "Grounded"
                elif "ungrounded" in potential_likelihood:
                    likelihood = "Ungrounded"

            if len(lines) > 1:
                 second_part = lines[1].strip()
                 if second_part.lower().startswith("explanation:"):
                     explanation = second_part.split("explanation:", 1)[1].strip()
                 else: # If "Explanation:" prefix is missing but there's a second line
                    explanation = second_part


        # Display parsed results
        if likelihood == "Grounded":
            st.success(f"**Likelihood: {likelihood}**")
        elif likelihood == "Ungrounded":
            st.warning(f"**Likelihood: {likelihood}**")
        else:
            st.error(f"**Likelihood: {likelihood}** (Model response might not follow the expected format)")

        st.markdown("**Explanation from AI:**")
        st.markdown(explanation) # Use markdown for potentially better formatting

        with st.expander("Show Raw AI Response"):
            st.text(evaluation_text)
        with st.expander("Show Prompt Sent to AI"):
            st.text(prompt)


    except Exception as e:
        st.error(f"An error occurred during API call: {e}")
        # Attempt to access candidate information if available for debugging safety blocks etc.
        try:
            st.error(f"Candidate info: {ai_response.candidates}")
        except:
            pass


elif evaluate_button:
    if not api_key:
        st.error("Evaluation failed: API Key is missing.")
    elif not context:
        st.error("Evaluation failed: Context is empty.")
    elif not response:
        st.error("Evaluation failed: Response is empty.")
