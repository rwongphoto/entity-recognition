import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

# --- Configuration ---
st.set_page_config(page_title="Grounding Predictor", layout="wide")
st.title("🔮 Prompt Grounding Likelihood Predictor")
st.caption("Predict if a prompt, question, or query is likely to require grounding (e.g., web search, recent data) for an accurate AI response.")

# --- Load API Key ---
load_dotenv()
api_key_env = os.getenv("GOOGLE_API_KEY")

api_key_input = st.sidebar.text_input(
    "Enter your Google AI API Key:",
    type="password",
    value=api_key_env if api_key_env else "",
    help="Get your key from https://aistudio.google.com/app/apikey"
)

# --- API Configuration & Model Selection ---
api_key = api_key_input if api_key_input else api_key_env
model = None
selected_model_name = None

if api_key:
    try:
        genai.configure(api_key=api_key)
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        preferred_models = [m for m in available_models if 'gemini' in m and 'vision' not in m and 'embed' not in m]
        if not preferred_models:
            preferred_models = available_models

        default_model = 'models/gemini-1.5-flash-latest' # Flash is often good enough for analysis
        if default_model not in preferred_models and preferred_models:
            default_model = preferred_models[0]
        elif not preferred_models:
             default_model = None

        if default_model:
            selected_model_name = st.sidebar.selectbox(
                "Select Google AI Model (for Prediction):",
                options=preferred_models,
                index=preferred_models.index(default_model)
            )
            model = genai.GenerativeModel(selected_model_name)
            st.sidebar.success(f"API Key configured. Using model: `{selected_model_name}`")
        else:
            st.sidebar.error("No suitable text generation models found for this API key.")
            api_key = None

    except Exception as e:
        st.sidebar.error(f"Error configuring API key or listing models: {e}")
        api_key = None
else:
    st.sidebar.warning("Please enter your Google AI API Key to proceed.")
    st.info("Enter your Google AI API Key in the sidebar to begin.")

# --- User Input ---
st.subheader("1. Enter the Prompt, Question, or Query to Analyze")
user_prompt = st.text_area(
    "Paste the input you want to evaluate:",
    height=200,
    key="user_prompt_input",
    placeholder="Examples:\n- What's the latest news about the Mars rover?\n- Explain the concept of quantum entanglement.\n- Who won the Best Picture Oscar in 2023?\n- Write a summary of the provided text: [text would go here]"
)

# --- Prediction Logic ---
predict_button = st.button(
    "Predict Grounding Likelihood",
    disabled=not api_key or not user_prompt,
    key="predict_button"
)

if predict_button and model:
    # --- Construct the Meta-Prompt for the Prediction Model ---
    prediction_prompt = f"""
    **Task:** Analyze the following "User Prompt". Predict whether an AI attempting to answer this prompt accurately and comprehensively would likely require **grounding**. Grounding means accessing external, specific, or up-to-date information beyond its general knowledge base (e.g., performing a web search, checking recent data feeds).

    **Consider these factors:**
    *   **Recency:** Does the prompt ask about very recent events, news, or data (e.g., today's weather, last night's game scores, current stock prices)?
    *   **Specificity:** Does it ask about highly specific entities, obscure facts, niche topics, or detailed product information that might not be in general training data?
    *   **Fact-Checking/Verification:** Does the prompt imply a need to verify a claim or check against a current external source?
    *   **Dynamic Information:** Does the prompt relate to information that changes frequently (e.g., rankings, availability, ongoing events)?
    *   **General Knowledge vs. Specific Data:** Can the prompt be answered sufficiently using well-established, common knowledge (like historical facts, scientific concepts), or does it demand specific, potentially volatile data points?

    **User Prompt to Analyze:**
    ---
    {user_prompt}
    ---

    **Prediction:**
    Based on the analysis, choose one primary likelihood and provide a brief justification:
    1.  **Likelihood:** [Likely Requires Grounding / Likely Self-Contained / Borderline or Ambiguous] (Choose one)
    2.  **Reasoning:** [Explain your prediction based on the factors above. Why would grounding be needed or not needed?]

    **Output Format:**
    Likelihood: [Your Choice Here]
    Reasoning: [Your Explanation Here]
    """

    try:
        with st.spinner(f"Asking `{selected_model_name}` to predict grounding need..."):
            # Use lower temperature for more analytical response
            generation_config_predict = genai.types.GenerationConfig(
                temperature=0.2
            )
            # Safety settings - usually less critical for analysis prompts, but adjust if needed
            # safety_settings_predict = [...]

            prediction_response = model.generate_content(
                prediction_prompt,
                generation_config=generation_config_predict,
                # safety_settings=safety_settings_predict # Add if needed
            )
            prediction_text = prediction_response.text

        st.subheader("2. Grounding Likelihood Prediction")

        # --- Parse the prediction result ---
        likelihood = "Could not parse"
        reasoning = prediction_text # Default

        lines = prediction_text.strip().split('\n', 1)
        if len(lines) > 0:
            first_line = lines[0].lower() # Case-insensitive check
            if "likelihood:" in first_line:
                potential_likelihood = first_line.split("likelihood:", 1)[1].strip()
                # Check for keywords robustly
                if "likely requires grounding" in potential_likelihood or "requires grounding" in potential_likelihood:
                    likelihood = "Likely Requires Grounding"
                elif "likely self-contained" in potential_likelihood or "self-contained" in potential_likelihood:
                    likelihood = "Likely Self-Contained"
                elif "borderline" in potential_likelihood or "ambiguous" in potential_likelihood:
                    likelihood = "Borderline / Ambiguous"
                # Keep "Could not parse" if none match clearly

            if len(lines) > 1:
                 second_part = lines[1].strip()
                 if second_part.lower().startswith("reasoning:"):
                     reasoning = second_part.split("reasoning:", 1)[1].strip()
                 else:
                     reasoning = second_part # Use as reasoning if format differs


        # --- Display parsed results ---
        if likelihood == "Likely Requires Grounding":
            st.warning(f"**Prediction: {likelihood}**  वेब") # Using a web emoji as indicator
            st.markdown("**Reasoning from AI:**")
            st.markdown(reasoning)
        elif likelihood == "Likely Self-Contained":
            st.success(f"**Prediction: {likelihood}** 🧠") # Using a brain emoji
            st.markdown("**Reasoning from AI:**")
            st.markdown(reasoning)
        elif likelihood == "Borderline / Ambiguous":
            st.info(f"**Prediction: {likelihood}** 🤔") # Using a thinking emoji
            st.markdown("**Reasoning from AI:**")
            st.markdown(reasoning)
        else: # Could not parse
            st.error(f"**Prediction: {likelihood}** ❓")
            st.markdown("**AI's Raw Prediction Output:**")
            st.markdown(prediction_text) # Show the full raw text if parsing failed


        # --- Expanders for Debugging ---
        with st.expander("Show Raw AI Prediction Response"):
            st.text(prediction_text)
        with st.expander("Show Prompt Sent to Prediction AI"):
            st.text(prediction_prompt)


    except Exception as e:
        st.error(f"An error occurred during prediction API call: {e}")
        # Attempt to access candidate information if available for debugging safety blocks etc.
        try:
            st.error(f"Prediction Candidate info: {prediction_response.candidates}")
        except:
            pass


elif predict_button: # If button was pressed but failed basic checks
    if not api_key:
        st.error("Prediction failed: API Key is missing.")
    elif not user_prompt:
        st.error("Prediction failed: Prompt to analyze is empty.")
