import streamlit as st
import google.generativeai as genai
import os
import re # Import regex for potentially easier number extraction

# --- Configuration ---
st.set_page_config(page_title="Grounding Predictor", layout="wide")
st.title("🔮 Prompt Grounding Likelihood Predictor")
st.caption("Predict if a prompt is likely to require grounding and estimate the confidence of that prediction.")

# --- Access API Key using Streamlit Secrets ---
api_key = None
api_key_configured = False

# Check if the secret is defined
if "google_api_key" in st.secrets:
    api_key = st.secrets["google_api_key"]
else:
    st.sidebar.error("Google AI API key not found in Streamlit secrets!")
    st.sidebar.caption("Please add your key to `.streamlit/secrets.toml` locally, or in the app's secrets settings on Streamlit Community Cloud.")
    st.info("API Key is missing. Please configure it in Streamlit secrets to use this tool.", icon="🔑")
    # Stop execution if the key isn't present
    # st.stop() # Option 1: Halt completely
    # Option 2: Allow app to load but disable functionality (handled below by checking api_key_configured)


# --- API Configuration & Model Selection (only if key exists) ---
model = None
selected_model_name = None

if api_key:
    try:
        genai.configure(api_key=api_key)
        api_key_configured = True # Mark key as successfully configured

        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        preferred_models = [m for m in available_models if 'gemini' in m and 'vision' not in m and 'embed' not in m]
        if not preferred_models:
            preferred_models = available_models

        default_model = 'models/gemini-1.5-flash-latest'
        if default_model not in preferred_models and preferred_models:
            default_model = preferred_models[0]
        elif not preferred_models:
             default_model = None

        if default_model:
            selected_model_name = st.sidebar.selectbox(
                "Select Google AI Model (for Prediction):",
                options=preferred_models,
                index=preferred_models.index(default_model) if default_model in preferred_models else 0
            )
            model = genai.GenerativeModel(selected_model_name)
            st.sidebar.success(f"API Key loaded. Using model: `{selected_model_name}`")
        else:
            st.sidebar.error("No suitable text generation models found for this API key.")
            api_key_configured = False # Mark as not configured if model selection fails

    except Exception as e:
        st.sidebar.error(f"Error configuring API key or listing models: {e}")
        api_key_configured = False # Mark as not configured on error
else:
     st.sidebar.warning("API Key not found in secrets.")


# --- User Input ---
st.subheader("1. Enter the Prompt, Question, or Query to Analyze")
user_prompt = st.text_area(
    "Paste the input you want to evaluate:",
    height=200,
    key="user_prompt_input",
    placeholder="Examples:\n- What's the latest news about the Mars rover?\n- Explain the concept of quantum entanglement.\n- Who won the Best Picture Oscar in 2023?\n- Write a summary of the provided text: [text would go here]"
)

# --- Prediction Logic ---
# Button is disabled if API key wasn't configured OR model isn't selected OR prompt is empty
predict_button = st.button(
    "Predict Grounding Likelihood",
    disabled=not api_key_configured or not model or not user_prompt,
    key="predict_button"
)

if predict_button and api_key_configured and model: # Double-check conditions
    # --- Construct the Meta-Prompt (same as before) ---
    prediction_prompt = f"""
    **Task:** Analyze the following "User Prompt". Predict whether an AI attempting to answer this prompt accurately and comprehensively would likely require **grounding**. Grounding means accessing external, specific, or up-to-date information beyond its general knowledge base (e.g., performing a web search, checking recent data feeds).

    **Consider these factors:**
    *   **Recency:** Does the prompt ask about very recent events, news, or data?
    *   **Specificity:** Does it ask about highly specific entities, obscure facts, or niche topics?
    *   **Fact-Checking/Verification:** Does the prompt imply a need to verify a claim against a current external source?
    *   **Dynamic Information:** Does the prompt relate to information that changes frequently?
    *   **General Knowledge vs. Specific Data:** Can it be answered using general knowledge or does it need specific data?

    **User Prompt to Analyze:**
    ---
    {user_prompt}
    ---

    **Prediction:**
    Based on the analysis, provide the likelihood, your estimated confidence in that likelihood, and a brief justification.
    1.  **Likelihood:** [Likely Requires Grounding / Likely Self-Contained / Borderline or Ambiguous] (Choose one)
    2.  **Confidence:** [Provide an estimated confidence percentage for your Likelihood prediction, e.g., 85%]
    3.  **Reasoning:** [Explain your prediction based on the factors above.]

    **Output Format:**
    Likelihood: [Your Choice Here]
    Confidence: [Your Estimated Percentage Here]%
    Reasoning: [Your Explanation Here]
    """

    try:
        with st.spinner(f"Asking `{selected_model_name}` to predict grounding need and confidence..."):
            generation_config_predict = genai.types.GenerationConfig(
                temperature=0.2
            )
            prediction_response = model.generate_content(
                prediction_prompt,
                generation_config=generation_config_predict,
            )
            prediction_text = prediction_response.text

        st.subheader("2. Grounding Likelihood Prediction")

        # --- Parsing Block (same as before) ---
        likelihood = "Could not parse"
        confidence_score = None
        reasoning = prediction_text

        likelihood_match = re.search(r"^\s*Likelihood:\s*(.*)", prediction_text, re.MULTILINE | re.IGNORECASE)
        confidence_match = re.search(r"^\s*Confidence:\s*(\d{1,3})\s*%", prediction_text, re.MULTILINE | re.IGNORECASE)
        reasoning_match = re.search(r"^\s*Reasoning:\s*(.*)", prediction_text, re.MULTILINE | re.IGNORECASE | re.DOTALL)

        if likelihood_match:
            potential_likelihood = likelihood_match.group(1).strip().lower()
            if "likely requires grounding" in potential_likelihood or "requires grounding" in potential_likelihood:
                likelihood = "Likely Requires Grounding"
            elif "likely self-contained" in potential_likelihood or "self-contained" in potential_likelihood:
                likelihood = "Likely Self-Contained"
            elif "borderline" in potential_likelihood or "ambiguous" in potential_likelihood:
                likelihood = "Borderline / Ambiguous"

        if confidence_match:
            try:
                confidence_score = int(confidence_match.group(1))
                if not (0 <= confidence_score <= 100): confidence_score = None
            except (ValueError, IndexError): confidence_score = None

        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        else:
             temp_reasoning = prediction_text
             if likelihood_match: temp_reasoning = temp_reasoning.replace(likelihood_match.group(0), "", 1)
             if confidence_match: temp_reasoning = temp_reasoning.replace(confidence_match.group(0), "", 1)
             reasoning = temp_reasoning.strip()

        # --- Display parsed results (same as before) ---
        confidence_text = f"(Confidence: {confidence_score}%)" if confidence_score is not None else "(Confidence: Not Provided)"

        if likelihood == "Likely Requires Grounding":
            st.warning(f"**Prediction: {likelihood}**  🌐 {confidence_text}")
        elif likelihood == "Likely Self-Contained":
            st.success(f"**Prediction: {likelihood}** 🧠 {confidence_text}")
        elif likelihood == "Borderline / Ambiguous":
            st.info(f"**Prediction: {likelihood}** 🤔 {confidence_text}")
        else: # Could not parse Likelihood
            st.error(f"**Prediction: {likelihood}** ❓")

        # Display reasoning if likelihood was parsed, otherwise show raw output.
        if likelihood != "Could not parse":
            st.markdown("**Reasoning from AI:**")
            st.markdown(reasoning)
        else:
             st.markdown("**AI's Raw Prediction Output:**")
             st.markdown(prediction_text)

        if confidence_score is not None or likelihood != "Could not parse":
             st.caption("ℹ️ The confidence score is the AI's own estimation and may not be statistically precise.")

        # --- Expanders for Debugging (same as before) ---
        with st.expander("Show Raw AI Prediction Response"):
            st.text(prediction_text)
        with st.expander("Show Prompt Sent to Prediction AI"):
            st.text(prediction_prompt)

    except Exception as e:
        st.error(f"An error occurred during prediction API call: {e}")
        try:
            if 'prediction_response' in locals() and hasattr(prediction_response, 'candidates'):
                 st.error(f"Prediction Candidate info: {prediction_response.candidates}")
            elif 'prediction_response' in locals() and hasattr(prediction_response, 'prompt_feedback'):
                 st.error(f"Prediction Prompt Feedback: {prediction_response.prompt_feedback}")
            else:
                 st.error("Could not retrieve detailed error information (e.g., safety blocking).")
        except Exception as e_detail:
             st.error(f"Error retrieving details: {e_detail}")


elif predict_button:
    # Add checks for why the button might be pressed but logic doesn't run
    if not api_key_configured:
         st.error("Prediction failed: API Key is not configured correctly via secrets.")
    elif not model:
         st.error("Prediction failed: AI Model could not be initialized.")
    elif not user_prompt:
        st.error("Prediction failed: Prompt to analyze is empty.")

# Add a general message if the API key wasn't configured at the start
if not api_key_configured and not api_key: # Show only if key wasn't even found
    st.warning("Please configure your Google AI API Key in Streamlit secrets to enable prediction.")
