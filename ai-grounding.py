import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
import time # To add slight delay for better UX between steps if needed

# --- Configuration ---
st.set_page_config(page_title="AI Groundedness Bot", layout="wide")
st.title("🤖 AI Response Generation & Groundedness Check")
st.caption("Generate an AI response based on context and then evaluate its groundedness using Google AI.")

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

        # Default to flash if available, otherwise the first preferred model
        default_model = 'models/gemini-1.5-flash-latest'
        if default_model not in preferred_models and preferred_models:
            default_model = preferred_models[0]
        elif not preferred_models:
             default_model = None # No suitable models

        if default_model:
            selected_model_name = st.sidebar.selectbox(
                "Select Google AI Model (for Generation & Evaluation):",
                options=preferred_models,
                index=preferred_models.index(default_model)
            )
            model = genai.GenerativeModel(selected_model_name)
            st.sidebar.success(f"API Key configured. Using model: `{selected_model_name}`")
        else:
            st.sidebar.error("No suitable text generation models found for this API key.")
            api_key = None # Prevent further steps

    except Exception as e:
        st.sidebar.error(f"Error configuring API key or listing models: {e}")
        api_key = None
else:
    st.sidebar.warning("Please enter your Google AI API Key to proceed.")
    st.info("Enter your Google AI API Key in the sidebar to begin.")

# --- User Inputs ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Context / Source Material")
    context = st.text_area(
        "Paste the text the AI response should be based on:",
        height=300,
        key="context_input",
        placeholder="Example: The Atacama Desert in Chile is the driest nonpolar desert in the world. Some weather stations there have never received rain. It covers a 1,600 km strip of land west of the Andes mountains."
    )

with col2:
    st.subheader("2. Prompt for AI Response")
    response_generation_prompt = st.text_area(
        "Ask the AI to generate a response based *only* on the context:",
        height=300,
        key="prompt_input",
        placeholder="Example: Summarize the key facts about the Atacama Desert mentioned in the context in one sentence."
        # Example leading to ungrounded: "Where is the Atacama Desert located and what is its average rainfall?" (Avg rainfall not in context)
    )

# --- Generation & Evaluation ---
# Use session state to store results across potential reruns if needed
if 'generated_response' not in st.session_state:
    st.session_state.generated_response = ""
if 'evaluation_result' not in st.session_state:
    st.session_state.evaluation_result = None
if 'evaluation_explanation' not in st.session_state:
    st.session_state.evaluation_explanation = ""

generate_button = st.button(
    "Generate Response & Evaluate Groundedness",
    disabled=not api_key or not context or not response_generation_prompt,
    key="generate_eval_button"
)

if generate_button and model:
    st.session_state.generated_response = ""
    st.session_state.evaluation_result = None
    st.session_state.evaluation_explanation = ""

    # --- Step 1: Generate Response ---
    try:
        generation_instruction = f"""
        Based **only** on the following context, please respond to the request. Do not use any external knowledge or information not explicitly stated in the context.

        Context:
        ---
        {context}
        ---

        Request:
        ---
        {response_generation_prompt}
        ---

        Response:
        """
        with st.spinner(f"Generating response using `{selected_model_name}`..."):
            # Configure generation - maybe slightly creative but constrained
            generation_config_gen = genai.types.GenerationConfig(
                temperature=0.5 # Allow some flexibility in phrasing
            )
            # Safety settings might be needed if prompts get blocked
            # safety_settings_gen = [...]

            response_gen = model.generate_content(
                generation_instruction,
                generation_config=generation_config_gen,
                # safety_settings=safety_settings_gen # Add if needed
            )
            st.session_state.generated_response = response_gen.text

    except Exception as e:
        st.error(f"An error occurred during response generation: {e}")
        # Attempt to access candidate information if available for debugging
        try: st.error(f"Generation Candidate info: {response_gen.candidates}")
        except: pass
        st.stop() # Stop execution if generation fails

    # --- Display Generated Response ---
    st.subheader("3. Generated Response")
    if st.session_state.generated_response:
        st.markdown(st.session_state.generated_response)
        st.divider() # Visual separator

        # --- Step 2: Evaluate Groundedness ---
        try:
            # Groundedness evaluation prompt (same as before)
            groundedness_prompt = f"""
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
            {st.session_state.generated_response}
            ---

            **Evaluation:**
            Based **strictly** on the provided Context:
            1.  Is the Response Grounded or Ungrounded? Answer with a single word: "Grounded" or "Ungrounded".
            2.  Provide a brief explanation for your reasoning, highlighting specific parts of the response and context if necessary.

            **Output Format:**
            Likelihood: [Grounded/Ungrounded]
            Explanation: [Your reasoning here]
            """

            with st.spinner(f"Evaluating groundedness using `{selected_model_name}`..."):
                 # Configure evaluation - more deterministic
                generation_config_eval = genai.types.GenerationConfig(
                    temperature=0.1 # Low temp for factual assessment
                )
                # Potentially stricter safety settings for evaluation output if needed
                # safety_settings_eval = [...]

                evaluation_response = model.generate_content(
                    groundedness_prompt,
                    generation_config=generation_config_eval,
                     # safety_settings=safety_settings_eval # Add if needed
                )
                evaluation_text = evaluation_response.text

            # --- Parse and Display Evaluation Result ---
            likelihood = "Could not parse"
            explanation = evaluation_text # Default

            lines = evaluation_text.strip().split('\n', 1)
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
                    else:
                        explanation = second_part # Use as explanation if format differs

            st.session_state.evaluation_result = likelihood
            st.session_state.evaluation_explanation = explanation

        except Exception as e:
            st.error(f"An error occurred during groundedness evaluation: {e}")
             # Attempt to access candidate information if available for debugging
            try: st.error(f"Evaluation Candidate info: {evaluation_response.candidates}")
            except: pass
            st.session_state.evaluation_result = "Error"
            st.session_state.evaluation_explanation = f"Evaluation failed: {e}"

    else:
        st.warning("Response generation resulted in empty output. Cannot evaluate.")
        st.session_state.evaluation_result = "Skipped"
        st.session_state.evaluation_explanation = "Generation produced no text."


# --- Display Final Evaluation Results (even after rerun if button wasn't pressed again) ---
if st.session_state.evaluation_result:
    st.subheader("4. Groundedness Evaluation")
    likelihood = st.session_state.evaluation_result
    explanation = st.session_state.evaluation_explanation

    if likelihood == "Grounded":
        st.success(f"**Likelihood: Grounded** 👍")
        st.markdown("**Explanation from AI:**")
        st.markdown(explanation)
    elif likelihood == "Ungrounded":
        st.warning(f"**Likelihood: Ungrounded** 👎")
        st.markdown("**Explanation from AI:**")
        st.markdown(explanation)
    elif likelihood == "Error":
         st.error(f"**Likelihood: Evaluation Error** ❗️")
         st.markdown("**Details:**")
         st.markdown(explanation)
    elif likelihood == "Skipped":
         st.info(f"**Likelihood: Evaluation Skipped**")
         st.markdown("**Reason:**")
         st.markdown(explanation)
    else: # Includes "Could not parse"
        st.error(f"**Likelihood: Could Not Parse** ❓")
        st.markdown("**AI's Raw Evaluation Output:**")
        st.markdown(explanation) # Show the raw output if parsing failed

    # Optional: Expander for raw evaluation response
    # with st.expander("Show Raw AI Evaluation Response"):
    #    st.text(explanation if likelihood not in ["Error", "Skipped"] else "N/A")


elif generate_button: # If button was pressed but something failed silently before results could be stored
    if not api_key:
        st.error("Action failed: API Key is missing.")
    elif not context:
        st.error("Action failed: Context is empty.")
    elif not response_generation_prompt:
        st.error("Action failed: Prompt for AI Response is empty.")
