import os
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
import streamlit as st
from transformers import pipeline

# --- THE UPDATED LABEL MAPPING SECTION ---
# Define your EXACT Trained Mapping from the model configuration
# Crucial: This must match your final Colab training run exactly.
id2label = {
    "LABEL_0": "Normal",
    "LABEL_1": "Depression",
    "LABEL_2": "Suicidal", # This is the high-risk label
    "LABEL_3": "Anxiety",
    "LABEL_4": "Stress"
}

# --- END OF MAPPING SECTION ---

# 1. Page Configuration (Makes it look professional)
# UPDATE: Title changed to reflect bilingual support.
st.set_page_config(page_title="Hinglish/English Mental Health AI", layout="centered")

# 2. Header Section
# UPDATE: Subtitle changed to explicitly state English support.
st.title("🧠 Mental Health Detection AI")
st.markdown("""
This AI model is built using **MuRIL** and fine-tuned to understand 
**English and Hinglish** (Hindi + English) social media text.
""")

# 3. Model Loading Logic
@st.cache_resource # This prevents the model from reloading every time you click a button
def load_model():
    # Calling your specific model from Hugging Face
    model_id = "jyotisangam/muril-mental-health-xai"
    # We load all labels ("top_k=None") to create the progress bars.
    return pipeline("text-classification", model=model_id, top_k=None)

with st.spinner("Loading MuRIL Model... Please wait."):
    classifier = load_model()

# 4. User Interaction Sidebar
st.sidebar.header("About the Model")
# UPDATE: Precision and threshold values are illustrative and should be confirmed with your results.
st.sidebar.info("""
- **Precision (Suicidal):** 94%
- **Threshold:** 0.75
- **Languages:** English, Hinglish
""")

# 5. Main Input Area
# UPDATE: Placeholder changed to show an English example.
user_text = st.text_area(
    "Analyze Text:", 
    placeholder="Type your Hinglish or English sentence here (e.g., 'Mujhe bohot stress ho raha hai' or 'I feel so alone...')"
)

if st.button("Run AI Analysis"):
    if user_text.strip() != "":
        # Get predictions
        raw_results = classifier(user_text)[0]
        
        st.subheader("Results:")
        
        # Display results in a clean way with progress bars
        for result in raw_results:
            raw_label = result['label'] # This is 'LABEL_0', 'LABEL_1', etc.
            score = result['score']     # This is the confidence score (0 to 1).
            
            # --- THE MAGIC HAPPENS HERE: Mapping the label ---
            # Use the id2label dictionary to get the human-readable name.
            # If the label is not found, we use the raw label as a backup.
            mapped_label = id2label.get(raw_label, raw_label)
            
            # Show progress bars for each category with the NEW label.
            st.write(f"**{mapped_label}**")
            st.progress(score)
            
            # Show the percentage confidence score.
            st.caption(f"Confidence: {score:.2%}")

            # Highlight Safety Logic (The 0.75 Threshold)
            # IMPORTANT: We check for the mapped label "Suicidal"
            if mapped_label == "Suicidal" and score > 0.75:
                st.error("🚨 HIGH ALERT: High Suicidal Ideation detected.")
                st.info("Helpline Resource: [Vandrevala Foundation](https://www.vandrevalafoundation.com/) or call 9999 666 555.")

    else:
        st.warning("Please enter some text to analyze.")