import streamlit as st
import spacy
import spacy.cli
import os

# Path where we store the model persistently
MODEL_DIR = "models/en_core_web_trf"

@st.cache_resource
def load_spacy_trf():
    """Load en_core_web_trf model, downloading if not already stored."""
    if not os.path.exists(MODEL_DIR):
        os.makedirs("models", exist_ok=True)
        with st.spinner("Downloading en_core_web_trf model (only once)..."):
            spacy.cli.download("en_core_web_trf", False)
        # Move the model to our persistent folder
        os.system(f"mv $(python -m spacy info en_core_web_trf | grep Location | awk '{{print $2}}') {MODEL_DIR}")

    # Load from persistent folder
    return spacy.load(MODEL_DIR)

# Load transformer-based model
nlp = load_spacy_trf()

# App title
st.title("Named Entity Recognition with spaCy (Transformer Model)")

# Text input
text_input = st.text_area("Enter text to analyze:", 
                          "Apple is looking at buying U.K. startup for $1 billion")

# Analyze button
if st.button("Analyze"):
    doc = nlp(text_input)
    ents = [(ent.text, ent.label_) for ent in doc.ents]
    
    if ents:
        st.subheader("Named Entities:")
        for text, label in ents:
            st.write(f"**{text}** → {label}")
    else:
        st.write("No named entities found.")
