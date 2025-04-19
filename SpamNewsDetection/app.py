import streamlit as st
import pickle
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Load custom CSS
css_file_path = os.path.join(os.path.dirname(__file__), "style.css")
with open(css_file_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load the trained model
model_path = "spam_model.pkl"  # Replace with your trained model path
vectorizer_path = "vectorizer.pkl"  # Updated path

with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

with open(vectorizer_path, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Text preprocessing function
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text

# Streamlit UI
st.title("📰 Spam News Detection App")
st.write("Enter a news article below to check if it's **real** or **spam**.")

# User input
user_input = st.text_area("Enter news text here:")

if user_input:
    processed_text = preprocess_text(user_input)
    vectorized_text = vectorizer.transform([processed_text])
    
    # Debugging: Print vectorized shape and raw scores
    print("Processed Text:", processed_text)
    print("Vectorized Shape:", vectorized_text.shape)
    print("Raw Prediction Score:", model.predict_proba(vectorized_text))  # Show probability

    prediction = model.predict(vectorized_text)[0]
    result = "🟢 Real News" if prediction == 0 else "🔴 Spam News"
    
    st.subheader(f"Prediction: {result}")

else:
    st.warning("Please enter some text to analyze.")

# File upload option (optional)
st.subheader("📂 Upload a file for single detection")
uploaded_file = st.file_uploader("Upload a .txt file containing a single news article", type=["txt"])

if uploaded_file is not None:
    st.write("File uploaded successfully!")

    # Process file content
    file_data = uploaded_file.read().decode("utf-8").strip()  # Read and strip extra spaces/newlines

    if file_data:
        processed_text = preprocess_text(file_data)
        vectorized_text = vectorizer.transform([processed_text])
        prediction = model.predict(vectorized_text)[0]

        # Display result
        result = "🟢 Real News" if prediction == 0 else "🔴 Spam News"
        st.subheader(f"Prediction: {result}")
    else:
        st.warning("The uploaded file is empty. Please upload a valid file.")