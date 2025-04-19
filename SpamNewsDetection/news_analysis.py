import pandas as pd
import nltk  
from nltk.tokenize import word_tokenize

# Download necessary NLTK data (only needed once)
nltk.download("punkt")

# Load the dataset
true_news = pd.read_csv("True.csv")
fake_news = pd.read_csv("Fake.csv")

# Add labels (1 = real, 0 = fake)
true_news["label"] = 1
fake_news["label"] = 0

# Select only necessary columns
true_news = true_news[["title", "text", "label"]]
fake_news = fake_news[["title", "text", "label"]]

# Combine the datasets
news_data = pd.concat([true_news, fake_news], axis=0)

# Function to tokenize text
def tokenize_text(text):
    return word_tokenize(text)

# Apply tokenization
news_data["title"] = news_data["title"].astype(str).apply(tokenize_text)
news_data["text"] = news_data["text"].astype(str).apply(tokenize_text)

# ✅ Display the first few rows to verify
print("Dataset after tokenization:")
print(news_data.head())
