import pickle

# Load vectorizer
with open("vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

# Test some real and spam news samples
sample_texts = [
    "The government announces a new policy to boost the economy.",
    "Congratulations! You've won a free iPhone! Click the link now.",
]

vectorized_samples = vectorizer.transform(sample_texts)

print("Vectorized Shape:", vectorized_samples.shape)
