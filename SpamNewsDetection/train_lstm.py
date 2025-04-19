import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Load dataset
df = pd.read_csv("spam_news_data.csv")

# Preprocessing
texts = df['text'].astype(str).tolist()  # Convert text column to string
labels = np.array(df['label'].apply(lambda x: 1 if x == "FAKE" else 0))  # Convert labels to binary

# Tokenization
max_words = 5000  # Vocabulary size
max_length = 200  # Max words per text

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

# Split dataset (80% train, 20% test)
split = int(0.8 * len(padded_sequences))
X_train, X_test = padded_sequences[:split], padded_sequences[split:]
y_train, y_test = labels[:split], labels[split:]

# Build LSTM Model
model = Sequential([
    Embedding(input_dim=max_words, output_dim=128, input_length=max_length),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile Model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train Model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"LSTM Accuracy: {accuracy:.2f}")

# Save the trained model
model.save("lstm_spam_news_model.h5")
print("LSTM model saved as lstm_spam_news_model.h5")
