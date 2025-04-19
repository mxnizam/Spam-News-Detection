import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ✅ **Improved Training Dataset (Balanced with Real & Spam News)**
X_train = [
    "Government announces new policies to improve healthcare services.",  # Real
    "Scientists successfully test a new cancer treatment.",  # Real
    "Stock markets rise as economic recovery strengthens.",  # Real
    "NASA's latest space mission reaches Mars successfully.",  # Real
    "UN calls for global efforts to combat climate change.",  # Real
    "Win a free iPhone! Click the link now.",  # Spam
    "Congratulations! You've won a lottery worth $1,000,000.",  # Spam
    "Urgent: Your bank account has been locked. Update your details now.",  # Spam
    "Breaking news: Major earthquake hits California.",  # Real
    "Breaking: Scientists discover a new vaccine for malaria.",  # Real
    "Claim your free prize today! Limited time offer!",  # Spam
    "Earn $5000 per week from home. Sign up now!",  # Spam
]

y_train = [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1]  # 0 = Real, 1 = Spam

# ✅ **Convert Text into Numerical Features**
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)

# ✅ **Train a Simple Logistic Regression Model**
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# ✅ **Save the Trained Model**
with open("spam_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

# ✅ **Save the Vectorizer**
with open("vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)

print("✅ Retrained model and vectorizer saved successfully!")
