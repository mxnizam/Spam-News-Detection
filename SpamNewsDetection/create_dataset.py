import pandas as pd

# Sample dataset with real and fake news
data = {
    "title": [
        "Breaking: Earthquake in City",
        "Win a Free iPhone!!!",
        "Government announces new policy",
        "You won $1,000,000!",
        "NASA plans new Mars mission",
        "Click here to earn money fast"
    ],
    "text": [
        "A strong earthquake hit the city today.",
        "Click the link to claim your free iPhone.",
        "The government has introduced a new policy for economic growth.",
        "Congratulations! Click here to claim your prize.",
        "NASA is preparing for its next Mars exploration.",
        "Make $5000 a day with this simple trick!"
    ],
    "label": [0, 1, 0, 1, 0, 1]  # 0 = Real, 1 = Spam
}

df = pd.DataFrame(data)
df.to_csv("news_dataset.csv", index=False)

print("Dataset created successfully!")
