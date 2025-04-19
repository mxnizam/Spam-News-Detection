import pandas as pd

# Load datasets
fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

# Add labels
fake_df["label"] = "FAKE"
true_df["label"] = "TRUE"

# Combine both datasets
combined_df = pd.concat([fake_df, true_df])

# Shuffle the dataset
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the merged dataset
combined_df.to_csv("spam_news_data.csv", index=False)

print("Merged dataset saved as spam_news_data.csv!")
