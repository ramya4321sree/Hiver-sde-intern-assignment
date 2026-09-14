import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

FILE = "data/apple_support_raw.csv"

print("Loading AppleSupport data...")

df = pd.read_csv(FILE)

# Keep only customer messages
customers = df[df["inbound"] == True].copy()

# Remove missing text
customers["text"] = customers["text"].fillna("")

print("Customer messages:", len(customers))
print()

# Remove the @AppleSupport mention before analysis
text = (
    customers["text"]
    .str.replace(r"@\w+", "", regex=True)
    .str.replace(r"https?://\S+", "", regex=True)
    .str.replace(r"\b\d{5,}\b", "", regex=True)
)

# Find common 1-word, 2-word and 3-word phrases
vectorizer = CountVectorizer(
    stop_words="english",
    ngram_range=(1, 3),
    min_df=50,
    max_features=100
)

X = vectorizer.fit_transform(text)

counts = X.sum(axis=0).A1
terms = vectorizer.get_feature_names_out()

results = sorted(
    zip(terms, counts),
    key=lambda x: x[1],
    reverse=True
)

print("Most common words/phrases:")
print("=" * 60)

for term, count in results[:100]:
    print(f"{term:40} {count}")