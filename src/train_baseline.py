import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------
# 1. Load labeled data
# -----------------------------

FILE = "data/apple_labeled.csv"

df = pd.read_csv(FILE)

# Remove missing values
df["clean_text"] = df["clean_text"].fillna("")
df["intent"] = df["intent"].fillna("other")

print("Total examples:", len(df))


# -----------------------------
# 2. Train / test split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"],
    df["intent"],
    test_size=0.20,
    random_state=42,
    stratify=df["intent"]
)

print("Training examples:", len(X_train))
print("Test examples:", len(X_test))


# -----------------------------
# 3. TF-IDF features
# -----------------------------

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF training shape:", X_train_tfidf.shape)
print("TF-IDF test shape:", X_test_tfidf.shape)


# -----------------------------
# 4. Logistic Regression
# -----------------------------

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_tfidf, y_train)


# -----------------------------
# 5. Predictions
# -----------------------------

y_pred = model.predict(X_test_tfidf)


# -----------------------------
# 6. Evaluation
# -----------------------------

accuracy = accuracy_score(y_test, y_pred)

print()
print("=" * 60)
print("BASELINE 1: TF-IDF + LOGISTIC REGRESSION")
print("=" * 60)

print(f"Accuracy: {accuracy:.4f}")

print()
print("Classification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        digits=4,
        zero_division=0
    )
)