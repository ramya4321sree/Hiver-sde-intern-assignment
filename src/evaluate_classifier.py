import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

TRAIN_FILE = "data/apple_labeled.csv"
GOLDEN_FILE = "data/golden_set_final.xlsx"

# ---------------------------------------------------------
# Load training data
# ---------------------------------------------------------
train_df = pd.read_csv(TRAIN_FILE)

train_df["tweet_id"] = (
    pd.to_numeric(train_df["tweet_id"], errors="coerce")
    .astype("Int64")
    .astype(str)
)

train_df["clean_text"] = train_df["clean_text"].fillna("")
train_df["intent"] = train_df["intent"].fillna("other")

# ---------------------------------------------------------
# Load golden set
# ---------------------------------------------------------
golden_df = pd.read_excel(GOLDEN_FILE)

golden_df["tweet_id"] = (
    pd.to_numeric(golden_df["tweet_id"], errors="coerce")
    .astype("Int64")
    .astype(str)
)

golden_df["text"] = golden_df["text"].fillna("").astype(str)
golden_df["human_intent"] = (
    golden_df["human_intent"]
    .fillna("")
    .astype(str)
    .str.strip()
)

golden_df = golden_df[
    golden_df["human_intent"] != ""
].copy()

# ---------------------------------------------------------
# Remove golden examples from training data
# ---------------------------------------------------------
golden_ids = set(golden_df["tweet_id"])

before = len(train_df)

train_df = train_df[
    ~train_df["tweet_id"].isin(golden_ids)
].copy()

removed = before - len(train_df)

print("=" * 70)
print("LEAKAGE-FREE CLASSIFIER EVALUATION")
print("=" * 70)

print("Original training examples:", before)
print("Golden examples:", len(golden_df))
print("Golden examples removed from training:", removed)
print("Final training examples:", len(train_df))

# ---------------------------------------------------------
# Train TF-IDF + Linear SVM
# ---------------------------------------------------------
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95,
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(train_df["clean_text"])

model = LinearSVC(
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, train_df["intent"])

# ---------------------------------------------------------
# Evaluate on golden set
# ---------------------------------------------------------
X_golden = vectorizer.transform(golden_df["text"])

y_true = golden_df["human_intent"]
y_pred = model.predict(X_golden)

accuracy = accuracy_score(y_true, y_pred)

print()
print(f"Accuracy: {accuracy:.4f}")

print()
print("Classification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        digits=4,
        zero_division=0
    )
)

# ---------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------
labels = sorted(set(y_true) | set(y_pred))

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=labels
)

print("Labels:")
print(labels)

print()
print("Confusion Matrix:")
print(cm)

# ---------------------------------------------------------
# Save predictions
# ---------------------------------------------------------
golden_df["predicted_intent"] = y_pred
golden_df["correct"] = (
    golden_df["human_intent"]
    == golden_df["predicted_intent"]
)

OUTPUT = "data/golden_predictions.csv"

golden_df.to_csv(
    OUTPUT,
    index=False
)

print()
print("Saved predictions to:", OUTPUT)
print("=" * 70)