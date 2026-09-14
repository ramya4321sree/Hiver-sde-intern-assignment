import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


REPLY_FILE = "data/apple_reply_pairs_eval.csv"
GOLDEN_FILE = "data/golden_review_claude.xlsx"
LABEL_FILE = "data/apple_labeled.csv"
OUTPUT_FILE = "data/retrieval_predictions.csv"


print("=" * 70)
print("LEAKAGE-FREE HISTORICAL REPLY RETRIEVAL EVALUATION")
print("=" * 70)

# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------
reply_df = pd.read_csv(REPLY_FILE)
golden_df = pd.read_excel(GOLDEN_FILE)
label_df = pd.read_csv(LABEL_FILE)

reply_df["customer_tweet_id"] = reply_df["customer_tweet_id"].astype(str)
golden_df["tweet_id"] = golden_df["tweet_id"].astype(str)
label_df["tweet_id"] = label_df["tweet_id"].astype(str)

# ---------------------------------------------------------
# 2. Add intent labels to retrieval corpus
# ---------------------------------------------------------
label_map = (
    label_df[["tweet_id", "intent"]]
    .drop_duplicates("tweet_id")
    .set_index("tweet_id")["intent"]
)

reply_df["intent"] = reply_df["customer_tweet_id"].map(label_map)

reply_df = reply_df.dropna(subset=["intent"]).reset_index(drop=True)

print("Retrieval corpus:", len(reply_df))

# ---------------------------------------------------------
# 3. Prepare text
# ---------------------------------------------------------
corpus_text = reply_df["customer_text"].fillna("").astype(str)
golden_text = golden_df["text"].fillna("").astype(str)

# ---------------------------------------------------------
# 4. Build TF-IDF index
# ---------------------------------------------------------
print("\nBuilding TF-IDF retrieval index...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    max_features=150000
)

X = vectorizer.fit_transform(corpus_text)
X = normalize(X)

Q = vectorizer.transform(golden_text)
Q = normalize(Q)

print("TF-IDF index built.")

# ---------------------------------------------------------
# 5. Retrieve top-5 similar historical messages
# ---------------------------------------------------------
K = 5
results = []

for i in range(len(golden_df)):

    query_id = golden_df.iloc[i]["tweet_id"]
    query_text = golden_df.iloc[i]["text"]
    gold_intent = golden_df.iloc[i]["human_intent"]

    scores = Q[i].dot(X.T).toarray().ravel()

    top_indices = np.argsort(scores)[::-1][:K]

    retrieved_intents = reply_df.iloc[top_indices]["intent"].tolist()
    retrieved_ids = reply_df.iloc[top_indices]["customer_tweet_id"].tolist()
    retrieved_scores = scores[top_indices].tolist()

    # Rank of first retrieved example with the correct intent
    rank = None

    for r, retrieved_intent in enumerate(retrieved_intents, start=1):
        if retrieved_intent == gold_intent:
            rank = r
            break

    recall1 = int(rank is not None and rank <= 1)
    recall3 = int(rank is not None and rank <= 3)
    recall5 = int(rank is not None and rank <= 5)

    reciprocal_rank = 1.0 / rank if rank is not None else 0.0

    results.append({
        "tweet_id": query_id,
        "text": query_text,
        "human_intent": gold_intent,
        "retrieved_intent_rank": rank,
        "recall_at_1": recall1,
        "recall_at_3": recall3,
        "recall_at_5": recall5,
        "reciprocal_rank": reciprocal_rank,
        "top1_tweet_id": retrieved_ids[0],
        "top1_similarity": retrieved_scores[0],
        "top1_intent": retrieved_intents[0],
        "top1_reply": reply_df.iloc[top_indices[0]]["apple_support_reply"]
    })

# ---------------------------------------------------------
# 6. Calculate metrics
# ---------------------------------------------------------
result_df = pd.DataFrame(results)

print("\nEvaluation examples:", len(result_df))

if len(result_df) > 0:

    recall1 = result_df["recall_at_1"].mean()
    recall3 = result_df["recall_at_3"].mean()
    recall5 = result_df["recall_at_5"].mean()
    mrr = result_df["reciprocal_rank"].mean()

    print("\nRetrieval Metrics")
    print("-" * 40)
    print(f"Recall@1 : {recall1:.4f}")
    print(f"Recall@3 : {recall3:.4f}")
    print(f"Recall@5 : {recall5:.4f}")
    print(f"MRR      : {mrr:.4f}")

    print("\nPercentage")
    print("-" * 40)
    print(f"Recall@1 : {recall1 * 100:.2f}%")
    print(f"Recall@3 : {recall3 * 100:.2f}%")
    print(f"Recall@5 : {recall5 * 100:.2f}%")
    print(f"MRR      : {mrr:.4f}")

    result_df.to_csv(OUTPUT_FILE, index=False)

    print("\nSaved results to:", OUTPUT_FILE)

else:
    print("\nNo evaluation examples were available.")

print("=" * 70)