import pandas as pd

RETRIEVAL_FILE = "data/retrieval_predictions.csv"
OUTPUT_FILE = "data/reply_quality_eval.csv"

df = pd.read_csv(RETRIEVAL_FILE)

# Keep the fields needed for reply-quality evaluation
eval_df = df[
    [
        "tweet_id",
        "text",
        "human_intent",
        "top1_similarity",
        "top1_intent",
        "top1_reply",
        "retrieved_intent_rank",
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
    ]
].copy()

# Add a simple grounding flag.
# A retrieved reply is considered grounded when a historical
# reply was actually retrieved for the customer message.
eval_df["historically_grounded"] = eval_df["top1_reply"].notna()

# Add empty fields for LLM-as-judge scores.
eval_df["judge_relevance"] = ""
eval_df["judge_helpfulness"] = ""
eval_df["judge_groundedness"] = ""
eval_df["judge_overall"] = ""
eval_df["judge_reason"] = ""

eval_df.to_csv(OUTPUT_FILE, index=False)

print("=" * 70)
print("REPLY QUALITY EVALUATION DATASET")
print("=" * 70)
print("Examples:", len(eval_df))
print("Saved to:", OUTPUT_FILE)
print()
print("Columns:")
for col in eval_df.columns:
    print(" -", col)
print("=" * 70)