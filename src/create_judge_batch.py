import pandas as pd
import json

INPUT_FILE = "data/reply_quality_eval.csv"
OUTPUT_FILE = "data/llm_judge_batch.jsonl"

df = pd.read_csv(INPUT_FILE)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        item = {
            "tweet_id": str(row["tweet_id"]),
            "customer_message": str(row["text"]),
            "human_intent": str(row["human_intent"]),
            "historical_reply": str(row["top1_reply"]),
            "retrieval_similarity": float(row["top1_similarity"]),
            "retrieved_intent": str(row["top1_intent"])
        }

        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("=" * 70)
print("LLM JUDGE BATCH CREATED")
print("=" * 70)
print("Examples:", len(df))
print("Saved to:", OUTPUT_FILE)
print("=" * 70)