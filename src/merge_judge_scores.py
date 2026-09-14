import pandas as pd
from pathlib import Path

DATA = Path("data")

REPLY_FILE = DATA / "reply_quality_eval.csv"
JUDGE_FILE = DATA / "proposed_human_scores_200.csv"

reply = pd.read_csv(REPLY_FILE)
judge = pd.read_csv(JUDGE_FILE)

# Normalize tweet IDs
reply["tweet_id"] = reply["tweet_id"].astype(str).str.strip()
judge["tweet_id"] = judge["tweet_id"].astype(str).str.strip()

# Keep the actual Claude/LLM scores
judge = judge[
    [
        "tweet_id",
        "llm_relevance",
        "llm_helpfulness",
        "llm_groundedness",
        "llm_overall",
        "reason",
    ]
].copy()

judge = judge.rename(
    columns={
        "llm_relevance": "judge_relevance",
        "llm_helpfulness": "judge_helpfulness",
        "llm_groundedness": "judge_groundedness",
        "llm_overall": "judge_overall",
        "reason": "judge_reason",
    }
)

# Remove any existing empty judge columns
for col in [
    "judge_relevance",
    "judge_helpfulness",
    "judge_groundedness",
    "judge_overall",
    "judge_reason",
]:
    if col in reply.columns:
        reply = reply.drop(columns=col)

# Merge scores
reply = reply.merge(
    judge,
    on="tweet_id",
    how="left"
)

# Check
print("Total reply examples:", len(reply))
print("Judge examples:", len(judge))

print("\n=== JUDGE SCORE CHECK ===")

for col in [
    "judge_relevance",
    "judge_helpfulness",
    "judge_groundedness",
    "judge_overall",
]:
    print(
        f"{col}: "
        f"{reply[col].notna().sum()} populated"
    )

print("\n=== JUDGE AVERAGES ===")

for col in [
    "judge_relevance",
    "judge_helpfulness",
    "judge_groundedness",
    "judge_overall",
]:
    print(
        f"{col}: "
        f"{reply[col].mean():.3f}/5"
    )

# Overwrite reply-quality evaluation with corrected version
reply.to_csv(
    REPLY_FILE,
    index=False
)

print(f"\nUpdated: {REPLY_FILE}")