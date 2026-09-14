import pandas as pd
from pathlib import Path

DATA = Path("data")
INPUT = DATA / "reply_quality_eval.csv"
OUTPUT = DATA / "failure_analysis.csv"


def n(x):
    try:
        return float(x)
    except:
        return 0.0


def classify(row):

    human = str(row["human_intent"])
    retrieved = str(row["top1_intent"])

    sim = n(row["top1_similarity"])
    relevance = n(row["judge_relevance"])
    helpfulness = n(row["judge_helpfulness"])
    groundedness = n(row["judge_groundedness"])
    overall = n(row["judge_overall"])

    # Most directly observable reply-quality failures first.

    if relevance <= 2:
        if retrieved != human:
            return (
                "Wrong historical example causing irrelevance",
                "The retrieved example belongs to another intent and the resulting response is not relevant."
            )
        return (
            "Irrelevant response",
            "The response does not adequately address the customer's issue."
        )

    if helpfulness <= 2:
        return (
            "Generic or insufficient response",
            "The response is related but provides limited actionable assistance."
        )

    if groundedness <= 2:
        return (
            "Weak historical grounding",
            "The response is not sufficiently supported by the retrieved historical example."
        )

    if sim < 0.25:
        return (
            "Low retrieval confidence",
            "The best historical match has low similarity to the customer message."
        )

    if retrieved != human and overall <= 2:
        return (
            "Wrong historical example / intent retrieved",
            "The retrieved example differs from the human-reviewed intent and produces a low-quality response."
        )

    return (
        "No major failure",
        "The response did not trigger a major failure criterion."
    )


def main():

    df = pd.read_csv(INPUT)

    df["tweet_id"] = df["tweet_id"].astype(str).str.strip()

    for col in [
        "top1_similarity",
        "judge_relevance",
        "judge_helpfulness",
        "judge_groundedness",
        "judge_overall"
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    result = df.apply(
        classify,
        axis=1,
        result_type="expand"
    )

    result.columns = [
        "failure_mode",
        "failure_explanation"
    ]

    df = pd.concat([df, result], axis=1)

    df.to_csv(OUTPUT, index=False)

    print("\n=== FINAL FAILURE MODE ANALYSIS ===")

    summary = (
        df["failure_mode"]
        .value_counts()
        .rename_axis("failure_mode")
        .reset_index(name="count")
    )

    summary["percentage"] = (
        summary["count"] / len(df) * 100
    ).round(1)

    print(summary.to_string(index=False))

    print("\n=== LLM JUDGE AVERAGES ===")

    print(f"Relevance:     {df['judge_relevance'].mean():.3f}/5")
    print(f"Helpfulness:   {df['judge_helpfulness'].mean():.3f}/5")
    print(f"Groundedness:  {df['judge_groundedness'].mean():.3f}/5")
    print(f"Overall:       {df['judge_overall'].mean():.3f}/5")

    print("\n=== TOP 5 FAILURE MODES ===")

    failures = summary[
        summary["failure_mode"] != "No major failure"
    ].head(5)

    for _, item in failures.iterrows():

        mode = item["failure_mode"]

        print("\n" + "=" * 70)
        print(
            f"{mode}: "
            f"{int(item['count'])} "
            f"({item['percentage']:.1f}%)"
        )
        print("=" * 70)

        examples = df[
            df["failure_mode"] == mode
        ].head(2)

        for _, row in examples.iterrows():

            print(f"\nTweet ID: {row['tweet_id']}")
            print(f"Customer: {row['text']}")
            print(f"Human intent: {row['human_intent']}")
            print(f"Retrieved intent: {row['top1_intent']}")
            print(f"Similarity: {n(row['top1_similarity']):.3f}")
            print(f"Relevance: {n(row['judge_relevance']):.1f}/5")
            print(f"Helpfulness: {n(row['judge_helpfulness']):.1f}/5")
            print(f"Groundedness: {n(row['judge_groundedness']):.1f}/5")
            print(f"Overall: {n(row['judge_overall']):.1f}/5")
            print(f"Judge reason: {row['judge_reason']}")

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()