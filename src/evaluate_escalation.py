import pandas as pd
from pathlib import Path

DATA = Path("data")
GOLDEN = DATA / "golden_set_final.xlsx"
REPLIES = DATA / "retrieval_predictions.csv"
OUTPUT = DATA / "escalation_eval.csv"


def decide_escalation(row):
    intent = row["human_intent"]
    similarity = float(row["top1_similarity"])

    text = str(row["text"]).lower()

    # Explicit request for a human
    human_request = any(
        phrase in text
        for phrase in [
            "human",
            "representative",
            "agent",
            "talk to someone",
            "speak to someone",
            "real person",
        ]
    )

    if human_request:
        return "ESCALATE", "Customer explicitly requests human assistance"

    # Low retrieval confidence
    if similarity < 0.25:
        return "ESCALATE", "Low historical-reply similarity"

    # Higher-risk intent
    if intent == "payments_purchases":
        return "ESCALATE", "Payment or purchase issue"

    # Account issues with insufficient historical support
    if intent == "account_activation" and similarity < 0.40:
        return "ESCALATE", "Account issue with limited historical support"

    # Ambiguous category
    if intent == "other":
        return "ESCALATE", "Message is outside supported intent categories"

    return "AUTO_HANDLE", "Supported intent with sufficient historical support"


def main():
    golden = pd.read_excel(GOLDEN)
    retrieval = pd.read_csv(REPLIES)

    golden["tweet_id"] = golden["tweet_id"].astype(str)
    retrieval["tweet_id"] = retrieval["tweet_id"].astype(str)

    df = golden[
        ["tweet_id", "text", "human_intent"]
    ].merge(
        retrieval[
            [
                "tweet_id",
                "top1_similarity",
                "top1_intent",
                "top1_reply",
            ]
        ],
        on="tweet_id",
        how="left",
    )

    decisions = df.apply(decide_escalation, axis=1, result_type="expand")
    decisions.columns = ["decision", "reason"]

    df = pd.concat([df, decisions], axis=1)

    # Summary
    total = len(df)
    auto = (df["decision"] == "AUTO_HANDLE").sum()
    escalate = (df["decision"] == "ESCALATE").sum()

    print("\n=== ESCALATION EVALUATION ===")
    print(f"Total examples: {total}")
    print(f"AUTO_HANDLE: {auto} ({auto / total:.1%})")
    print(f"ESCALATE:    {escalate} ({escalate / total:.1%})")

    print("\nEscalation reasons:")
    print(df.loc[df["decision"] == "ESCALATE", "reason"].value_counts())

    print("\nDecisions by intent:")
    print(
        pd.crosstab(
            df["human_intent"],
            df["decision"],
            margins=True,
        )
    )

    df.to_csv(OUTPUT, index=False)

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()