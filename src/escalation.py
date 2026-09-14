import re


def decide_escalation(
    intent,
    retrieval_similarity,
    customer_text
):
    """
    Decide whether the AI should auto-handle or escalate
    the customer message to a human.
    """

    text = customer_text.lower()

    # 1. Explicit request for a human
    human_request = re.search(
        r"\b(human|agent|representative|person|real person|speak to someone|talk to someone)\b",
        text
    )

    if human_request:
        return {
            "decision": "ESCALATE",
            "reason": "Customer explicitly requested a human agent."
        }

    # 2. Very weak historical match
    if retrieval_similarity < 0.25:
        return {
            "decision": "ESCALATE",
            "reason": "No sufficiently similar historical support case was found."
        }

    # 3. Payment / billing cases
    if intent == "payments_purchases":
        return {
            "decision": "ESCALATE",
            "reason": "Payment or purchase-related issue requires human review."
        }

    # 4. Account / activation issues with weak retrieval
    if intent == "account_activation" and retrieval_similarity < 0.40:
        return {
            "decision": "ESCALATE",
            "reason": "Account-related issue has insufficient historical support evidence."
        }

    # 5. General ambiguous cases
    if intent == "other":
        return {
            "decision": "ESCALATE",
            "reason": "Message could not be confidently mapped to a supported support intent."
        }

    # 6. Good historical match → auto-handle
    return {
        "decision": "AUTO_HANDLE",
        "reason": "Intent is supported and a sufficiently similar historical case was found."
    }


if __name__ == "__main__":

    print("=" * 60)
    print("ESCALATION POLICY TEST")
    print("=" * 60)

    tests = [
        {
            "text": "My iPhone battery is draining very quickly",
            "intent": "battery_power",
            "similarity": 0.49
        },
        {
            "text": "I want to speak to a human agent",
            "intent": "other",
            "similarity": 0.80
        },
        {
            "text": "I was charged twice for my purchase",
            "intent": "payments_purchases",
            "similarity": 0.70
        },
        {
            "text": "Something weird is happening",
            "intent": "other",
            "similarity": 0.15
        }
    ]

    for test in tests:

        result = decide_escalation(
            test["intent"],
            test["similarity"],
            test["text"]
        )

        print()
        print("Customer:", test["text"])
        print("Intent:", test["intent"])
        print("Similarity:", test["similarity"])
        print("Decision:", result["decision"])
        print("Reason:", result["reason"])

    print()
    print("=" * 60)