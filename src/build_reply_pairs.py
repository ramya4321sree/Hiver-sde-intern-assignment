import pandas as pd

INPUT = "data/apple_support_raw.csv"
OUTPUT = "data/apple_reply_pairs.csv"

print("Loading AppleSupport data...")

df = pd.read_csv(INPUT)

# Clean text
df["text"] = df["text"].fillna("")

# Normalize tweet IDs properly
df["tweet_id"] = pd.to_numeric(
    df["tweet_id"], errors="coerce"
).astype("Int64").astype(str)

df["in_response_to_tweet_id"] = pd.to_numeric(
    df["in_response_to_tweet_id"], errors="coerce"
).astype("Int64").astype(str)

# Customer messages
customers = df[df["inbound"] == True].copy()

# Outbound AppleSupport replies
replies = df[df["inbound"] == False].copy()

print("Customer messages:", len(customers))
print("Outbound replies:", len(replies))

# Keep replies that directly respond to a customer tweet
customer_ids = set(customers["tweet_id"])

replies = replies[
    replies["in_response_to_tweet_id"].isin(customer_ids)
].copy()

print("Direct replies found:", len(replies))

# --------------------------------------------------
# Match customer → reply
# --------------------------------------------------

pairs = customers.merge(
    replies[
        [
            "tweet_id",
            "text",
            "created_at",
            "in_response_to_tweet_id"
        ]
    ],
    left_on="tweet_id",
    right_on="in_response_to_tweet_id",
    how="inner",
    suffixes=("_customer", "_reply")
)

pairs = pairs[
    [
        "tweet_id_customer",
        "text_customer",
        "text_reply",
        "created_at_customer",
        "created_at_reply"
    ]
]

pairs = pairs.rename(
    columns={
        "tweet_id_customer": "customer_tweet_id",
        "text_customer": "customer_text",
        "text_reply": "apple_support_reply",
        "created_at_customer": "customer_created_at",
        "created_at_reply": "reply_created_at"
    }
)

# Remove empty messages
pairs = pairs[
    (pairs["customer_text"].str.strip() != "") &
    (pairs["apple_support_reply"].str.strip() != "")
]

# One reply pair per customer tweet
pairs = pairs.drop_duplicates(
    subset=["customer_tweet_id"]
)

pairs.to_csv(OUTPUT, index=False)

print()
print("=" * 60)
print("REPLY PAIRS CREATED")
print("=" * 60)
print("Customer → AppleSupport pairs:", len(pairs))
print("Saved to:", OUTPUT)

print()
print("Sample:")
print(
    pairs[
        ["customer_text", "apple_support_reply"]
    ].head(5).to_string(index=False)
)

print("=" * 60)