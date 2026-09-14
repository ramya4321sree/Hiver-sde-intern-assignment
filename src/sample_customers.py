import pandas as pd

FILE = "data/apple_support_raw.csv"

df = pd.read_csv(FILE)

# Keep only customer messages
customers = df[df["inbound"] == True].copy()

# Remove missing text
customers["text"] = customers["text"].fillna("")

# Take a reproducible random sample
sample = customers.sample(
    n=50,
    random_state=42
)

# Keep only the useful columns
sample = sample[
    ["tweet_id", "created_at", "text", "response_tweet_id"]
]

# Save the sample
output = "data/apple_customer_sample_50.csv"
sample.to_csv(output, index=False)

print("Customer messages available:", len(customers))
print("Sample created:", len(sample))
print("Saved to:", output)

print()
print("Sample customer messages:")
print("=" * 80)

for i, text in enumerate(sample["text"], start=1):
    print(f"{i}. {text}")
    print("-" * 80)