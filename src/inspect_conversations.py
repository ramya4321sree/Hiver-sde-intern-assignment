import pandas as pd

FILE = "data/apple_support_raw.csv"

df = pd.read_csv(FILE)

# Get AppleSupport replies
replies = df[
    (df["inbound"] == False)
    & (df["author_id"] == "AppleSupport")
].copy()

# Get customer messages
customers = df[df["inbound"] == True].copy()

# Map tweet IDs to text
customer_text = dict(
    zip(customers["tweet_id"], customers["text"])
)

print("AppleSupport replies available:", len(replies))
print()
print("=" * 80)

shown = 0

for _, reply in replies.iterrows():

    parent_id = reply["in_response_to_tweet_id"]

    if pd.isna(parent_id):
        continue

    try:
        parent_id = int(parent_id)
    except:
        continue

    if parent_id not in customer_text:
        continue

    print("CUSTOMER:")
    print(customer_text[parent_id])

    print()
    print("APPLE SUPPORT:")
    print(reply["text"])

    print()
    print("-" * 80)

    shown += 1

    if shown == 10:
        break

print()
print("Displayed conversations:", shown)