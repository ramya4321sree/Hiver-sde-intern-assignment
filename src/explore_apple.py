# import pandas as pd
# from pathlib import Path

# DATA = Path(r"C:\Users\RAMYA SREE\Downloads\archive\twcs\twcs.csv")

# BRAND = "@AppleSupport"
# CHUNK_SIZE = 100_000

# apple_customer_ids = set()
# apple_customer_count = 0

# print("Step 1: Finding AppleSupport customer tweets...")

# for chunk in pd.read_csv(
#     DATA,
#     usecols=["tweet_id", "inbound", "text"],
#     chunksize=CHUNK_SIZE
# ):
#     mask = (
#         chunk["inbound"].eq(True)
#         & chunk["text"].fillna("").str.contains(
#             BRAND,
#             case=False,
#             regex=False
#         )
#     )

#     matching = chunk.loc[mask, "tweet_id"].dropna()

#     apple_customer_count += len(matching)

#     apple_customer_ids.update(
#         matching.astype("int64").astype(str).tolist()
#     )

# print("AppleSupport customer tweets:", apple_customer_count)
# print("Unique customer tweet IDs:", len(apple_customer_ids))

# print()
# print("Step 2: Finding actual company replies to those tweets...")

# reply_count = 0
# reply_authors = {}

# for chunk in pd.read_csv(
#     DATA,
#     usecols=[
#         "tweet_id",
#         "author_id",
#         "inbound",
#         "created_at",
#         "text",
#         "in_response_to_tweet_id"
#     ],
#     chunksize=CHUNK_SIZE
# ):
#     parent_ids = (
#         chunk["in_response_to_tweet_id"]
#         .astype("Int64")
#         .astype("string")
#     )

#     mask = (
#         chunk["inbound"].eq(False)
#         & parent_ids.isin(apple_customer_ids)
#     )

#     replies = chunk.loc[mask]

#     reply_count += len(replies)

#     for author, count in replies["author_id"].value_counts().items():
#         author = str(author)
#         reply_authors[author] = (
#             reply_authors.get(author, 0) + int(count)
#         )

# print("Actual outbound replies:", reply_count)

# print()
# print("Top reply authors:")

# for author, count in sorted(
#     reply_authors.items(),
#     key=lambda x: x[1],
#     reverse=True
# )[:20]:
#     print(f"{author}: {count}")




import pandas as pd
from pathlib import Path

DATA = Path(r"C:\Users\RAMYA SREE\Downloads\archive\twcs\twcs.csv")
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "apple_support_raw.csv"

CHUNK_SIZE = 100_000

print("Extracting AppleSupport data...")
print()

total_rows = 0

for chunk in pd.read_csv(
    DATA,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id"
    ],
    chunksize=CHUNK_SIZE
):

    mask = (
        (chunk["author_id"] == "AppleSupport")
        |
        (
            chunk["inbound"].eq(True)
            & chunk["text"].fillna("").str.contains(
                "@AppleSupport",
                case=False,
                regex=False
            )
        )
    )

    selected = chunk.loc[mask]

    if not selected.empty:
        selected.to_csv(
            OUTPUT_FILE,
            mode="a",
            header=not OUTPUT_FILE.exists(),
            index=False
        )

        total_rows += len(selected)

print()
print("Finished!")
print("AppleSupport working rows:", total_rows)
print("Saved to:", OUTPUT_FILE)