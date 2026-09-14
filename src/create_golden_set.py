import pandas as pd

INPUT = "data/apple_labeled.csv"
OUTPUT = "data/golden_set.csv"

GOLDEN_SIZE = 200
RANDOM_STATE = 42


# Load weakly labelled data
df = pd.read_csv(INPUT)

# Keep required columns
df = df[["tweet_id", "text", "intent"]].copy()

# Remove empty messages
df["text"] = df["text"].fillna("").str.strip()
df = df[df["text"] != ""]

# Calculate proportional sample size
# while ensuring every intent gets at least 10 examples
intent_counts = df["intent"].value_counts()

base_per_intent = 10
remaining = GOLDEN_SIZE - (base_per_intent * len(intent_counts))

# Proportional allocation for remaining examples
proportions = intent_counts / intent_counts.sum()
extra_counts = (proportions * remaining).round().astype(int)

# Adjust rounding so total is exactly GOLDEN_SIZE
difference = GOLDEN_SIZE - (
    base_per_intent * len(intent_counts) + extra_counts.sum()
)

if difference > 0:
    largest_intents = intent_counts.sort_values(ascending=False).index

    for intent in largest_intents[:difference]:
        extra_counts[intent] += 1

elif difference < 0:
    smallest_extras = extra_counts.sort_values(ascending=False).index

    for intent in smallest_extras[:abs(difference)]:
        if extra_counts[intent] > 0:
            extra_counts[intent] -= 1


# Create stratified sample
samples = []

for intent in intent_counts.index:

    n = base_per_intent + extra_counts[intent]

    intent_df = df[df["intent"] == intent]

    sample = intent_df.sample(
        n=min(n, len(intent_df)),
        random_state=RANDOM_STATE
    )

    samples.append(sample)


golden = pd.concat(samples, ignore_index=True)

# Shuffle the final dataset
golden = golden.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)


# Add blank column for human annotation
golden["human_intent"] = ""

# Keep only what we need for annotation
golden = golden[
    ["tweet_id", "text", "intent", "human_intent"]
]


# Save
golden.to_csv(OUTPUT, index=False)


print("Golden set created!")
print("Number of examples:", len(golden))
print()
print("Weak-label distribution:")
print(golden["intent"].value_counts())
print()
print("Saved to:", OUTPUT)