import pandas as pd
import os

INPUT = "data/golden_set.csv"
OUTPUT = "data/golden_set.csv"


# Intent labels
INTENTS = {
    "1": "ios_update",
    "2": "device_performance",
    "3": "keyboard_input",
    "4": "connectivity",
    "5": "battery_power",
    "6": "apps_services",
    "7": "music_media",
    "8": "account_activation",
    "9": "payments_purchases",
    "10": "hardware_accessories",
    "11": "other",
}


# Load golden set
df = pd.read_csv(INPUT)

# Make sure human_intent exists
if "human_intent" not in df.columns:
    df["human_intent"] = ""

df["human_intent"] = df["human_intent"].fillna("").astype(str)
# Find first unlabeled example
unlabeled = df[
    df["human_intent"].isna() |
    (df["human_intent"].astype(str).str.strip() == "")
]

if len(unlabeled) == 0:
    print("All examples are already labeled!")
    exit()


print("=" * 70)
print("APPLE SUPPORT — GOLDEN SET ANNOTATION")
print("=" * 70)

print()
print("Choose the intent that BEST describes the customer's message.")
print()

for key, value in INTENTS.items():
    print(f"{key:>2}. {value}")

print()
print("Commands:")
print("  s = skip this example")
print("  q = save and quit")
print()


for index in unlabeled.index:

    print()
    print("-" * 70)
    print(f"Example {index + 1} / {len(df)}")
    print("-" * 70)

    print()
    print("CUSTOMER MESSAGE:")
    print(df.loc[index, "text"])

    print()
    print("Weak label:", df.loc[index, "intent"])

    while True:

        choice = input("\nYour choice: ").strip().lower()

        if choice == "q":
            df.to_csv(OUTPUT, index=False)
            print()
            print("Progress saved.")
            print("Labeled examples:",
                  df["human_intent"].fillna("").astype(str).str.strip().ne("").sum())
            exit()

        if choice == "s":
            print("Skipped.")
            break

        if choice in INTENTS:
            df.loc[index, "human_intent"] = INTENTS[choice]

            # Save after every annotation
            df.to_csv(OUTPUT, index=False)

            print("Saved:", INTENTS[choice])
            break

        print("Invalid choice. Enter 1-11, s, or q.")


print()
print("=" * 70)
print("ANNOTATION COMPLETE")
print("=" * 70)

df.to_csv(OUTPUT, index=False)

print("Saved to:", OUTPUT)
print(
    "Human-labeled examples:",
    df["human_intent"].fillna("").astype(str).str.strip().ne("").sum()
)