import pandas as pd

FILE = "data/apple_labeled.csv"

df = pd.read_csv(FILE)

other = df[df["intent"] == "other"].copy()

sample = other.sample(
    n=50,
    random_state=42
)

print("Other messages:", len(other))
print()
print("=" * 80)

for i, text in enumerate(sample["text"], start=1):
    print(f"{i}. {text}")
    print("-" * 80)