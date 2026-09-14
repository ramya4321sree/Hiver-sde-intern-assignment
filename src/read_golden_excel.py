import pandas as pd

FILE = "data/golden_annotation_final.xlsx"

df = pd.read_excel(FILE)

print("=" * 60)
print("GOLDEN SET LOADED")
print("=" * 60)

print("Total examples:", len(df))
print()
print("Columns:")
print(df.columns.tolist())
print()
print(df.head(10).to_string(index=False))