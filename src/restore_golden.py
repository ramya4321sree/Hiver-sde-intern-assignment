import pandas as pd
from openpyxl import load_workbook
from openpyxl.worksheet.datavalidation import DataValidation

INPUT = "data/golden_annotation_clean.xlsx"
OUTPUT = "data/golden_review.xlsx"

df = pd.read_excel(INPUT)
df["human_intent"] = df["human_intent"].fillna("")

# Create suggested labels using the weak labels already generated.
df["suggested_intent"] = df["weak_intent"]

# Keep your 11 genuine labels separate.
df["reviewed"] = df["human_intent"].ne("")

# For already-labelled rows, suggestion is the actual human label.
df.loc[df["reviewed"], "suggested_intent"] = df.loc[
    df["reviewed"], "human_intent"
]

# Put the 11 already-labelled examples first.
df["sort_order"] = df["reviewed"].map({True: 0, False: 1})
df = df.sort_values(["sort_order"]).drop(columns=["sort_order"])

df = df[
    [
        "tweet_id",
        "text",
        "weak_intent",
        "suggested_intent",
        "human_intent",
        "reviewed",
    ]
]

df.to_excel(OUTPUT, index=False)

# Add dropdown for human review.
wb = load_workbook(OUTPUT)
ws = wb.active

options = (
    "ios_update,device_performance,keyboard_input,connectivity,"
    "battery_power,apps_services,music_media,account_activation,"
    "payments_purchases,hardware_accessories,other"
)

dv = DataValidation(
    type="list",
    formula1=f'"{options}"',
    allow_blank=True
)

ws.add_data_validation(dv)
dv.add(f"E2:E{len(df)+1}")

ws.freeze_panes = "A2"
ws.auto_filter.ref = ws.dimensions

wb.save(OUTPUT)

print("=" * 60)
print("GOLDEN REVIEW FILE CREATED")
print("=" * 60)
print("Total examples:", len(df))
print("Already genuinely labelled:", df["human_intent"].ne("").sum())
print("Remaining to review:", df["human_intent"].eq("").sum())
print()
print("File:", OUTPUT)