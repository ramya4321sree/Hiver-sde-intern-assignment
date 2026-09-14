import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, Alignment
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.utils import get_column_letter

INPUT = "data/golden_set.csv"
OUTPUT = "data/golden_annotation.xlsx"

INTENTS = [
    "ios_update",
    "device_performance",
    "keyboard_input",
    "connectivity",
    "battery_power",
    "apps_services",
    "music_media",
    "account_activation",
    "payments_purchases",
    "hardware_accessories",
    "other",
]

# Read golden set
df = pd.read_csv(INPUT)

# Make sure human_intent exists
if "human_intent" not in df.columns:
    df["human_intent"] = ""

df["human_intent"] = df["human_intent"].fillna("").astype(str)

# Keep only the columns needed for annotation
df = df[["tweet_id", "text", "intent", "human_intent"]]

# Rename weak label for clarity
df = df.rename(columns={"intent": "weak_intent"})

# Create Excel file
df.to_excel(OUTPUT, index=False, sheet_name="Golden Set")

# Open workbook
wb = load_workbook(OUTPUT)
ws = wb["Golden Set"]

# Header formatting
for cell in ws[1]:
    cell.font = Font(bold=True)
    cell.alignment = Alignment(horizontal="center", vertical="center")

# Freeze header
ws.freeze_panes = "A2"

# Enable text wrapping
for row in ws.iter_rows():
    for cell in row:
        cell.alignment = Alignment(
            vertical="top",
            wrap_text=True
        )

# Column widths
ws.column_dimensions["A"].width = 18
ws.column_dimensions["B"].width = 80
ws.column_dimensions["C"].width = 25
ws.column_dimensions["D"].width = 30

# Create dropdown for human_intent
intent_list = ",".join(INTENTS)

dropdown = DataValidation(
    type="list",
    formula1=f'"{intent_list}"',
    allow_blank=True
)

dropdown.error = "Please select an intent from the dropdown."
dropdown.errorTitle = "Invalid Intent"
dropdown.prompt = "Select the correct human intent."
dropdown.promptTitle = "Human Intent"

ws.add_data_validation(dropdown)

# Apply dropdown to all 200 rows
dropdown.add(f"D2:D{len(df) + 1}")

# Save
wb.save(OUTPUT)

print("=" * 60)
print("GOLDEN SET EXCEL CREATED")
print("=" * 60)
print()
print("Total examples:", len(df))
print("Excel file:", OUTPUT)
print()
print("Open the Excel file and fill the 'human_intent' column.")
print("You can use the dropdown to select each intent.")
print()
print("You only need to label 150-250 examples.")
print("=" * 60)