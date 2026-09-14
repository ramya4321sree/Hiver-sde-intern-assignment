import pandas as pd
import re

INPUT = "data/golden_annotation_clean.xlsx"
OUTPUT = "data/golden_review.xlsx"

df = pd.read_excel(INPUT)
df["human_intent"] = df["human_intent"].fillna("").astype(str).str.strip()

def suggest_intent(text):
    text = str(text).lower()

    # 1. Payments / purchases
    if re.search(r"\b(apple pay|payment|billing|purchase|purchased|charged|refund|debit card|credit card|price|cost)\b", text):
        return "payments_purchases"

    # 2. Account / login / activation
    if re.search(r"\b(apple id|password|login|log in|logged in|sign in|signin|activation|activate|locked out|account)\b", text):
        return "account_activation"

    # 3. Battery / charging
    if re.search(r"\b(battery|batteries|charging|charger|charge|drain|draining|dies|dead battery|power)\b", text):
        return "battery_power"

    # 4. Connectivity
    if re.search(r"\b(wifi|wi-fi|bluetooth|network|connection|connected|connect|disconnect|signal|cellular|mobile data)\b", text):
        return "connectivity"

    # 5. Keyboard / typing
    if re.search(r"\b(keyboard|typing|typing|type|typed|letter|letters|character|characters|texting|autocorrect)\b", text):
        return "keyboard_input"

    # 6. Music / media
    if re.search(r"\b(music|itunes|airplay|apple tv|tv app|audio|song|songs|video|videos|podcast)\b", text):
        return "music_media"

    # 7. Hardware / physical accessories
    if re.search(r"\b(earphone|earphones|headphone|headphones|cable|button|buttons|accessory|accessories|screen|display|speaker|microphone|camera lens)\b", text):
        return "hardware_accessories"

    # 8. Apps / Apple services
    if re.search(r"\b(maps|map|navigation|safari|browser|app|apps|app store|icloud|photos|photo app|email|emails|mail|facetime|imessage|messages|macbook|mac os|osx|store)\b", text):
        return "apps_services"

    # 9. iOS updates
    if re.search(r"\b(update|updating|updated|upgrade|upgraded|downgrade|ios\s*\d+)\b", text):
        return "ios_update"

    # 10. General device/software problems
    if re.search(r"\b(freez|crash|crashing|slow|overheat|overheating|stuck|lag|lagging|bug|glitch|broken|problem|issue|error|not working|doesn't work|wont work|won't work)\b", text):
        return "device_performance"

    return "other"


suggestions = []

for _, row in df.iterrows():
    if row["human_intent"] != "":
        suggestions.append(row["human_intent"])
    else:
        suggestions.append(suggest_intent(row["text"]))

df["suggested_intent"] = suggestions
df["reviewed"] = df["human_intent"] != ""

df = df[
    ["tweet_id", "text", "weak_intent",
     "suggested_intent", "human_intent", "reviewed"]
]

df.to_excel(OUTPUT, index=False)

print("=" * 60)
print("CORRECTED GOLDEN SUGGESTIONS CREATED")
print("=" * 60)
print("Total examples:", len(df))
print("Already human-labelled:", df["human_intent"].ne("").sum())
print("Remaining to review:", df["human_intent"].eq("").sum())
print("Output:", OUTPUT)
