import pandas as pd
import re

FILE = "data/apple_support_raw.csv"
OUTPUT = "data/apple_labeled.csv"


# Load AppleSupport working dataset
df = pd.read_csv(FILE)

# Keep only customer messages
df = df[df["inbound"] == True].copy()

# Handle missing text
df["text"] = df["text"].fillna("")

# Clean text
df["clean_text"] = (
    df["text"]
    .str.replace(r"@\w+", "", regex=True)
    .str.replace(r"https?://\S+", "", regex=True)
    .str.lower()
    .str.strip()
)


def assign_intent(text):

    # 1. iOS updates / upgrades
    if re.search(
        r"\b(update|updating|updated|upgrade|upgraded|downgrade|ios\s*\d+|ios11|ios12)\b",
        text
    ):
        return "ios_update"

    # 2. Keyboard / typing / text input
    if re.search(
        r"\b(keyboard|typing|type|typed|letter|letters|character|characters|texting|text input)\b",
        text
    ):
        return "keyboard_input"

    # 3. Battery / charging / power
    if re.search(
        r"\b(battery|charging|charge|drain|power|dies|dead battery)\b",
        text
    ):
        return "battery_power"

    # 4. Connectivity
    if re.search(
        r"\b(wifi|wi-fi|bluetooth|network|connected|connection|connect|disconnect|signal)\b",
        text
    ):
        return "connectivity"

    # 5. Account / activation / login
    if re.search(
        r"\b(apple id|password|activation|activate|login|log in|logged in|locked|lock|sign in|signin)\b",
        text
    ):
        return "account_activation"

    # 6. Payments / purchases / Apple Pay
    if re.search(
        r"\b(apple pay|debit card|credit card|payment|billing|purchase|purchased|charged|charge me|refund)\b",
        text
    ):
        return "payments_purchases"

    # 7. Music / iTunes / AirPlay / Apple TV
    if re.search(
        r"\b(music|itunes|airplay|apple tv|tv app|audio|video)\b",
        text
    ):
        return "music_media"

    # 8. Hardware / physical accessories
    if re.search(
        r"\b(earphone|earphones|headphone|headphones|cable|charger|charging cable|button|buttons|accessory|accessories|screen|display)\b",
        text
    ):
        return "hardware_accessories"

    # 9. Apps / Apple services / Safari / App Store / iCloud / Photos / Mail
    if re.search(
        r"\b(app|apps|app store|icloud|photos|photo app|email|emails|mail|safari|browser|macbook|mac os|osx|store|facetime)\b",
        text
    ):
        return "apps_services"

    # 10. General device performance / software problems
    if re.search(
        r"\b(freez|crash|crashing|slow|overheat|overheating|stuck|lag|lagging|bug|glitch|broken|problem|issue|error|not working|doesn't work|wont work|won't work)\b",
        text
    ):
        return "device_performance"

    # 11. Everything else
    return "other"


# Assign weak labels
df["intent"] = df["clean_text"].apply(assign_intent)

# Save labeled dataset
df.to_csv(OUTPUT, index=False)


print("Finished!")
print("Customer messages:", len(df))
print()
print("Intent distribution:")
print(df["intent"].value_counts())
print()
print("Saved to:", OUTPUT)