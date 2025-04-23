import json
import subprocess
from pymongo import MongoClient
from datetime import datetime
from sklearn.utils import shuffle

client = MongoClient("mongodb+srv://lprabodha1998:SfXnuKZIecrv3TUJ@cluster0.e2m4j.mongodb.net/")
db = client["vehicle_prices"]
history = db["chat_history"]

intents_path = "data/intents.json"

with open(intents_path, "r") as f:
    current_intents = json.load(f)

if isinstance(current_intents, dict) and "intents" in current_intents:
    base_entries = [{"text": pat, "intent": i["tag"]} for i in current_intents["intents"] for pat in i.get("patterns", [])]
else:
    base_entries = [{"text": entry["text"], "intent": entry["intent"]} for entry in current_intents]

existing_texts = set(e["text"] for e in base_entries)

samples = []
for item in history.find({"score": {"$gte": 1}, "confidence": {"$gte": 0.5}}):
    text = item.get("user_message")
    intent = item.get("intent", "fallback_guess")
    if text:
        samples.append({"text": text.strip(), "intent": intent.strip()})

new_entries = [s for s in samples if s["text"] not in existing_texts]
total_added = len(new_entries)

if total_added:
    merged = base_entries + new_entries
    merged = shuffle(merged, random_state=42)
    with open(intents_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"✅ Added {total_added} new training samples from MongoDB feedback.")
else:
    print("ℹ️ No new samples to add.")

# Retrain model
print("🔄 Running retrain pipeline...")
subprocess.run(["python", "train_bot.py"])
print("✅ Retraining complete. Model updated with feedback.")