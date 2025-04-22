# retrain_from_feedback.py

import json
from pymongo import MongoClient
from datetime import datetime
import subprocess

client = MongoClient("mongodb+srv://lprabodha1998:SfXnuKZIecrv3TUJ@cluster0.e2m4j.mongodb.net/")
db = client["vehicle_prices"]
history = db["chat_history"]

with open("data/intents.json", "r") as f:
    current_intents = json.load(f)

samples = []
for item in history.find({"score": {"$gte": 1}}):
    if item.get("user_message") and item.get("intent"):
        samples.append({"text": item["user_message"], "intent": item["intent"]})

existing_texts = set(entry["text"] for entry in current_intents)
new_entries = [s for s in samples if s["text"] not in existing_texts]

total_added = len(new_entries)

if total_added > 0:
    merged_intents = current_intents + new_entries
    with open("data/intents.json", "w") as f:
        json.dump(merged_intents, f, indent=2)
    print(f"✅ Added {total_added} new training samples from feedback.")
else:
    print("ℹ️ No new entries found to add.")

# Retrain the model
print("🔄 Retraining the model with updated intents.json...")
subprocess.call(["python", "train_bot.py"])
print("✅ Model retrained and ready!")