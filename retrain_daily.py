import json

from sympy import false

from mongo_service import get_feedbacks_for_retraining, mark_feedback_as_retrained
from train import train_model
from bson.objectid import ObjectId

INTENTS_PATH = "data/intents.json"

def load_intents():
    with open(INTENTS_PATH, "r") as f:
        return json.load(f)

def save_intents(intents):
    with open(INTENTS_PATH, "w") as f:
        json.dump(intents, f, indent=2)

def retrain_from_feedback():
    feedbacks = get_feedbacks_for_retraining()
    if not feedbacks:
        print("No new feedbacks for retraining.")
        return

    intents = load_intents()
    intent_map = {i["tag"]: i for i in intents["intents"]}

    used_ids = []
    for fb in feedbacks:
        tag = fb["correct_intent"]
        query = fb["query"]
        if tag in intent_map:
            if query not in intent_map[tag]["patterns"]:
                intent_map[tag]["patterns"].append(query)
        else:
            print(f"Warning: {tag} not found in intents.json")
        used_ids.append(ObjectId(fb["_id"]))

    save_intents({"intents": list(intent_map.values())})
    print("Intents updated. Starting training...")
    train_model(show_metrics=false)
    mark_feedback_as_retrained(used_ids)
    print("Retraining completed.")

if __name__ == "__main__":
    retrain_from_feedback()