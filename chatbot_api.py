# ai_chatbot_dynamic.py (chatbot_api.py upgraded)

from fastapi import FastAPI, Request
from pydantic import BaseModel
from pymongo import MongoClient
import torch
import pickle
import random
import re
from datetime import datetime
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from fuzzywuzzy import fuzz

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

def load_model_and_tools():
    global model, vectorizer, label_encoder
    from train_bot import ChatModel
    with open("model/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("model/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    model = ChatModel(len(vectorizer.get_feature_names_out()), 64, len(label_encoder.classes_))
    model.load_state_dict(torch.load("model/intent_model.pt"))
    model.eval()

load_model_and_tools()

client = MongoClient("mongodb+srv://lprabodha1998:SfXnuKZIecrv3TUJ@cluster0.e2m4j.mongodb.net/")
db = client["vehicle_prices"]
history_col = db["chat_history"]

app = FastAPI()

class ChatInput(BaseModel):
    user_id: str
    message: str

def tokenize_and_lemmatize(text):
    tokens = word_tokenize(text)
    return " ".join([lemmatizer.lemmatize(word.lower()) for word in tokens if word.isalpha()])

@app.post("/chat")
def chat(chat_input: ChatInput):
    query = chat_input.message
    user_id = chat_input.user_id
    clean_query = tokenize_and_lemmatize(query)

    vec = vectorizer.transform([clean_query])
    input_tensor = torch.FloatTensor(vec.toarray())
    output = model(input_tensor)
    intent_idx = torch.argmax(output).item()
    intent = label_encoder.inverse_transform([intent_idx])[0]

    prev = history_col.find_one({"user_message": query}, sort=[("score", -1)])
    response = prev["bot_response"] if prev and prev.get("score", 0) > 0 else get_response(intent, query)

    history_col.insert_one({
        "user_id": user_id,
        "user_message": query,
        "bot_response": response,
        "intent": intent,
        "timestamp": datetime.utcnow(),
        "score": 0
    })
    return {"response": response, "intent": intent}

@app.post("/feedback/{doc_id}/{action}")
def feedback(doc_id: str, action: str):
    from bson import ObjectId
    doc = history_col.find_one({"_id": ObjectId(doc_id)})
    if not doc:
        return {"status": "not found"}
    history_col.update_one({"_id": doc["_id"]}, {"$inc": {"score": 1 if action == "like" else -1}})
    return {"status": "updated"}

@app.post("/reload")
def reload_model():
    load_model_and_tools()
    return {"status": "Model reloaded dynamically."}

def extract_attributes(query):
    keywords = {
        "location": ["location", "where", "place", "city"],
        "price": ["price", "cost", "amount", "value"],
        "brand_name": ["brand", "make"],
        "model_name": ["model"],
        "model_year": ["year"],
        "engine_capacity": ["engine", "cc", "power"],
        "transmission": ["transmission", "gear"],
        "fuel_type": ["fuel", "petrol", "electric", "diesel"],
        "phone_number": ["phone", "contact"],
        "seller_name": ["seller", "owner"]
    }
    return [key for key, words in keywords.items() if any(w in query.lower() for w in words)]

def fuzzy_match_location(query_location, db_location):
    return fuzz.partial_ratio(query_location.lower(), db_location.lower()) > 80

def fetch_selected_details(query):
    attrs = extract_attributes(query)
    keyword = query.lower()
    results = []
    for coll in ["cars", "motorcycles"]:
        for v in db[coll].find():
            brand = v.get("brand_name", "").lower()
            model = v.get("model_name", "").lower()
            location = v.get("location", "").lower()

            if brand in keyword or model in keyword or fuzzy_match_location(keyword, location):
                detail = []
                if not attrs:
                    attrs = ["price", "location"]
                for attr in attrs:
                    value = v.get(attr) or v.get("engine") if attr == "engine_capacity" else None
                    if value:
                        detail.append(f"{attr.replace('_', ' ').title()}: {value}")
                detail.append(f"More Info: {v['ad_url']}")
                results.append("\n".join(detail))
    return "\n---\n".join(results[:5]) if results else "I couldn’t find the vehicle you're asking about."

def get_dynamic_keywords(field):
    values = set()
    for coll in ["cars", "motorcycles"]:
        values.update(v.lower() for v in db[coll].distinct(field) if isinstance(v, str))
    return list(values)

def fetch_by_budget(query):
    numbers = re.findall(r"\d+", query.replace(",", ""))
    if not numbers:
        return "Please specify your budget clearly (e.g., 'under 5 million')."
    max_price = int(numbers[0]) * (1_000_000 if int(numbers[0]) < 1000 else 1)

    keyword = query.lower()
    brand_keywords = get_dynamic_keywords("brand_name")
    location_keywords = get_dynamic_keywords("location")

    matched_brand = next((b for b in brand_keywords if b in keyword), None)
    matched_location = next((l for l in location_keywords if l in keyword), None)

    found = []
    for coll in ["cars", "motorcycles"]:
        for v in db[coll].find({"price": {"$lte": max_price}}):
            brand = v.get("brand_name", "").lower()
            loc = v.get("location", "").lower()
            if matched_brand and matched_brand not in brand:
                continue
            if matched_location and matched_location not in loc:
                continue

            vehicle_name = v.get("vehicle_name", "Unnamed Vehicle")
            location = v.get("location", "Unknown Location")
            price = v.get("price", 0)
            url = v.get("ad_url", "")

            found.append(f"{vehicle_name} – Rs. {price:,} in {location}\n🌐 {url}")

    return "\n\n".join(found[:5]) if found else "No vehicles found matching your query."

def fetch_price_details(query):
    keyword = query.lower()
    results = []
    for coll in ["cars", "motorcycles"]:
        for v in db[coll].find():
            brand = v.get("brand_name", "").lower()
            model = v.get("model_name", "").lower()
            if brand in keyword or model in keyword:
                price = v.get("price", "N/A")
                name = v.get("vehicle_name", "Unknown")
                loc = v.get("location", "Unknown")
                url = v.get("ad_url", "")
                results.append(f"{name} – Rs. {price:,} in {loc}\n🌐 {url}")
    return "\n\n".join(results[:5]) if results else "Sorry, I couldn't find that vehicle."


def get_response(intent, query):
    fallback = False
    if intent not in ["vehicle_detail_request", "vehicle_price_query", "vehicle_location", "vehicle_by_budget_query"]:
        fallback = True

    if fallback:
        if "price" in query.lower():
            intent = "vehicle_price_query"
        elif "where" in query.lower() or "location" in query.lower():
            intent = "vehicle_location"
        elif re.search(r"under\\s+\\d+", query.lower()):
            intent = "vehicle_by_budget_query"
        else:
            return "Sorry, I didn't understand. Try asking about a car, bike, price, or brand."

    if intent == "vehicle_detail_request" or intent == "vehicle_price_query" or intent == "vehicle_location":
        return fetch_selected_details(query)
    elif intent == "vehicle_by_budget_query":
        return fetch_by_budget(query)
    elif intent == "vehicle_price_query":
        return fetch_price_details(query)
    elif intent == "thank":
        return "You're welcome! Let me know if you need more help."
    elif intent == "greet":
        return random.choice(["Hi there! 👋", "Hello! How can I help with your vehicle search today?"])
    
    return "I'm not sure how to help with that yet."

