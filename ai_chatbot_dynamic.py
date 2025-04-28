from fastapi import FastAPI
from pydantic import BaseModel
from pymongo import MongoClient
import torch
import pickle
import random
import re
from datetime import datetime
from fetch_utils import fetch_selected_details, fetch_by_budget, fetch_by_model_year, fetch_seller_info, fetch_by_model_year
import json
from preprocessing.clean_text import preprocess


client = MongoClient("mongodb+srv://lprabodha1998:SfXnuKZIecrv3TUJ@cluster0.e2m4j.mongodb.net/")
db = client["vehicle_prices"]
history_col = db["chat_history"]

app = FastAPI()

def load_model_and_tools():
    global model, vectorizer, label_encoder
    from train_model import ChatModel
    with open("model/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("model/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    model = ChatModel(len(vectorizer.get_feature_names_out()), 128, len(label_encoder.classes_))
    model.load_state_dict(torch.load("model/intent_model.pt"))
    model.eval()

load_model_and_tools()


class ChatInput(BaseModel):
    user_id: str
    message: str

@app.post("/chat")
def chat(chat_input: ChatInput):
    query = chat_input.message
    user_id = chat_input.user_id
    clean_query = preprocess(query)

    vec = vectorizer.transform([clean_query])
    input_tensor = torch.FloatTensor(vec.toarray())
    output = model(input_tensor)
    intent_idx = torch.argmax(output).item()
    confidence = torch.softmax(output, dim=1)[0][intent_idx].item()
    intent = label_encoder.inverse_transform([intent_idx])[0]

    response = get_response(intent, query)

    history_col.insert_one({
        "user_id": user_id,
        "user_message": query,
        "bot_response": response,
        "intent": intent,
        "confidence": round(confidence, 3),
        "timestamp": datetime.utcnow(),
        "score": 0
    })

    return {"response": response, "intent": intent, "confidence": round(confidence, 3)}

@app.post("/reload")
def reload_model():
    load_model_and_tools()
    return {"status": "Model reloaded dynamically."}

@app.post("/feedback/{doc_id}/{action}")
def feedback(doc_id: str, action: str):
    from bson import ObjectId
    doc = history_col.find_one({"_id": ObjectId(doc_id)})
    if not doc:
        return {"status": "not found"}
    history_col.update_one({"_id": doc["_id"]}, {"$inc": {"score": 1 if action == "like" else -1}})
    return {"status": "updated"}

def get_response(intent, query):
    try:
        with open("data/intents.json", "r") as f:
            intent_data = json.load(f)
        static_map = {i['tag']: i.get('responses', []) for i in intent_data['intents']}
    except Exception:
        static_map = {}

    lowered = query.lower()

    if any(w in lowered for w in ["hi", "hello", "hey", "greetings", "good morning"]):
        return random.choice(static_map.get("greet", ["Hi there! 👋 How can I help you find a vehicle?"]))
    
    if any(w in lowered for w in ["bye", "goodbye", "see you"]):
        return random.choice(static_map.get("goodbye", ["Goodbye! 👋 Have a great day!"]))
    
    if any(w in lowered for w in ["thank", "thanks", "appreciate"]):
        return random.choice(static_map.get("thank", ["You're welcome! 🚗"]))
    
    if any(w in lowered for w in ["your name", "who are you", "who made you", "are you a chatbot", "what do you do"]):
        intent = "bot_identity"
    
    finance_keywords = ["lease", "financing", "loan", "monthly payment", "installment", "emi", "hire purchase"]
    if any(w in lowered for w in finance_keywords):
        intent = "vehicle_financing_query"
    
    seller_keywords = ["seller", "contact", "phone number", "owner", "call seller", "seller info"]
    if any(w in lowered for w in seller_keywords):
        intent = "seller_name_query"
        
    auto_keywords = ["car", "bike", "vehicle", "price", "model", "brand", "motorcycle", "seller", "year", "buy", "sell"]
    if not any(word in lowered for word in auto_keywords):
        return "❓ I'm specialized in assisting with auto sales. 🚗🏍️ Please ask about vehicles, cars, motorcycles, prices, locations, or sellers."

    if intent in static_map and static_map[intent]:
        return random.choice(static_map[intent])

    if intent in ["vehicle_detail_request", "vehicle_price_query", "vehicle_location"]:
        return fetch_selected_details(query)
    elif intent == "vehicle_by_budget_query":
        return fetch_by_budget(query)
    elif intent == "vehicle_model_year_query":
        return fetch_by_model_year(query)
    elif intent == "seller_name_query":
        return fetch_seller_info(query)
    elif intent == "vehicle_financing_query":
        return random.choice(static_map.get("vehicle_financing_query", ["Some sellers offer financing options. 🚗💳"]))

    fallback_responses = [
        "😔 Sorry, I couldn't find an exact match. Let’s try another search!",
        "🚗 No matching results. Want to check another vehicle?",
        "🤔 Hmm, I didn't find that. Maybe try a different brand or model?",
    ]
    return random.choice(fallback_responses)
