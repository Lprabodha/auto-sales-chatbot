from pymongo import MongoClient
from datetime import datetime

client = MongoClient("mongodb+srv://lprabodha1998:SfXnuKZIecrv3TUJ@cluster0.e2m4j.mongodb.net/")
db = client["vehicle_prices"]
collection = db["cars"]

def get_vehicles_by_brand_model(brand, model=None):
    query = {"brand_name": {"$regex": brand, "$options": "i"}}
    if model:
        query["model_name"] = {"$regex": model, "$options": "i"}
    return list(collection.find(query))

def get_vehicles_by_price_range(min_price, max_price):
    return list(collection.find({"price": {"$gte": min_price, "$lte": max_price}}))

def get_vehicles_by_fuel(fuel_type):
    return list(collection.find({"fuel_type": {"$regex": fuel_type, "$options": "i"}}))

def get_vehicles_by_type(vehicle_type):
    return list(collection.find({"vehicle_type": {"$regex": vehicle_type, "$options": "i"}}))

def get_available_brands():
    return collection.distinct("brand_name")

def get_available_vehicle_types():
    return collection.distinct("vehicle_type")

def get_vehicles_by_location(location):
    return list(collection.find({"location": {"$regex": location, "$options": "i"}}))

def get_available_locations():
    return [loc for loc in collection.distinct("location") if isinstance(loc, str) and loc.strip()]

feedback_collection = client.auto_sales_bot.feedback

def save_feedback(query, response, predicted_intent, prob, thumbs_up):
    feedback = {
        "query": query,
        "response": response,
        "predicted_intent": predicted_intent,
        "prob": prob,
        "thumbs_up": thumbs_up,
        "checked_by_admin": False,
        "is_retrained": False,
        "timestamp": datetime.utcnow()
    }
    feedback_collection.insert_one(feedback)

def get_feedbacks_for_retraining():
    return list(feedback_collection.find({
        "checked_by_admin": True,
        "is_retrained": False,
        "correct_intent": {"$exists": True}
    }))

def mark_feedback_as_retrained(feedback_ids):
    from bson.objectid import ObjectId
    feedback_collection.update_many(
        {"_id": {"$in": [ObjectId(fb_id) for fb_id in feedback_ids]}},
        {"$set": {"is_retrained": True}}
    )
