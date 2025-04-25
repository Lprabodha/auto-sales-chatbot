import pickle
from pymongo import MongoClient
from sklearn.utils import shuffle

client = MongoClient("mongodb+srv://lprabodha1998:SfXnuKZIecrv3TUJ@cluster0.e2m4j.mongodb.net/")
db = client["vehicle_prices"]

collections = ["cars", "motorcycles"]
samples = []

for coll in collections:
    for doc in db[coll].find({}, {
        "brand_name": 1,
        "model_name": 1,
        "model_year": 1,
        "fuel_type": 1,
        "transmission": 1,
        "engine": 1,
        "engine_capacity": 1,
        "location": 1,
    }):
        brand = (doc.get("brand_name") or "").strip()
        model = (doc.get("model_name") or "").strip()
        year = str(doc.get("model_year") or "").strip()
        location = (doc.get("location") or "").strip()
        transmission = (doc.get("transmission") or "").strip()
        engine = (doc.get("engine") or "").strip()
        engine_capacity = (doc.get("engine_capacity") or "").strip()
        fuel = (doc.get("fuel_type") or "").strip()

        if brand and model:
            samples.append({"text": f"{brand} {model}", "intent": "vehicle_detail_request"})
            samples.append({"text": f"Tell me about {brand} {model}", "intent": "vehicle_detail_request"})
            samples.append({"text": f"Price of {brand} {model}", "intent": "vehicle_price_query"})
            samples.append({"text": f"Who is selling {brand} {model}?", "intent": "seller_name_query"})

        if year and brand and model:
            samples.append({"text": f"{year} {brand} {model} vehicles", "intent": "vehicle_model_year_query"})

        if fuel and coll == "cars":
            samples.append({"text": f"{fuel} {brand} cars", "intent": "vehicle_detail_request"})

        if transmission and coll == "motorcycles":
            samples.append({"text": f"{transmission} {brand} bikes", "intent": "vehicle_detail_request"})

        if engine_capacity and coll == "cars":
            samples.append({"text": f"{engine_capacity}cc {brand} vehicles", "intent": "vehicle_detail_request"})

        if engine and coll == "motorcycles":
            samples.append({"text": f"{engine}cc {brand} bikes", "intent": "vehicle_detail_request"})

        if location and brand:
            samples.append({"text": f"{brand} vehicles in {location}", "intent": "vehicle_location"})

unique_samples = {f"{item['text']}::{item['intent']}": item for item in samples}
final_samples = shuffle(list(unique_samples.values()), random_state=42)

output_path = "data/mongo_generated_training_data.pkl"
with open(output_path, "wb") as f:
    pickle.dump(final_samples, f)

print(f"✅ Mongo-based training data saved to {output_path}")
print(f"✅ Total training samples generated: {len(final_samples)}")
