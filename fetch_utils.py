from pymongo import MongoClient
import re
from fuzzywuzzy import fuzz

client = MongoClient("mongodb+srv://lprabodha1998:SfXnuKZIecrv3TUJ@cluster0.e2m4j.mongodb.net/")
db = client["vehicle_prices"]

def is_bike_related(query):
    return any(w in query.lower() for w in ["bike", "motorbike", "motorcycle", "scooter"])

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

def get_dynamic_keywords(field):
    values = set()
    for coll in ["cars", "motorcycles"]:
        values.update(v.lower() for v in db[coll].distinct(field) if isinstance(v, str))
    return list(values)

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
                    if attr == "engine_capacity" and is_bike_related(query):
                        continue
                    if attr == "transmission" and not is_bike_related(query):
                        continue
                    value = v.get(attr) or v.get("engine") if attr == "engine_capacity" else v.get(attr)
                    if value:
                        detail.append(f"{attr.replace('_', ' ').title()}: {value}")
                detail.append(f"More Info: {v.get('ad_url', '')}")
                results.append("\n".join(detail))
    return "\n---\n".join(results[:5]) if results else "No matching vehicles found."

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

def fetch_by_model_year(query):
    match = re.search(r"\b(20\d{2})\b", query)
    if not match:
        return "Please specify a valid model year (e.g., 2023, 2024)."
    year = match.group(1)
    results = []
    for coll in ["cars", "motorcycles"]:
        for v in db[coll].find({"model_year": year}):
            vehicle_name = v.get("vehicle_name", "Unnamed Vehicle")
            location = v.get("location", "Unknown Location")
            price = v.get("price", 0)
            url = v.get("ad_url", "")
            results.append(f"{vehicle_name} – Rs. {price:,} in {location}\n🌐 {url}")
    return "\n\n".join(results[:5]) if results else f"No vehicles found from {year}."

def fetch_seller_info(query):
    keyword = query.lower()
    results = []
    for coll in ["cars", "motorcycles"]:
        for v in db[coll].find():
            brand = v.get("brand_name", "").lower()
            model = v.get("model_name", "").lower()
            if brand in keyword or model in keyword:
                seller = v.get("seller_name", "Not listed")
                phone = v.get("phone_number", "N/A")
                url = v.get("ad_url", "")
                results.append(f"Seller: {seller} | Contact: {phone}\n🌐 {url}")
    return "\n\n".join(results[:5]) if results else "No seller info found for that vehicle."

def fetch_by_model_year(query):
    match = re.search(r"\b(20\\d{2})\b", query)
    if not match:
        return "Please specify a valid model year (e.g., 2023, 2024)."
    year = match.group(1)
    results = []
    for coll in ["cars", "motorcycles"]:
        for v in db[coll].find({"model_year": year}):
            name = v.get("vehicle_name", "Unnamed")
            loc = v.get("location", "Unknown")
            price = v.get("price", 0)
            url = v.get("ad_url", "")
            results.append(f"{name} – Rs. {price:,} in {loc}\n🌐 {url}")
    return "\n\n".join(results[:5]) if results else f"No vehicles found from {year}."
