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

def fuzzy_match(user_text: str, db_text: str, threshold: int = 80) -> bool:

    if not user_text or not db_text:
        return False
    score = fuzz.partial_ratio(user_text.lower(), db_text.lower())
    return score >= threshold

def fuzzy_match_location(query_location, db_location):
    return fuzz.partial_ratio(query_location.lower(), db_location.lower()) > 80

def get_dynamic_keywords(field):
    values = set()
    for coll in ["cars", "motorcycles"]:
        values.update(v.lower() for v in db[coll].distinct(field) if isinstance(v, str))
    return list(values)

def fetch_selected_details(query):
    lowered = query.lower()
    attrs = extract_attributes(query)
    results = []

    if "fuel" in lowered or "efficiency" in lowered or "economy" in lowered:
        vehicles = db["cars"].find({"fuel_type": {"$regex": "hybrid|electric|petrol", "$options": "i"}})
        return format_vehicle_list(vehicles, "Fuel Efficient Options 🚗💨")

    if "low cost" in lowered or "cheap" in lowered or "budget" in lowered:
        vehicles = db["cars"].find().sort("price", 1).limit(5)
        return format_vehicle_list(vehicles, "Low Cost Vehicles 🏷️")
    
    cc_match = re.search(r"(\d{3,4})\s*cc", lowered)
    if cc_match:
        requested_cc = int(cc_match.group(1))
        return fetch_by_engine_capacity(requested_cc)

    for coll in ["cars", "motorcycles"]:
        for v in db[coll].find():
            brand = v.get("brand_name", "").lower()
            model = v.get("model_name", "").lower()
            location = v.get("location", "").lower()

            if fuzzy_match(lowered, brand) or fuzzy_match(lowered, model) or fuzzy_match_location(lowered, location):
                detail = []
                for attr in (attrs or ["price", "location"]):
                    value = v.get(attr) or v.get("engine") if attr == "engine_capacity" else v.get(attr)
                    if value:
                        detail.append(f"{attr.replace('_', ' ').title()}: {value}")
                detail.append(f"More Info: {v.get('ad_url', '')}")
                results.append("\n".join(detail))

    if results:
        return "\n---\n".join(results[:5])
    else:
        random_vehicles = db["cars"].aggregate([{"$sample": {"size": 3}}])
        return format_vehicle_list(random_vehicles, "Popular Vehicles You Might Like 🚗")

def format_vehicle_list(vehicles, title):
    return f"**{title}**\n\n" + "\n---\n".join([
        f"{v.get('vehicle_name', 'Unknown')} – Rs. {v.get('price', 'N/A'):,} in {v.get('location', 'Unknown')}\n🌐 {v.get('ad_url', '')}"
        for v in vehicles
    ])

def fetch_by_engine_capacity(requested_cc, tolerance=100):

    lower_bound = requested_cc - tolerance
    upper_bound = requested_cc + tolerance
    results = []

    for coll in ["cars", "motorcycles"]:
        for v in db[coll].find():
            engine = v.get("engine_capacity") or v.get("engine")
            if engine:
                try:
                    engine_value = int(engine)
                    if lower_bound <= engine_value <= upper_bound:
                        vehicle_name = v.get("vehicle_name", "Unknown Vehicle")
                        price = v.get("price", "N/A")
                        location = v.get("location", "Unknown")
                        fuel = v.get("fuel_type", "Unknown")
                        url = v.get("ad_url", "")
                        results.append(
                            f"**{vehicle_name}**\nPrice: Rs. {price:,}\nEngine: {engine_value}cc\nFuel: {fuel}\nLocation: {location}\n🌐 [More Info]({url})"
                        )
                except ValueError:
                    continue  

    if results:
        return "**Vehicles around {}cc:**\n\n".format(requested_cc) + "\n\n---\n\n".join(results[:5])
    else:
        return f"❓ Sorry, no vehicles found around {requested_cc}cc. Try another search?"


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
                results.append(f"👤 Seller: {seller}\n📞 Phone: {phone}\n🌐 [View Ad]({url})")
    if results:
        return "\n\n".join(results[:5])
    else:
        return "❓ Please specify the vehicle name (like Honda Vezel, Suzuki Swift) so I can find the seller for you!"


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
