import torch
import json
from difflib import get_close_matches
from utils.preprocessing import (
    preprocess,
    bag_of_words,
    extract_price_range,
    safe_int,
    lemmatize_words
)
from mongo_service import *

# Load model
data = torch.load("models/model.pth")

class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.layer1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return out

model = NeuralNet(data["input_size"], data["hidden_size"], data["output_size"])
model.load_state_dict(data["model_state"])
model.eval()

all_words = data["all_words"]
tags = data["tags"]

with open("data/intents.json") as f:
    intents = json.load(f)

def predict_intent(sentence):
    tokens = preprocess(sentence)
    bow = bag_of_words(tokens, all_words)
    bow = torch.tensor(bow, dtype=torch.float32)
    output = model(bow)
    _, predicted = torch.max(output, dim=0)
    prob = torch.softmax(output, dim=0)[predicted.item()]
    return tags[predicted.item()], prob.item()

def format_vehicle_results(vehicles, brand=None, model=None, price_filter=False):
    if not vehicles:
        if price_filter:
            return f"Sorry, no {brand or 'vehicles'} found in that price range."
        return f"Sorry, no listings found for {brand} {model or ''}."

    formatted = []
    for v in vehicles:
        model_name = v.get("model_name", v.get("vehicle_name", "Unknown Model"))
        year = v.get("model_year", "")
        price = safe_int(v.get("price", 0))
        brand_actual = v.get("brand_name", brand or "Unknown")
        formatted.append(f"{brand_actual} {model_name} {year} - {price:,} LKR")

    return f"Here are some {brand or 'vehicles'} available:\n" + "\n".join(formatted[:5])

def get_cheapest_vehicle(brand=None):
    query = {} if not brand else {"brand_name": {"$regex": brand, "$options": "i"}}
    vehicles = list(collection.find(query))
    valid_vehicles = [v for v in vehicles if safe_int(v.get("price", 0)) > 0]
    if not valid_vehicles:
        return None
    sorted_vehicles = sorted(valid_vehicles, key=lambda v: safe_int(v.get("price", 0)))
    return sorted_vehicles[0]

def get_response(intent_tag, sentence):
    sentence_lower = sentence.lower()
    brands = get_available_brands()
    vehicle_types = get_available_vehicle_types()

    for intent in intents["intents"]:
        if intent["tag"] == intent_tag:

            if intent_tag == "ask_brand_model":
                brand = next((b for b in brands if b.lower() in sentence_lower), None)
                if not brand:
                    return "Can you please mention the brand name?"

                vehicles = get_vehicles_by_brand_model(brand)
                all_models = list(set(v["model_name"].lower() for v in vehicles if "model_name" in v))

                stop_words = {
                    "show", "me", "details", "about", "do", "you", "have", "any", "looking",
                    "for", "the", "a", "an", "is", "there", "car", "vehicle", "model",
                    "what", "are", "available"
                }

                raw_keywords = [
                    word for word in sentence_lower.split()
                    if word not in stop_words and word not in brand.lower() and len(word) > 2
                ]
                cleaned_keywords = lemmatize_words(raw_keywords)

                matched_keywords = []
                for kw in cleaned_keywords:
                    match = get_close_matches(kw, all_models, n=1, cutoff=0.7)
                    if match:
                        matched_keywords.append(match[0])

                filtered = []
                for v in vehicles:
                    name_combo = (v.get("model_name", "") + " " + v.get("vehicle_name", "")).lower().replace("-", " ")
                    if all(kw in name_combo for kw in matched_keywords):
                        filtered.append(v)

                if cleaned_keywords:
                    if filtered:
                        display_model = " ".join(matched_keywords).title()
                        return format_vehicle_results(filtered, brand + f" {display_model}")
                    else:
                        readable_kw = " ".join(raw_keywords)
                        return f"Sorry, no listings found for {brand} {readable_kw}."
                else:
                    return format_vehicle_results(vehicles, brand)

            elif intent_tag == "ask_price_range":
                # Normalize plural vehicle types
                vehicle_type_map = {
                    "cars": "car", "vans": "van", "suvs": "suv", "sedans": "sedan",
                    "trucks": "truck", "hatchbacks": "hatchback", "crossovers": "crossover", "wagons": "wagon"
                }
                sentence_lower = ' '.join([vehicle_type_map.get(word, word) for word in sentence_lower.split()])

                price_range = extract_price_range(sentence_lower)
                if not price_range:
                    return "Can you please specify the price range again?"

                min_price, max_price = price_range

                matched_brand = next((b for b in brands if b.lower() in sentence_lower), None)
                matched_type = next((t for t in vehicle_types if t.lower() in sentence_lower), None)

                if matched_brand:
                    vehicles = get_vehicles_by_brand_model(matched_brand)
                    filtered = [v for v in vehicles if min_price <= safe_int(v.get("price", 0)) <= max_price]
                    return format_vehicle_results(filtered, matched_brand, price_filter=True)

                elif matched_type:
                    vehicles = get_vehicles_by_type(matched_type)
                    filtered = [v for v in vehicles if min_price <= safe_int(v.get("price", 0)) <= max_price]
                    return format_vehicle_results(filtered, matched_type.upper(), price_filter=True)

                else:
                    vehicles = get_vehicles_by_price_range(min_price, max_price)
                    if not vehicles:
                        return "Sorry, no vehicles found in that price range."
                    formatted = [
                        f"{v['brand_name']} {v['model_name']} {v.get('model_year', '')} - {safe_int(v.get('price', 0)):,} LKR"
                        for v in vehicles[:5]
                    ]
                    return "Here are some vehicles within your budget:\n" + "\n".join(formatted)

            elif intent_tag == "ask_cheapest":
                brand = next((b for b in brands if b.lower() in sentence_lower), None)
                cheapest = get_cheapest_vehicle(brand)
                if not cheapest:
                    return "Sorry, no vehicles found."
                return f"The cheapest vehicle{' from ' + brand if brand else ''} is:\n{cheapest['brand_name']} {cheapest['model_name']} {cheapest.get('model_year', '')} - {safe_int(cheapest.get('price')):,} LKR"

            elif intent_tag == "ask_by_fuel":
                fuel_types = ["diesel", "petrol", "hybrid", "electric"]
                for fuel in fuel_types:
                    if fuel in sentence_lower:
                        vehicles = get_vehicles_by_fuel(fuel)
                        if vehicles:
                            return f"These are the vehicles with {fuel} engines:\n" + \
                                   "\n".join(f"{v['vehicle_name']} - {safe_int(v.get('price', 0)):,} LKR" for v in vehicles[:5])
                        else:
                            return f"Sorry, no {fuel} vehicles found."

            elif intent_tag == "ask_by_vehicle_type":
                for vtype in vehicle_types:
                    if vtype.lower() in sentence_lower:
                        vehicles = get_vehicles_by_type(vtype)
                        if vehicles:
                            return f"Here are some {vtype.upper()}s available:\n" + \
                                   "\n".join(f"{v['vehicle_name']} - {safe_int(v.get('price', 0)):,} LKR" for v in vehicles[:5])
                        else:
                            return f"Sorry, no {vtype.upper()} vehicles found."

            elif intent_tag == "ask_leasing":
                return "Please contact the vehicle owner directly for leasing details."

            return intent["responses"][0]

    return "Sorry, I didn’t understand that."

def format_response_with_suggestions(vehicles, brand=None, model=None, price_filter=False):
    """
    Format results for API: returns (response_text, suggestions_list)
    """
    if not vehicles:
        if price_filter:
            return f"Sorry, no {brand or 'vehicles'} found in that price range.", []
        return f"Sorry, no listings found for {brand} {model or ''}.", []

    suggestions = []
    for i, v in enumerate(vehicles[:5]):
        model_name = v.get("model_name", v.get("vehicle_name", "Unknown Model"))
        year = v.get("model_year", "")
        price = safe_int(v.get("price", 0))
        mileage = safe_int(v.get("mileage", 0))
        vehicle_name = v.get("vehicle_name", "")
        suggestions.append({
            "id": i + 1,
            "model_name": model_name,
            "vehicle_name": vehicle_name,
            "year": year,
            "price": price,
            "mileage": mileage
        })

    # Construct user-friendly message
    model_hint = f"{model}" if model else ""
    if brand:
        message = f"Here are some {brand} {model_hint} available:".strip()
    else:
        message = "Here are some vehicles within your budget:"

    return message, suggestions


def get_response_with_suggestions(intent_tag, sentence):
    """
    API-friendly wrapper of get_response() that includes structured suggestions.
    """
    sentence_lower = sentence.lower()
    brands = get_available_brands()
    vehicle_types = get_available_vehicle_types()

    for intent in intents["intents"]:
        if intent["tag"] == intent_tag:

            if intent_tag == "ask_brand_model":
                brand = next((b for b in brands if b.lower() in sentence_lower), None)
                if not brand:
                    return "Can you please mention the brand name?", []

                vehicles = get_vehicles_by_brand_model(brand)
                all_models = list(set(v["model_name"].lower() for v in vehicles if "model_name" in v))

                stop_words = {
                    "show", "me", "details", "about", "do", "you", "have", "any", "looking",
                    "for", "the", "a", "an", "is", "there", "car", "vehicle", "model",
                    "what", "are", "available", "under", "above", "between", "suggest"
                }

                raw_keywords = [
                    word for word in sentence_lower.split()
                    if word not in stop_words and word not in brand.lower() and len(word) > 2
                ]
                cleaned_keywords = lemmatize_words(raw_keywords)

                matched_keywords = []
                for kw in cleaned_keywords:
                    match = get_close_matches(kw, all_models, n=1, cutoff=0.7)
                    if match:
                        matched_keywords.append(match[0])

                filtered = []
                for v in vehicles:
                    name_combo = (v.get("model_name", "") + " " + v.get("vehicle_name", "")).lower().replace("-", " ")
                    if all(kw in name_combo for kw in matched_keywords):
                        filtered.append(v)

                if cleaned_keywords:
                    if filtered:
                        display_model = " ".join(matched_keywords).title()
                        return format_response_with_suggestions(filtered, brand + f" {display_model}")
                    else:
                        readable_kw = " ".join(raw_keywords)
                        return f"Sorry, no listings found for {brand} {readable_kw}.", []
                else:
                    return format_response_with_suggestions(vehicles, brand)

            elif intent_tag == "ask_price_range":
                vehicle_type_map = {
                    "cars": "car", "vans": "van", "suvs": "suv", "sedans": "sedan",
                    "trucks": "truck", "hatchbacks": "hatchback", "crossovers": "crossover", "wagons": "wagon"
                }
                sentence_lower = ' '.join([vehicle_type_map.get(word, word) for word in sentence_lower.split()])

                price_range = extract_price_range(sentence_lower)
                if not price_range:
                    return "Can you please specify the price range again?", []

                min_price, max_price = price_range
                matched_brand = next((b for b in brands if b.lower() in sentence_lower), None)
                matched_type = next((t for t in vehicle_types if t.lower() in sentence_lower), None)

                if matched_brand:
                    vehicles = get_vehicles_by_brand_model(matched_brand)
                    filtered = [v for v in vehicles if min_price <= safe_int(v.get("price", 0)) <= max_price]
                    return format_response_with_suggestions(filtered, matched_brand, price_filter=True)

                elif matched_type:
                    vehicles = get_vehicles_by_type(matched_type)
                    filtered = [v for v in vehicles if min_price <= safe_int(v.get("price", 0)) <= max_price]
                    return format_response_with_suggestions(filtered, matched_type.upper(), price_filter=True)

                else:
                    vehicles = get_vehicles_by_price_range(min_price, max_price)
                    return format_response_with_suggestions(vehicles, "vehicles", price_filter=True)

            elif intent_tag == "ask_leasing":
                return "Please contact the vehicle owner directly for leasing details.", []

            return intent["responses"][0], []

    return "Sorry, I didn’t understand that.", []

def chat():
    print("\n🚗 AutoBot is running. Type 'quit' to exit.")
    prev_intent = None
    while True:
        sentence = input("You: ")
        if sentence.lower() == "quit":
            print("Bot: Goodbye!")
            break

        intent, prob = predict_intent(sentence)

        if prob < 0.7 and prev_intent:
            sentence = f"{prev_intent.replace('_', ' ')} {sentence}"
            intent, prob = predict_intent(sentence)

        if prob > 0.7:
            response = get_response(intent, sentence)
        else:
            response = get_response("fallback", sentence)

        print("Bot:", response)
        prev_intent = intent

if __name__ == "__main__":
    chat()