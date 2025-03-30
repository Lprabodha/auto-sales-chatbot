import os
import json
import random
import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from pymongo import MongoClient
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
import sqlite3

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Load environment variables
load_dotenv()

class AutoSalesChatbot:
    def __init__(self, intents_path, mongo_uri=None):
        self.model = None
        self.intents_path = intents_path
        self.client = MongoClient(mongo_uri or os.getenv("MONGO_URI"))
        self.db = self.client['vehicle_prices']
        self.vehicles_col = self.db['cars']
        self.motorcycles_col = self.db['motorcycles']
        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}
        self.word_frequencies = defaultdict(int)
        self.conversation_context = defaultdict(dict)
        self.vehicle_cache = {}
        self.tfidf_vectorizer = None
        
        # Initialize SQLite database
        self.db_conn = sqlite3.connect('chatbot.db', check_same_thread=False)
        self.setup_database()
        
        print("Initializing Auto Sales Chatbot...")
        self.load_intents()
        self.cache_vehicle_data()
        self.enhance_intents_with_vehicles()

    def setup_database(self):
        """Create database tables if they don't exist"""
        cursor = self.db_conn.cursor()
        
        # Conversation logs table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversation_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            message TEXT,
            response TEXT,
            intent TEXT,
            entities TEXT,
            timestamp DATETIME
        )
        ''')
        
        # Feedback table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            conversation_id INTEGER,
            rating INTEGER,
            comments TEXT,
            timestamp DATETIME,
            FOREIGN KEY(conversation_id) REFERENCES conversation_logs(id)
        )
        ''')
        
        self.db_conn.commit()

    def log_conversation(self, user_id, message, response, intent, entities):
        """Log conversation to SQLite database"""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute('''
            INSERT INTO conversation_logs 
            (user_id, message, response, intent, entities, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                message,
                response,
                intent,
                json.dumps(entities),
                datetime.now().isoformat()
            ))
            self.db_conn.commit()
            return cursor.lastrowid
        except Exception as e:
            print(f"Error logging conversation: {e}")
            return None

    def save_feedback(self, user_id, conversation_id, rating, comments=None):
        """Save user feedback to SQLite database"""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute('''
            INSERT INTO feedback 
            (user_id, conversation_id, rating, comments, timestamp)
            VALUES (?, ?, ?, ?, ?)
            ''', (
                user_id,
                conversation_id,
                rating,
                comments,
                datetime.now().isoformat()
            ))
            self.db_conn.commit()
            return True
        except Exception as e:
            print(f"Error saving feedback: {e}")
            return False

    def cache_vehicle_data(self):
        """Cache all vehicle data from MongoDB for faster access"""
        print("Caching vehicle data...")
        for vehicle in self.vehicles_col.find():
            key = f"{vehicle.get('brand_name', '').lower()}_{vehicle.get('model_name', '').lower().replace(' ', '_')}"
            self.vehicle_cache[key] = vehicle
        for bike in self.motorcycles_col.find():
            key = f"{bike.get('brand_name', '').lower()}_{bike.get('model_name', '').lower().replace(' ', '_')}"
            self.vehicle_cache[key] = bike
        print(f"Cached {len(self.vehicle_cache)} vehicles")

    def load_intents(self):
        """Load and process the intents.json file"""
        print(f"Loading intents from {self.intents_path}...")
        if os.path.exists(self.intents_path):
            with open(self.intents_path, 'r') as f:
                intents_data = json.load(f)
            
            # Initialize TF-IDF vectorizer
            patterns = []
            for intent in intents_data['intents']:
                tag = intent['tag']
                if tag not in self.intents:
                    self.intents.append(tag)
                    self.intents_responses[tag] = {
                        'responses': intent.get('responses', []),
                        'follow_up': intent.get('follow_up', None)
                    }
                patterns.extend(intent['patterns'])
            
            self.tfidf_vectorizer = TfidfVectorizer(
                tokenizer=self.tokenize_and_lemmatize,
                stop_words='english',
                min_df=1
            )
            self.tfidf_vectorizer.fit(patterns)
            
            # Process documents for training
            for intent in intents_data['intents']:
                for pattern in intent['patterns']:
                    words = self.tokenize_and_lemmatize(pattern)
                    self.documents.append((words, intent['tag']))
                    for word in words:
                        self.word_frequencies[word] += 1
            
            self.vocabulary = sorted(set(self.vocabulary))
            print(f"Loaded {len(self.intents)} intents with {len(self.vocabulary)} unique words")
        else:
            print("Warning: Intents file not found")

    def enhance_intents_with_vehicles(self):
        """Enhance training data with patterns from actual vehicles"""
        print("Enhancing intents with vehicle data...")
        for vehicle in self.vehicle_cache.values():
            brand = vehicle.get('brand_name', '').lower()
            model = vehicle.get('model_name', '').lower()
            location = vehicle.get('location', '').lower()
            vehicle_type = vehicle.get('vehicle_type', '').lower()
            
            # Price inquiry patterns
            price_patterns = [
                f"price of {brand} {model}",
                f"how much for {brand} {model}",
                f"{brand} {model} price",
                f"cost of {brand} {model}",
                f"{model} price in {location}"
            ]
            
            # Availability patterns
            availability_patterns = [
                f"do you have {brand} {model}",
                f"is {brand} {model} available",
                f"any {brand} {model} in {location}",
                f"available {brand} {model}"
            ]
            
            # Location patterns
            location_patterns = [
                f"cars in {location}",
                f"vehicles in {location}",
                f"{vehicle_type} in {location}",
                f"what's available in {location}"
            ]
            
            # Recommendation patterns
            recommendation_patterns = [
                f"best {vehicle_type} under {random.randint(10, 50)}00000",
                f"recommend {brand} vehicles",
                f"good {vehicle_type} in {location}",
                f"what {vehicle_type} should I buy"
            ]
            
            # Detail patterns
            detail_patterns = [
                f"specs of {brand} {model}",
                f"details about {brand} {model}",
                f"features of {model}",
                f"tell me about {brand} {model}"
            ]
            
            # Add to training data
            for pattern in (price_patterns + availability_patterns + 
                          location_patterns + recommendation_patterns + 
                          detail_patterns):
                words = self.tokenize_and_lemmatize(pattern)
                if 'price' in pattern or 'cost' in pattern:
                    tag = 'vehicle_price_inquiry'
                elif 'available' in pattern or 'have' in pattern:
                    tag = 'vehicle_availability'
                elif 'in ' in pattern:
                    tag = 'filter_by_location'
                elif 'recommend' in pattern or 'best' in pattern:
                    tag = 'vehicle_recommendation'
                elif 'specs' in pattern or 'details' in pattern:
                    tag = 'vehicle_details'
                else:
                    tag = 'general_inquiry'
                self.documents.append((words, tag))
                self.vocabulary.extend(words)
        
        self.vocabulary = sorted(set(self.vocabulary))
        print(f"Enhanced training data with vehicle patterns. Total samples: {len(self.documents)}")

    def tokenize_and_lemmatize(self, text):
        """Tokenize and lemmatize text while preserving vehicle model numbers"""
        lemmatizer = nltk.WordNetLemmatizer()
        tokens = nltk.word_tokenize(text.lower())
        
        # Special handling for model numbers (like PCX 125)
        words = []
        i = 0
        while i < len(tokens):
            if tokens[i].isdigit() and i > 0 and tokens[i-1].isalpha():
                # Combine model names with their numbers (e.g., "pcx" + "125")
                words[-1] = words[-1] + " " + tokens[i]
            elif tokens[i].isalpha():
                words.append(lemmatizer.lemmatize(tokens[i]))
            i += 1
        
        return [w for w in words if w not in nltk.corpus.stopwords.words('english')]

    def prepare_data(self):
        """Prepare training data with TF-IDF features"""
        print("Preparing training data...")
        bags = []
        labels = []
        
        # Get TF-IDF features
        patterns = [' '.join(words) for words, _ in self.documents]
        tfidf_features = self.tfidf_vectorizer.transform(patterns).toarray()
        
        # Combine with bag-of-words
        for i, (words, tag) in enumerate(self.documents):
            bow = np.zeros(len(self.vocabulary))
            for word in words:
                if word in self.vocabulary:
                    bow[self.vocabulary.index(word)] = 1
            combined = np.concatenate([bow, tfidf_features[i]])
            bags.append(combined)
            labels.append(self.intents.index(tag))
        
        self.X = np.array(bags)
        self.y = np.array(labels)
        print(f"Prepared {len(self.X)} training samples with {self.X.shape[1]} features")

    def create_model(self, input_size):
        """Create the neural network model"""
        return nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, len(self.intents))
        )

    def train_model(self, epochs=200, batch_size=32, lr=0.001):
        """Train the model with enhanced features"""
        print("\nStarting model training...")
        self.prepare_data()
        
        # Initialize model
        input_size = self.X.shape[1]
        self.model = self.create_model(input_size)
        
        # Prepare data loaders
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)
        
        # Handle class imbalance
        class_counts = np.bincount(self.y)
        weights = 1. / class_counts
        samples_weights = weights[self.y]
        sampler = WeightedRandomSampler(
            weights=samples_weights,
            num_samples=len(samples_weights),
            replacement=True
        )
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        best_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == batch_y).sum().item()
            
            avg_loss = total_loss / len(loader)
            accuracy = correct / len(self.X)
            
            # Update learning rate
            scheduler.step(avg_loss)
            
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2%}, LR: {optimizer.param_groups[0]["lr"]:.2e}')
            
            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break
        
        # Load best model
        if os.path.exists('best_model.pth'):
            self.model.load_state_dict(torch.load('best_model.pth'))
        self.model.eval()

    def save_model(self, model_path, metadata_path):
        """Save the trained model and metadata"""
        print(f"\nSaving model to {model_path} and metadata to {metadata_path}...")
        torch.save(self.model.state_dict(), model_path)
        
        metadata = {
            'vocabulary': self.vocabulary,
            'intents': self.intents,
            'intents_responses': self.intents_responses,
            'input_size': self.X.shape[1],
            'output_size': len(self.intents),
            'tfidf_vocab': self.tfidf_vectorizer.vocabulary_
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        print("Model saved successfully")

    def load_model(self, model_path, metadata_path):
        """Load a trained model and metadata"""
        print(f"\nLoading model from {model_path} and metadata from {metadata_path}...")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.vocabulary = metadata['vocabulary']
        self.intents = metadata['intents']
        self.intents_responses = metadata['intents_responses']
        
        # Reinitialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            vocabulary=metadata['tfidf_vocab'],
            tokenizer=self.tokenize_and_lemmatize,
            stop_words='english'
        )
        self.tfidf_vectorizer.fit(["dummy text"])
        
        # Initialize model
        self.model = self.create_model(metadata['input_size'])
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        print(f"Loaded model with {len(self.intents)} intents and {len(self.vocabulary)} vocabulary words")

    def extract_entities(self, text):
        """Enhanced entity extraction for vehicle sales"""
        entities = {}
        if not text or not text.strip():
            return entities
            
        text_lower = text.lower()
        words = self.tokenize_and_lemmatize(text)
        
        # Common Sri Lankan locations
        sri_lanka_locations = [
            'colombo', 'galle', 'kandy', 'negombo', 'matara', 
            'jaffna', 'ratnapura', 'anuradhapura', 'badulla',
            'kurunegala', 'gampaha', 'kalutara', 'trincomalee'
        ]
        
        # Extract location
        for loc in sri_lanka_locations:
            if loc in text_lower:
                entities['location'] = loc
                break
        
        # Extract vehicle type
        vehicle_types = {
            'car': ['car', 'sedan', 'hatchback', 'saloon', 'auto'],
            'motorcycle': ['bike', 'motorcycle', 'scooter', 'moped'],
            'suv': ['suv', 'jeep', '4x4', 'crossover']
        }
        for v_type, terms in vehicle_types.items():
            if any(term in text_lower for term in terms):
                entities['vehicle_type'] = v_type
                break
        
        # Extract price range
        price_terms = ['under', 'below', 'less than', 'over', 'above', 'around', 'budget']
        for i, word in enumerate(words):
            if word in price_terms and i+1 < len(words) and words[i+1].replace(',', '').isdigit():
                entities['price_range'] = {
                    'operator': word,
                    'value': int(words[i+1].replace(',', ''))
                }
        
        # Extract brand and model
        for key in self.vehicle_cache:
            brand, model = key.split('_', 1)
            model = model.replace('_', ' ')
            
            # Check for brand match
            if brand in text_lower:
                entities['brand'] = brand
                
                # Check for model match with more flexibility
                model_words = model.lower().split()
                if all(mw in text_lower for mw in model_words):
                    entities['model'] = model
                elif any(mw in text_lower for mw in model_words):
                    # Partial match - use the part that matches
                    matched = [mw for mw in model_words if mw in text_lower]
                    entities['model'] = ' '.join(matched)
        
        # Extract year
        for word in words:
            if word.isdigit() and len(word) == 4 and 1990 <= int(word) <= datetime.now().year + 1:
                entities['year'] = word
        
        # Extract other specs
        specs = {
            'engine_capacity': ['cc', 'engine', 'capacity'],
            'fuel_type': ['petrol', 'diesel', 'electric', 'hybrid'],
            'condition': ['new', 'used', 'reconditioned'],
            'transmission': ['automatic', 'manual', 'cvt'],
            'color': ['red', 'blue', 'black', 'white', 'silver', 'gray']
        }
        
        for spec, terms in specs.items():
            for term in terms:
                if term in text_lower:
                    entities[spec] = term
                    break
        
        return entities

    def predict_intent(self, text):
        """Predict intent from text"""
        if not text or not text.strip():
            return None, 0.0
            
        words = self.tokenize_and_lemmatize(text)
        
        # Get features
        bow = np.zeros(len(self.vocabulary))
        for word in words:
            if word in self.vocabulary:
                bow[self.vocabulary.index(word)] = 1
        
        tfidf = self.tfidf_vectorizer.transform([' '.join(words)]).toarray()[0]
        features = np.concatenate([bow, tfidf])
        
        # Predict
        with torch.no_grad():
            input_tensor = torch.tensor([features], dtype=torch.float32)
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)
        
        return self.intents[predicted_idx.item()], confidence.item()

    def generate_vehicle_price_response(self, entities, user_id):
        """Generate detailed price response"""
        brand = entities.get('brand', '')
        model = entities.get('model', '')
        location = entities.get('location', '')
        
        if brand and model:
            key = f"{brand}_{model.replace(' ', '_')}"
            vehicle = self.vehicle_cache.get(key)
            
            if vehicle:
                price = f"Rs. {vehicle.get('price', 'N/A'):,}"
                year = vehicle.get('year', 'N/A')
                engine = vehicle.get('engine_capacity', 'N/A')
                condition = vehicle.get('condition', 'N/A').capitalize()
                fuel = vehicle.get('fuel_type', 'N/A').capitalize()
                seller = vehicle.get('seller_name', 'Private seller')
                loc = vehicle.get('location', 'Unknown location')
                phone = vehicle.get('phone_number', '')
                
                response = f"🚗 *{brand.title()} {model.title()}*\n"
                response += f"💰 Price: {price}\n"
                response += f"📅 Year: {year}\n"
                response += f"⚙️ Engine: {engine}\n"
                response += f"🔄 Condition: {condition}\n"
                response += f"⛽ Fuel: {fuel}\n"
                response += f"👤 Seller: {seller}\n"
                response += f"📍 Location: {loc}\n"
                
                if phone:
                    response += f"📞 Contact: {phone}\n"
                
                response += "\nWould you like to contact the seller or see similar vehicles?"
                
                # Update context
                self.conversation_context[user_id]['last_vehicle'] = {
                    'brand': brand,
                    'model': model
                }
                
                return response
        
        # If we couldn't find exact match, show similar vehicles
        return self.generate_recommendation_response(entities, user_id)

    def generate_recommendation_response(self, entities, user_id):
        """Generate vehicle recommendations based on criteria"""
        # Get filters from entities
        price_max = None
        if 'price_range' in entities:
            if entities['price_range']['operator'] in ['under', 'below', 'less than']:
                price_max = entities['price_range']['value']
        
        vehicle_type = entities.get('vehicle_type')
        location = entities.get('location')
        brand = entities.get('brand')
        year = entities.get('year')
        fuel_type = entities.get('fuel_type')
        condition = entities.get('condition')
        
        # Find matching vehicles
        matching_vehicles = []
        for vehicle in self.vehicle_cache.values():
            matches_type = not vehicle_type or vehicle.get('vehicle_type', '').lower() == vehicle_type
            matches_brand = not brand or vehicle.get('brand_name', '').lower() == brand
            matches_location = not location or vehicle.get('location', '').lower() == location
            matches_year = not year or str(vehicle.get('year', '')) == year
            matches_fuel = not fuel_type or vehicle.get('fuel_type', '').lower() == fuel_type
            matches_condition = not condition or vehicle.get('condition', '').lower() == condition
            matches_price = not price_max or vehicle.get('price', float('inf')) <= price_max
            
            if (matches_type and matches_brand and matches_location and 
                matches_year and matches_fuel and matches_condition and matches_price):
                matching_vehicles.append(vehicle)
        
        if not matching_vehicles:
            return ("Sorry, we don't have any vehicles matching your criteria. "
                   "Would you like to try different filters?")
        
        # Sort by price (ascending)
        matching_vehicles.sort(key=lambda x: x.get('price', float('inf')))
        
        # Build response
        if price_max:
            response = f"Here are the best vehicles under Rs. {price_max:,}:\n\n"
        else:
            response = "Here are some vehicle recommendations:\n\n"
        
        for i, vehicle in enumerate(matching_vehicles[:5]):  # Show top 5
            brand = vehicle.get('brand_name', 'Unknown').title()
            model = vehicle.get('model_name', 'Unknown').title()
            price = f"Rs. {vehicle.get('price', 'N/A'):,}"
            year = vehicle.get('year', 'N/A')
            engine = vehicle.get('engine_capacity', 'N/A')
            condition = vehicle.get('condition', 'N/A').capitalize()
            
            response += f"{i+1}. 🚗 *{brand} {model}*\n"
            response += f"   💰 {price} | 📅 {year} | ⚙️ {engine}\n"
            response += f"   🔄 {condition} | 📍 {vehicle.get('location', 'Unknown')}\n\n"
        
        response += "Would you like more details about any of these vehicles?"
        
        # Store recommendations in context
        self.conversation_context[user_id]['recommendations'] = matching_vehicles[:5]
        
        return response

    def generate_availability_response(self, entities):
        """Generate response for availability queries"""
        brand = entities.get('brand')
        vehicle_type = entities.get('vehicle_type', '').lower()
        location = entities.get('location')
        
        # Find matching vehicles
        matching_vehicles = []
        for v in self.vehicle_cache.values():
            matches_brand = not brand or v.get('brand_name', '').lower() == brand
            matches_type = not vehicle_type or v.get('vehicle_type', '').lower() == vehicle_type
            matches_location = not location or v.get('location', '').lower() == location
            
            if matches_brand and matches_type and matches_location:
                matching_vehicles.append(v)
        
        if not matching_vehicles:
            return (f"Sorry, we don't currently have any {' '.join(filter(None, [brand, vehicle_type])).title()} "
                   f"{'in ' + location if location else ''}in stock. Would you like information on similar vehicles?")
        
        # Group by model
        model_groups = {}
        for vehicle in matching_vehicles:
            model = vehicle.get('model_name', 'Unknown model')
            if model not in model_groups:
                model_groups[model] = []
            model_groups[model].append(vehicle)
        
        # Build response
        response = "Here are the available options:\n\n"
        
        for model, vehicles in model_groups.items():
            brand_name = vehicles[0].get('brand_name', 'Unknown brand')
            min_price = min(v.get('price', float('inf')) for v in vehicles)
            max_price = max(v.get('price', 0) for v in vehicles)
            locations = set(v.get('location') for v in vehicles)
            
            response += f"🚗 *{brand_name} {model}*\n"
            response += f"💰 Price range: Rs. {min_price:,} - Rs. {max_price:,}\n"
            response += f"📍 Available in {len(locations)} locations\n\n"
        
        response += "Would you like details about any specific model?"
        return response

    def generate_location_filter_response(self, entities):
        """Generate response for location-based queries"""
        location = entities.get('location', '').title()
        vehicle_type = entities.get('vehicle_type', 'car')  # Default to car
        
        if not location:
            return "Which location are you interested in?"
        
        # Find matching vehicles
        matching_vehicles = []
        for v in self.vehicle_cache.values():
            if v.get('location', '').lower() == location.lower():
                if not vehicle_type or v.get('vehicle_type', '').lower() == vehicle_type:
                    matching_vehicles.append(v)
        
        if not matching_vehicles:
            return (f"Sorry, we don't have any {vehicle_type}s "
                   f"available in {location}. Try another location?")

        # Group by brand and model
        brand_model_groups = {}
        for vehicle in matching_vehicles:
            brand = vehicle.get('brand_name', 'Unknown')
            model = vehicle.get('model_name', 'Unknown')
            key = f"{brand}_{model}"
            
            if key not in brand_model_groups:
                brand_model_groups[key] = []
            brand_model_groups[key].append(vehicle)
        
        # Build response
        response = f"Available {vehicle_type}s in {location}:\n\n"
        
        for key, vehicles in brand_model_groups.items():
            brand, model = key.split('_', 1)
            min_price = min(v.get('price', float('inf')) for v in vehicles)
            max_price = max(v.get('price', 0) for v in vehicles)
            
            response += f"🚗 *{brand} {model.replace('_', ' ')}*\n"
            response += f"💰 Price Range: Rs. {min_price:,} - Rs. {max_price:,}\n"
            response += f"📅 Years: {', '.join(sorted({str(v.get('year', '')).strip() for v in vehicles if v.get('year')}))}\n\n"
        
        response += "Would you like details about any specific vehicle?"
        return response

    def generate_seller_contact_response(self, entities):
        """Generate response with seller contact information"""
        brand = entities.get('brand', '')
        model = entities.get('model', '')
        key = f"{brand}_{model.replace(' ', '_')}"
        
        vehicle = self.vehicle_cache.get(key)
        if not vehicle:
            return f"Sorry, we don't have contact information for {brand} {model}."
        
        seller = vehicle.get('seller_name', 'Private seller')
        phone = vehicle.get('phone_number', '')
        location = vehicle.get('location', 'unknown location')
        
        response = f"Seller information for {brand.title()} {model.title()}:\n"
        response += f"👤 Seller: {seller}\n"
        response += f"📍 Location: {location}\n"
        
        if phone:
            response += f"📞 Phone: {phone}\n"
        else:
            response += "Contact details not provided. Please check the listing.\n"
        
        response += "\nWould you like to schedule a test drive?"
        return response

    def generate_vehicle_details_response(self, entities):
        """Generate detailed vehicle specifications"""
        brand = entities.get('brand', '')
        model = entities.get('model', '')
        key = f"{brand}_{model.replace(' ', '_')}"
        
        vehicle = self.vehicle_cache.get(key)
        if not vehicle:
            return f"Sorry, I couldn't find details for {brand} {model}."
        
        response = f"🚗 *{brand.title()} {model.title()} Specifications*\n\n"
        response += f"📅 Year: {vehicle.get('year', 'N/A')}\n"
        response += f"⚙️ Engine: {vehicle.get('engine_capacity', 'N/A')}cc\n"
        response += f"⛽ Fuel Type: {vehicle.get('fuel_type', 'N/A').title()}\n"
        response += f"🔄 Condition: {vehicle.get('condition', 'N/A').title()}\n"
        response += f"🚪 Transmission: {vehicle.get('transmission', 'N/A').title()}\n"
        response += f"🛣️ Mileage: {vehicle.get('mileage', 'N/A')} km\n"
        response += f"🎨 Color: {vehicle.get('color', 'N/A').title()}\n\n"
        response += "Would you like the seller's contact information or price details?"
        
        return response

    def handle_follow_up(self, user_id, intent, entities):
        """Handle follow-up questions based on context"""
        if intent == 'vehicle_price_inquiry':
            return self.generate_vehicle_price_response(entities, user_id)
        elif intent == 'vehicle_availability':
            return self.generate_availability_response(entities)
        elif intent == 'vehicle_recommendation':
            return self.generate_recommendation_response(entities, user_id)
        elif intent == 'contact_seller':
            return self.generate_seller_contact_response(entities)
        elif intent == 'filter_by_location':
            return self.generate_location_filter_response(entities)
        elif intent == 'vehicle_details':
            return self.generate_vehicle_details_response(entities)
        else:
            return "How can I help you with that?"

    def handle_low_confidence(self, message, entities):
        """Handle low confidence predictions"""
        text_lower = message.lower()
        
        # Check for location terms
        if 'location' in entities:
            if 'vehicle_type' in entities:
                return self.generate_location_filter_response(entities)
            else:
                return f"What type of vehicle are you looking for in {entities['location'].title()}? (cars, motorcycles, SUVs)"
        
        # Check for vehicle terms
        vehicle_terms = ['car', 'bike', 'suv', 'motorcycle', 'vehicle']
        if any(term in text_lower for term in vehicle_terms):
            return ("I understand you're asking about vehicles. "
                   "Could you be more specific? (e.g., 'Honda PCX price' or 'Cars in Colombo')")
        
        # Check for price terms
        price_terms = ['price', 'cost', 'how much', 'budget']
        if any(term in text_lower for term in price_terms):
            return ("I can help with price information. "
                   "Please specify the vehicle make and model or your budget range")
        
        # Check for recommendation terms
        recommendation_terms = ['recommend', 'best', 'good', 'suggest']
        if any(term in text_lower for term in recommendation_terms):
            return ("I can recommend vehicles based on your needs. "
                   "Please specify your budget and preferred vehicle type")
        
        return random.choice(self.intents_responses['fallback']['responses'])

    def collect_feedback(self, user_id, conversation_id=None):
        """Collect feedback from user"""
        print("\nPlease rate your experience (1-5):")
        print("1: Very dissatisfied")
        print("5: Very satisfied")
        
        while True:
            rating = input("Your rating (1-5): ")
            if rating.isdigit() and 1 <= int(rating) <= 5:
                break
            print("Please enter a number between 1 and 5")
        
        comments = input("Any comments? (Press enter to skip): ")
        
        if self.save_feedback(user_id, conversation_id, int(rating), comments):
            print("Thank you for your feedback!")
        else:
            print("Sorry, we couldn't save your feedback.")

    def process_message(self, message, user_id="default_user"):
        """Main method to process user messages"""
        if not message or not message.strip():
            return "Please enter a valid message."
        
        print(f"\nProcessing message from {user_id}: '{message}'")
        
        # Handle simple greetings
        if message.lower() in ['hi', 'hello', 'hey', 'good morning', 'good afternoon']:
            response = random.choice(self.intents_responses['greeting']['responses'])
            self.log_conversation(user_id, message, response, 'greeting', {})
            return response
        
        # Handle goodbye
        if message.lower() in ['bye', 'goodbye', 'see you', 'quit', 'exit']:
            response = random.choice(self.intents_responses['goodbye']['responses'])
            self.log_conversation(user_id, message, response, 'goodbye', {})
            self.conversation_context.pop(user_id, None)
            return response
        
        # Extract entities first
        entities = self.extract_entities(message)
        print(f"Extracted entities: {entities}")
        
        # Log the conversation (initially without response)
        log_id = self.log_conversation(user_id, message, "", "", entities)
        
        # Check for conversation continuation
        ctx = self.conversation_context.get(user_id, {})
        if ctx.get('awaiting_follow_up'):
            intent = ctx.get('current_intent')
            response = self.handle_follow_up(user_id, intent, entities)
            
            # Clear follow-up flag if we got what we needed
            if not any(k in entities for k in ctx.get('expected_entities', [])):
                self.conversation_context[user_id]['awaiting_follow_up'] = False
                self.conversation_context[user_id]['current_intent'] = None
                self.conversation_context[user_id]['expected_entities'] = []
            
            # Update the log with the response
            if log_id:
                cursor = self.db_conn.cursor()
                cursor.execute('''
                UPDATE conversation_logs 
                SET response = ?, intent = ?
                WHERE id = ?
                ''', (response, intent, log_id))
                self.db_conn.commit()
            
            self.update_conversation_history(user_id, message, response, intent)
            return response
        
        # Special handling for common patterns before intent prediction
        if any(term in message.lower() for term in ['price of', 'how much', 'cost of']):
            response = self.handle_vehicle_price_inquiry(entities, user_id)
            self.update_log_and_history(log_id, user_id, message, response, 'vehicle_price_inquiry')
            return response
        
        if any(term in message.lower() for term in ['do you have', 'available', 'in stock']):
            response = self.handle_vehicle_availability(entities, user_id)
            self.update_log_and_history(log_id, user_id, message, response, 'vehicle_availability')
            return response
        
        if any(term in message.lower() for term in ['contact seller', 'phone number', 'how to contact']):
            response = self.handle_contact_seller(entities, user_id)
            self.update_log_and_history(log_id, user_id, message, response, 'contact_seller')
            return response
        
        if any(term in message.lower() for term in ['in ', 'near me', 'available in']):
            response = self.handle_location_filter(entities, user_id)
            self.update_log_and_history(log_id, user_id, message, response, 'filter_by_location')
            return response
        
        if any(term in message.lower() for term in ['specs', 'details', 'features']):
            response = self.handle_vehicle_details(entities, user_id)
            self.update_log_and_history(log_id, user_id, message, response, 'vehicle_details')
            return response
        
        # Predict intent
        intent, confidence = self.predict_intent(message)
        print(f"Predicted intent: {intent} ({confidence:.2%} confidence)")
        
        # Handle low confidence
        if confidence < 0.65:
            response = self.handle_low_confidence(message, entities)
            self.update_log_and_history(log_id, user_id, message, response, 'low_confidence')
            return response
        
        # Generate appropriate response
        if intent == 'vehicle_price_inquiry':
            response = self.generate_vehicle_price_response(entities, user_id)
        elif intent == 'vehicle_availability':
            response = self.generate_availability_response(entities)
        elif intent == 'vehicle_recommendation':
            response = self.generate_recommendation_response(entities, user_id)
        elif intent == 'contact_seller':
            response = self.generate_seller_contact_response(entities)
        elif intent == 'filter_by_location':
            response = self.generate_location_filter_response(entities)
        elif intent == 'vehicle_details':
            response = self.generate_vehicle_details_response(entities)
        elif intent == 'greeting':
            response = random.choice(self.intents_responses['greeting']['responses'])
        elif intent == 'goodbye':
            response = random.choice(self.intents_responses['goodbye']['responses'])
            self.conversation_context.pop(user_id, None)
        else:
            response = random.choice(self.intents_responses['fallback']['responses'])
        
        # Update the log with the response
        self.update_log_and_history(log_id, user_id, message, response, intent)
        
        return response

    def update_log_and_history(self, log_id, user_id, message, response, intent):
        """Update both the log and conversation history"""
        if log_id:
            cursor = self.db_conn.cursor()
            cursor.execute('''
            UPDATE conversation_logs 
            SET response = ?, intent = ?
            WHERE id = ?
            ''', (response, intent, log_id))
            self.db_conn.commit()
        
        self.update_conversation_history(user_id, message, response, intent)

    def handle_vehicle_price_inquiry(self, entities, user_id):
        """Handle vehicle price inquiries"""
        if 'brand' in entities and 'model' in entities:
            return self.generate_vehicle_price_response(entities, user_id)
        
        # Store context for follow-up
        self.conversation_context[user_id] = {
            'awaiting_follow_up': True,
            'current_intent': 'vehicle_price_inquiry',
            'expected_entities': ['brand', 'model'],
            'history': self.conversation_context.get(user_id, {}).get('history', [])
        }
        return self.intents_responses['vehicle_price_inquiry']['follow_up']['prompt']

    def handle_vehicle_availability(self, entities, user_id):
        """Handle vehicle availability checks"""
        if 'brand' in entities or 'vehicle_type' in entities:
            return self.generate_availability_response(entities)
        
        self.conversation_context[user_id] = {
            'awaiting_follow_up': True,
            'current_intent': 'vehicle_availability',
            'expected_entities': ['brand', 'vehicle_type'],
            'history': self.conversation_context.get(user_id, {}).get('history', [])
        }
        return self.intents_responses['vehicle_availability']['follow_up']['prompt']

    def handle_contact_seller(self, entities, user_id):
        """Handle seller contact requests"""
        if 'brand' in entities and 'model' in entities:
            return self.generate_seller_contact_response(entities)
        
        self.conversation_context[user_id] = {
            'awaiting_follow_up': True,
            'current_intent': 'contact_seller',
            'expected_entities': ['brand', 'model'],
            'history': self.conversation_context.get(user_id, {}).get('history', [])
        }
        return self.intents_responses['contact_seller']['follow_up']['prompt']

    def handle_location_filter(self, entities, user_id):
        """Handle location-based filtering"""
        if 'location' in entities:
            return self.generate_location_filter_response(entities)
        
        self.conversation_context[user_id] = {
            'awaiting_follow_up': True,
            'current_intent': 'filter_by_location',
            'expected_entities': ['location'],
            'history': self.conversation_context.get(user_id, {}).get('history', [])
        }
        return self.intents_responses['filter_by_location']['follow_up']['prompt']

    def handle_vehicle_details(self, entities, user_id):
        """Handle vehicle detail requests"""
        if 'brand' in entities and 'model' in entities:
            return self.generate_vehicle_details_response(entities)
        
        self.conversation_context[user_id] = {
            'awaiting_follow_up': True,
            'current_intent': 'vehicle_details',
            'expected_entities': ['brand', 'model'],
            'history': self.conversation_context.get(user_id, {}).get('history', [])
        }
        return self.intents_responses['vehicle_details']['follow_up']['prompt']

    def update_conversation_history(self, user_id, user_message, bot_response, intent):
        """Update the conversation history"""
        if user_id not in self.conversation_context:
            self.conversation_context[user_id] = {'history': []}
        
        # Initialize history if it doesn't exist
        if 'history' not in self.conversation_context[user_id]:
            self.conversation_context[user_id]['history'] = []
        
        self.conversation_context[user_id]['history'].append({
            'user': user_message,
            'bot': bot_response,
            'intent': intent,
            'timestamp': datetime.now()
        })
        
        # Keep only last 5 messages
        self.conversation_context[user_id]['history'] = self.conversation_context[user_id]['history'][-5:]

    def close(self):
        """Clean up resources"""
        print("Closing chatbot...")
        self.db_conn.close()
        self.client.close()

if __name__ == '__main__':
    chatbot = AutoSalesChatbot('intents.json', mongo_uri=os.getenv("MONGO_URI"))
    
    if not os.path.exists('vehicle_chatbot_model.pth'):
        print("Training new model...")
        chatbot.train_model(epochs=200)
        chatbot.save_model('vehicle_chatbot_model.pth', 'vehicle_metadata.json')
    else:
        print("Loading existing model...")
        chatbot.load_model('vehicle_chatbot_model.pth', 'vehicle_metadata.json')
    
    print("\nAuto Sales Chatbot (type 'quit' to exit)")
    
    user_id = "user123"
    last_conversation_id = None
    
    while True:
        try:
            message = input("You: ")
            if message.lower() in ['quit', 'exit', 'bye']:
                # Ask for feedback before exiting
                if last_conversation_id:
                    chatbot.collect_feedback(user_id, last_conversation_id)
                break
            
            response = chatbot.process_message(message, user_id)
            print("Bot:", response)
            
            # Get the last conversation ID for feedback
            cursor = chatbot.db_conn.cursor()
            cursor.execute('SELECT id FROM conversation_logs ORDER BY id DESC LIMIT 1')
            result = cursor.fetchone()
            last_conversation_id = result[0] if result else None
        
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    chatbot.close()