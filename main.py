import os
import json
import random
import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pymongo import MongoClient
from dotenv import load_dotenv
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client['riyasewana']
stop_words = set(stopwords.words('english'))

class ChatbotModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ChatbotModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Chatbot Assistant
class ChatbotAssistant:
    def __init__(self, intents_path, function_mappings=None):
        self.model = None
        self.intents_path = intents_path
        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}
        self.function_mappings = function_mappings
        self.X = None
        self.y = None

    def tokenize_and_lemmatize(self, text):
        lemmatizer = nltk.WordNetLemmatizer()
        words = nltk.word_tokenize(text.lower())
        words = [lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stop_words]
        return words

    def bag_of_words(self, words):
        return [1 if word in words else 0 for word in self.vocabulary]

    def parse_intents(self):
        if os.path.exists(self.intents_path):
            with open(self.intents_path, 'r') as f:
                intents_data = json.load(f)
            for intent in intents_data['intents']:
                if intent['tag'] not in self.intents:
                    self.intents.append(intent['tag'])
                    self.intents_responses[intent['tag']] = intent['responses']
                for pattern in intent['patterns']:
                    pattern_words = self.tokenize_and_lemmatize(pattern)
                    self.vocabulary.extend(pattern_words)
                    self.documents.append((pattern_words, intent['tag']))
            self.vocabulary = sorted(set(self.vocabulary))

    def prepare_data(self):
        bags, indices = [], []
        for words, tag in self.documents:
            bags.append(self.bag_of_words(words))
            indices.append(self.intents.index(tag))
        self.X = np.array(bags)
        self.y = np.array(indices)

    def train_model(self, batch_size=8, lr=0.001, epochs=100):
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.model = ChatbotModel(self.X.shape[1], len(self.intents))
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            running_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch + 1}: Loss: {running_loss / len(loader):.4f}")

    def save_model(self, model_path, dimensions_path):
        torch.save(self.model.state_dict(), model_path)
        with open(dimensions_path, 'w') as f:
            json.dump({'input_size': self.X.shape[1], 'output_size': len(self.intents)}, f)

    def load_model(self, model_path, dimensions_path):
        with open(dimensions_path, 'r') as f:
            dimensions = json.load(f)
        self.model = ChatbotModel(dimensions['input_size'], dimensions['output_size'])
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def process_message(self, input_message, threshold=0.6):
        words = self.tokenize_and_lemmatize(input_message)
        bag = self.bag_of_words(words)
        bag_tensor = torch.tensor([bag], dtype=torch.float32)
        with torch.no_grad():
            predictions = self.model(bag_tensor)
            probs = F.softmax(predictions, dim=1)
            confidence, predicted_class_index = torch.max(probs, dim=1)
        if confidence.item() < threshold:
            return "🤖 I'm not sure I understand. Could you rephrase?"
        predicted_intent = self.intents[predicted_class_index.item()]
        if self.function_mappings and predicted_intent in self.function_mappings:
            self.function_mappings[predicted_intent]()
        return random.choice(self.intents_responses.get(predicted_intent, ["I'm not sure about that."]))

    def save_vocab(self, path):
        with open(path, 'w') as f:
            json.dump({'vocabulary': self.vocabulary, 'intents': self.intents}, f)

    def load_vocab(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        self.vocabulary = data['vocabulary']
        self.intents = data['intents']


def fetch_vehicle_info_from_db(brand):
    vehicle = db['cars'].find_one({"brand_name": {"$regex": brand, "$options": "i"}}, sort=[("scrape_date", -1)])
    if vehicle:
        print(f"Latest {brand} vehicle: {vehicle['model_name']} ({vehicle['model_year']}) - {vehicle['price']}")
    else:
        print(f"No recent listings found for {brand}.")

def get_subscrption_cancel():
    reasons = ['customer_service', 'low_quality', 'missing_features', 'switched_service', 'too_expensive', 'unused']
    print("Possible reasons for cancelation:", random.sample(reasons, 3))

if __name__ == '__main__':
    assistant = ChatbotAssistant('intents.json', function_mappings={
        'subscription_cancel': get_subscrption_cancel,
        'vehicle_price_inquiry': lambda: fetch_vehicle_info_from_db("Toyota")
    })
    assistant.parse_intents()
    assistant.load_vocab('metadata.json')
    assistant.load_model('chatbot_model.pth', 'dimensions.json')

    while True:
        message = input('Enter your message: ')
        if message.lower() == '/quit':
            break
        print(assistant.process_message(message))