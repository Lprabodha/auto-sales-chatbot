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
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

load_dotenv()

engine = create_engine(os.getenv("DB_URI"))
Base = declarative_base()
Session = sessionmaker(bind=engine)
session = Session()

class Feedback(Base):
    __tablename__ = 'chatbot_feedback'
    id = Column(Integer, primary_key=True)
    user_id = Column(String(50))
    user_message = Column(Text)
    bot_response = Column(Text)
    feedback = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(engine)

class VehicleChatbot:
    def __init__(self, intents_path, mongo_uri=None):
        self.model = None
        self.intents_path = intents_path
        self.client = MongoClient(mongo_uri or os.getenv("MONGO_URI"))
        self.db = self.client['riyasewana']
        self.vehicles_col = self.db['vehicles'] 
        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}
        self.word_frequencies = defaultdict(int)
        self.context = {}
        self.unknown_questions = []
        self.load_intents()

    def load_intents(self):
        if os.path.exists(self.intents_path):
            with open(self.intents_path, 'r') as f:
                intents_data = json.load(f)
            
            for intent in intents_data['intents']:
                tag = intent['tag']
                if tag not in self.intents:
                    self.intents.append(tag)
                    self.intents_responses[tag] = intent['responses']
                
                for pattern in intent['patterns']:
                    words = self.tokenize_and_lemmatize(pattern)
                    self.vocabulary.extend(words)
                    self.documents.append((words, tag))

                    for word in words:
                        self.word_frequencies[word] += 1
            self.vocabulary = sorted(set(self.vocabulary))

    def tokenize_and_lemmatize(self, text):
        lemmatizer = nltk.WordNetLemmatizer()
        tokens = nltk.word_tokenize(text.lower())
        words = [
            lemmatizer.lemmatize(token) 
            for token in tokens 
            if token.isalpha() and token not in nltk.corpus.stopwords.words('english')
        ]
        return words

    def bag_of_words(self, words):
        bag = [0] * len(self.vocabulary)
        for word in words:
            if word in self.vocabulary:
                index = self.vocabulary.index(word)
                bag[index] = 1
        return bag

    def tfidf_vector(self, words):
        vector = np.zeros(len(self.vocabulary))
        total_words = sum(self.word_frequencies.values())
        for word in words:
            if word in self.vocabulary:
                index = self.vocabulary.index(word)
                tf = words.count(word) / len(words)
                idf = np.log(total_words / (1 + self.word_frequencies[word]))
                vector[index] = tf * idf
        return vector

    def prepare_data(self):
        bags = []
        labels = []
        for words, tag in self.documents:
            bow = self.bag_of_words(words)
            tfidf = self.tfidf_vector(words)
            combined = np.concatenate([bow, tfidf])
            bags.append(combined)
            labels.append(self.intents.index(tag))
        self.X = np.array(bags)
        self.y = np.array(labels)

    def train_model(self, epochs=100, batch_size=8, lr=0.001):
        self.prepare_data()
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        input_size = self.X.shape[1]
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, len(self.intents))
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}')

    def save_model(self, model_path, metadata_path):
        torch.save(self.model.state_dict(), model_path)
        metadata = {
            'vocabulary': self.vocabulary,
            'intents': self.intents,
            'intents_responses': self.intents_responses,
            'input_size': self.X.shape[1],
            'output_size': len(self.intents)
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

    def load_model(self, model_path, metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        self.vocabulary = metadata['vocabulary']
        self.intents = metadata['intents']
        self.intents_responses = metadata['intents_responses']
        self.model = nn.Sequential(
            nn.Linear(metadata['input_size'], 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, metadata['output_size'])
        )
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def get_dynamic_response(self, intent, entities):
        return random.choice(self.intents_responses.get(intent, ["I'm not sure about that."]))

    def extract_entities(self, text):
        return {}

    def process_message(self, message, user_id=None, threshold=0.7):
        if user_id and user_id in self.context:
            context_intent = self.context[user_id].get('intent')
            if context_intent == 'awaiting_feedback':
                self.handle_feedback(message, user_id)
                return "Thank you for your feedback! How else can I help you?"
        words = self.tokenize_and_lemmatize(message)
        entities = self.extract_entities(message)
        bow = self.bag_of_words(words)
        tfidf = self.tfidf_vector(words)
        input_vector = np.concatenate([bow, tfidf])
        with torch.no_grad():
            input_tensor = torch.tensor([input_vector], dtype=torch.float32)
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)
        confidence = confidence.item()
        predicted_idx = predicted_idx.item()
        predicted_intent = self.intents[predicted_idx]
        if confidence < threshold:
            return "I'm not sure I understand. Could you rephrase or provide more details?"
        response = self.get_dynamic_response(predicted_intent, entities)
        if predicted_intent == 'feedback':
            if user_id:
                self.context[user_id] = {
                    'intent': 'awaiting_feedback',
                    'last_message': message,
                    'bot_response': response
                }
            return "Sure, please tell me what you'd like to provide feedback about."
        return response

    def handle_feedback(self, message, user_id):
        last = self.context[user_id]
        del self.context[user_id]
        fb = Feedback(
            user_id=user_id,
            user_message=last['last_message'],
            bot_response=last['bot_response'],
            feedback=message
        )
        session.add(fb)
        session.commit()
        print(f"[Saved Feedback] From {user_id}: {message}")

    def close(self):
        self.client.close()

if __name__ == '__main__':
    chatbot = VehicleChatbot('intents.json', mongo_uri=os.getenv("MONGO_URI"))
    if not os.path.exists('vehicle_chatbot_model.pth'):
        print("Training new model...")
        chatbot.train_model(epochs=50)
        chatbot.save_model('vehicle_chatbot_model.pth', 'vehicle_metadata.json')
    else:
        print("Loading existing model...")
        chatbot.load_model('vehicle_chatbot_model.pth', 'vehicle_metadata.json')
    print("Vehicle Sales Chatbot (type '/quit' to exit)")
    user_id = "user123"
    while True:
        message = input("You: ")
        if message.lower() == '/quit':
            break
        response = chatbot.process_message(message, user_id)
        print("Bot:", response)
    chatbot.close()