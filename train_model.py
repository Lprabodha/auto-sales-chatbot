import torch
import torch.nn as nn
import json
import pickle
import os
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.corpus import stopwords, wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def get_synonyms(word):
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            if lemma.name() != word:
                synonyms.add(lemma.name().replace('_', ' '))
    return synonyms

def preprocess(text):
    tokens = word_tokenize(text.lower())
    clean_words = [lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words]
    augmented = set(clean_words)
    for w in clean_words:
        augmented.update(get_synonyms(w))
    return ' '.join(augmented)

# Load dataset
dataset_path = "data/intents.json"
if os.path.exists("data/mongo_generated_training_data.pkl"):
    with open("data/mongo_generated_training_data.pkl", "rb") as f:
        dataset = pickle.load(f)
    texts = [preprocess(d["text"]) for d in dataset]
    intents = [d["intent"] for d in dataset]
else:
    with open(dataset_path, "r") as f:
        data = json.load(f)
    dataset = [{"text": pattern, "intent": intent["tag"]} for intent in data["intents"] for pattern in intent.get("patterns", [])]
    texts = [preprocess(d["text"]) for d in dataset]
    intents = [d["intent"] for d in dataset]

vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 3))
X = vectorizer.fit_transform(texts)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(intents)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

class ChatModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

model = ChatModel(X.shape[1], 128, len(set(y)))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

X_tensor = torch.FloatTensor(X_train.toarray())
y_tensor = torch.LongTensor(y_train)

# Training
for epoch in range(400):
    model.train()
    optimizer.zero_grad()
    output = model(X_tensor)
    loss = criterion(output, y_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch} - Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    test_tensor = torch.FloatTensor(X_test.toarray())
    preds = model(test_tensor)
    predicted = torch.argmax(preds, dim=1)
    acc = accuracy_score(y_test, predicted)
    print(f"\n✅ Test Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, predicted, target_names=label_encoder.classes_, zero_division=0))

    cm = confusion_matrix(y_test, predicted)
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    os.makedirs("model", exist_ok=True)
    plt.savefig("model/confusion_matrix.png")
    plt.close()

# Save model and tools
os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/intent_model.pt")
with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open("model/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("\n✅ Model training complete. Saved in /model.")
