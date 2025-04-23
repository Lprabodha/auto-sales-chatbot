import torch
import torch.nn as nn
import pickle
import json
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

if os.path.exists("data/preprocessed_training_data.pkl"):
    with open("data/preprocessed_training_data.pkl", "rb") as f:
        dataset = pickle.load(f)
    print("✅ Loaded training data from preprocessed_training_data.pkl")
else:
    with open("data/intents.json", "r") as f:
        data = json.load(f)
    dataset = [{"text": pattern, "intent": intent["tag"]} for intent in data["intents"] for pattern in intent["patterns"]]
    print("✅ Loaded training data from intents.json")

texts = [preprocess(item["text"]) for item in dataset]
intents = [item["intent"] for item in dataset]

vectorizer = TfidfVectorizer(max_features=2500, ngram_range=(1, 2), sublinear_tf=True)
X = vectorizer.fit_transform(texts)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(intents)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

class ChatModel(nn.Module):
    def __init__(self, input_size, hidden, output):
        super(ChatModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(hidden, output)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

model = ChatModel(X.shape[1], 64, len(set(y)))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

X_tensor = torch.FloatTensor(X_train.toarray())
y_tensor = torch.LongTensor(y_train)

epochs = 300
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}/{epochs} - Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    test_tensor = torch.FloatTensor(X_test.toarray())
    preds = model(test_tensor)
    predicted = torch.argmax(preds, dim=1)
    acc = accuracy_score(y_test, predicted)
    print(f"\n✅ Test Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:\n")
    print(classification_report(
        y_test,
        predicted,
        labels=label_encoder.transform(label_encoder.classes_),
        target_names=label_encoder.classes_,
        zero_division=0
    ))

    cm = confusion_matrix(y_test, predicted, labels=label_encoder.transform(label_encoder.classes_))
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    os.makedirs("model", exist_ok=True)
    plt.savefig("model/confusion_matrix.png")
    plt.close()

torch.save(model.state_dict(), "model/intent_model.pt")
with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open("model/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("\n✅ Model training completed and saved. Confusion matrix saved to model/confusion_matrix.png")
