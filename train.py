import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils.preprocessing import preprocess, bag_of_words
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def train_model(show_metrics=True):
    # Load intents
    with open("data/intents.json") as f:
        intents = json.load(f)

    all_words = []
    tags = []
    xy = []

    # Extract patterns and tags
    for intent in intents["intents"]:
        tag = intent["tag"]
        tags.append(tag)
        for pattern in intent["patterns"]:
            w = preprocess(pattern)
            all_words.extend(w)
            xy.append((w, tag))

    # Remove duplicates
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    # Create training data
    X_train = [bag_of_words(x, all_words) for (x, y) in xy]
    y_train = [tags.index(y) for (x, y) in xy]

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    class ChatDataset(Dataset):
        def __init__(self):
            self.n_samples = len(X_train)
            self.x_data = X_train
            self.y_data = y_train

        def __getitem__(self, index):
            return self.x_data[index], self.y_data[index]

        def __len__(self):
            return self.n_samples

    class NeuralNet(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(NeuralNet, self).__init__()
            self.layer1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.layer2 = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            out = self.layer1(x)
            out = self.relu(out)
            out = self.layer2(out)
            return out

    dataset = ChatDataset()
    train_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True)

    model = NeuralNet(len(all_words), 16, len(tags))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    for epoch in range(100):
        epoch_loss = 0.0
        for words, labels in train_loader:
            outputs = model(words)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch [{epoch + 1}/100], Loss: {avg_loss:.4f}")

    if show_metrics:
        # Optional: visualize confusion matrix
        y_true, y_pred = [], []
        with torch.no_grad():
            for words, labels in train_loader:
                outputs = model(words)
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.numpy())
                y_pred.extend(predicted.numpy())

        try:
            cm = confusion_matrix(y_true, y_pred)
            labels_used = sorted(set(y_true + y_pred))
            if cm.shape[0] == cm.shape[1] == len(labels_used):
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[tags[i] for i in labels_used])
                disp.plot()
                plt.show()
            else:
                print("Skipping confusion matrix display due to label mismatch.")
        except Exception as e:
            print(f"Error plotting confusion matrix: {e}")

    # Save model
    torch.save({
        "model_state": model.state_dict(),
        "input_size": len(all_words),
        "hidden_size": 16,
        "output_size": len(tags),
        "all_words": all_words,
        "tags": tags
    }, "models/model.pth")


if __name__ == "__main__":
    train_model()