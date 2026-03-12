import torch
import torch.nn as nn
from torch.optim import Adam

from src.neural_models.cnn_model import CNNTextClassifier
from src.neural_models.lstm_model import LSTMTextClassifier

from torch.utils.data import DataLoader, TensorDataset
from data.preprocessing import preprocess_ag_news


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_loader, epochs=10, lr=1e-3, clip_grad=5.0):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for texts, labels in train_loader:
            texts, labels = texts.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            loss = criterion(model(texts), labels)
            loss.backward()

            # For LSTM
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

            total_loss += loss.item() * labels.size(0)

        print(f"Epoch {epoch}/{epochs}  loss: {total_loss / len(train_loader.dataset):.4f}")

    return model


if __name__ == "__main__":
    datasets, vocab = preprocess_ag_news()

    def make_loader(split, shuffle=False):
        texts  = torch.tensor(datasets[split]["text"], dtype=torch.long)
        labels = torch.tensor(datasets[split]["label"], dtype=torch.long)
        return DataLoader(TensorDataset(texts, labels), batch_size=64, shuffle=shuffle)

    train_loader = make_loader("train", shuffle=True)

    cnn  = CNNTextClassifier(vocab_size=len(vocab), embedding_dim=128, num_classes=4)
    lstm = LSTMTextClassifier(vocab_size=len(vocab), embedding_dim=128, hidden_size=256, num_classes=4)

    train(cnn,  train_loader, epochs=10)
    train(lstm, train_loader, epochs=10)

    torch.save(cnn.state_dict(),  "best_cnn.pt")
    torch.save(lstm.state_dict(), "best_lstm.pt")