import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score

from src.neural_models.cnn_model import CNNTextClassifier
from src.neural_models.lstm_model import LSTMTextClassifier
from data.preprocessing import preprocess_ag_news
import itertools


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for texts, labels in loader:
        texts, labels = texts.to(DEVICE), labels.to(DEVICE)
        logits = model(texts)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, macro_f1


def train(
    model,
    train_loader,
    dev_loader,
    epochs=10,
    lr=1e-3,
    clip_grad=5.0,
    patience=2,
    model_path="best.pt",
):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "dev_loss": [],
        "dev_acc": [],
        "dev_f1": [],
    }

    best_dev_f1 = -1.0
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for texts, labels in train_loader:
            texts, labels = texts.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(texts)
            loss = criterion(logits, labels)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

            total_loss += loss.item() * labels.size(0)

        train_loss = total_loss / len(train_loader.dataset)
        dev_loss, dev_acc, dev_f1 = evaluate(model, dev_loader)

        history["train_loss"].append(train_loss)
        history["dev_loss"].append(dev_loss)
        history["dev_acc"].append(dev_acc)
        history["dev_f1"].append(dev_f1)

        print(
            f"Epoch {epoch}/{epochs}  "
            f"train_loss: {train_loss:.4f}  "
            f"dev_loss: {dev_loss:.4f}  "
            f"dev_acc: {dev_acc:.4f}  "
            f"dev_f1: {dev_f1:.4f}"
        )

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            bad_epochs = 0
            torch.save(model.state_dict(), model_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping triggered.")
                break

    return history


def tune_hyperparameters(model, train_loader, dev_loader, param_grid, epochs=10, patience=2):
    best_params, best_f1, best_history = None, -1.0, None
    keys, values = zip(*param_grid.items())
 
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        history = train(model, train_loader, dev_loader, epochs=epochs, patience=patience, **params)
        f1 = max(history["dev_f1"])
        print(f"{params} -> macro F1: {f1:.4f}")
        if f1 > best_f1:
            best_f1, best_params, best_history = f1, params, history
 
    print(f"\nBest macro F1: {best_f1:.4f} | Best params: {best_params}")
    return best_history

def plot_learning_curves(history, model_name):
    """"Plot the learning curves"""
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train loss")
    plt.plot(epochs, history["dev_loss"], label="Dev loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{model_name.lower()}_loss_curve.png")
    plt.show()

    plt.figure()
    plt.plot(epochs, history["dev_acc"], label="Dev accuracy")
    plt.plot(epochs, history["dev_f1"], label="Dev macro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title(f"{model_name} Dev Performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{model_name.lower()}_dev_curve.png")
    plt.show()


if __name__ == "__main__":
    datasets, vocab = preprocess_ag_news()

    def make_loader(split, shuffle=False):
        texts = torch.tensor(datasets[split]["text"], dtype=torch.long)
        labels = torch.tensor(datasets[split]["label"], dtype=torch.long)
        return DataLoader(TensorDataset(texts, labels), batch_size=64, shuffle=shuffle)

    train_loader = make_loader("train", shuffle=True)
    dev_loader = make_loader("dev", shuffle=False)

    cnn = CNNTextClassifier(vocab_size=len(vocab), embedding_dim=128, num_classes=4)
    lstm = LSTMTextClassifier(
        vocab_size=len(vocab), embedding_dim=128, hidden_size=256, num_classes=4
    )
    
    tune_grid = {"lr": [1e-2, 1e-3, 3e-4], "clip_grad": [1.0, 5.0]}
    # cnn_history = tune_hyperparameters(cnn, train_loader, dev_loader, param_grid=tune_grid, model_path="best_ccn.pt")
    # lstm_history = tune_hyperparameters(lstm, train_loader, dev_loader, param_grid=tune_grid, model_path="best_lstm.pt")

    
    # Since both hyperparameters are the same, I put them as defaults in train.py
    cnn_history = train(cnn, train_loader, dev_loader, model_path="best_ccn.pt")
    lstm_history = train(lstm, train_loader, dev_loader, model_path="best_lstm.pt")  
    
    plot_learning_curves(cnn_history, "CNN")
    plot_learning_curves(lstm_history, "LSTM")

    with open("cnn_history.json", "w") as f:
        json.dump(cnn_history, f, indent=2)

    with open("lstm_history.json", "w") as f:
        json.dump(lstm_history, f, indent=2)
