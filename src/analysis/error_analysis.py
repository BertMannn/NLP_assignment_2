import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data.preprocessing import preprocess_ag_news
from src.neural_models.cnn_model import CNNTextClassifier
from src.neural_models.lstm_model import LSTMTextClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_cnn_model(vocab_size: int) -> dict:
    model = CNNTextClassifier(vocab_size=vocab_size, embedding_dim=128, num_classes=4)
    model.load_state_dict(torch.load("best_cnn.pt", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def load_lstm_model(vocab_size: int) -> dict:
    model = LSTMTextClassifier(vocab_size=vocab_size, embedding_dim=128, hidden_size = 256, num_classes=4)
    model.load_state_dict(torch.load("best_lstm.pt", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

@torch.no_grad()
def get_predictions(model: nn.Module, tokens: torch.Tensor, batch_size: int = 64) -> list:
    model.eval().to(DEVICE)
    loader = DataLoader(TensorDataset(tokens), batch_size=batch_size)
    preds = []
    for (batch,) in loader:
        logits = model(batch.to(DEVICE))
        preds.extend(logits.argmax(dim=1).cpu().tolist())
    return preds






def collect_misclassified(model, model_name,data,n=20):

    X_test = torch.tensor(data["test"]["text"], dtype=torch.long)
    y_test = list(data["test"]["label"])
    raw_test = list(data["test"]["raw_text"])

    y_pred = get_predictions(model,X_test)

    print(f"Collecting first 20 misclassified examples for {model_name} :\n")

    count = 0
    for i in range(len(y_test)):
        if y_pred[i] != y_test[i]:
            print("=" * 80)
            print(f"Index: {i}")
            print(f"TRUE: {int(y_test[i])}    PRED: {int(y_pred[i])}")
            print(f"TEXT: {raw_test[i]}")
            count += 1
            if count == n:
                break

def run_error_analysis(n=20):
    datasets, vocab = preprocess_ag_news()
    vocab_size = len(vocab)
    cnn = load_cnn_model(vocab_size)
    lstm = load_lstm_model(vocab_size)
    collect_misclassified(cnn,"CNN",datasets,n=20)
    collect_misclassified(lstm,"LSTM",datasets, n=20)
    

if __name__ == "__main__":
    run_error_analysis(n=20)

