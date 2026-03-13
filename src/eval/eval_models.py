import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
import matplotlib.pyplot as plt

from data.preprocessing import preprocess_ag_news
from src.neural_models.cnn_model import CNNTextClassifier
from src.neural_models.lstm_model import LSTMTextClassifier


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_nn_models(vocab_size: int) -> dict:
    cnn = CNNTextClassifier(vocab_size=vocab_size, embedding_dim=128, num_classes=4)
    cnn.load_state_dict(torch.load("best_cnn.pt", map_location=DEVICE))

    cnn_ablation = CNNTextClassifier(vocab_size=vocab_size, embedding_dim=128, num_classes=4)
    cnn_ablation.load_state_dict(torch.load("best_ccn_0_dropout.pt", map_location=DEVICE))

    lstm = LSTMTextClassifier(
        vocab_size=vocab_size, embedding_dim=128, hidden_size=256, num_classes=4
    )
    lstm.load_state_dict(torch.load("best_lstm.pt", map_location=DEVICE))
    
    lstm_ablation =LSTMTextClassifier(
        vocab_size=vocab_size, embedding_dim=128, hidden_size=256, num_classes=4
    )
    lstm_ablation.load_state_dict(torch.load("best_lstm_0_dropout.pt", map_location=DEVICE))

    return {"CNN": cnn, "CNN (dropout 0.0)":cnn_ablation, "LSTM": lstm, "LSTM (dropout 0.0)": lstm_ablation}


@torch.no_grad()
def get_predictions(
    model: nn.Module, tokens: torch.Tensor, batch_size: int = 64
) -> list:
    model.eval().to(DEVICE)
    loader = DataLoader(TensorDataset(tokens), batch_size=batch_size)
    preds = []
    for (batch,) in loader:
        logits = model(batch.to(DEVICE))
        preds.extend(logits.argmax(dim=1).cpu().tolist())
    return preds


def evaluate_nn_models() -> None:
    """Evaluate CNN and LSTM models on the AG News dataset.
    Reports Macro-F1, Accuracy, and a Confusion Matrix for both test and dev splits.
    """
    datasets, vocab = preprocess_ag_news()
    models = load_nn_models(vocab_size=len(vocab))

    for split in ("test", "dev"):
        print(f"Evaluating neural models on the {split} set")
        text = datasets[split]["text"]
        labels = datasets[split]["label"]

        for model_name, model in models.items():
            y_pred = get_predictions(model, text)
            print(f"Model: {model_name}")
            print(f"Accuracy: {accuracy_score(labels, y_pred):.4f}")
            print(f"Macro-F1: {f1_score(labels, y_pred, average='macro'):.4f}")
            ConfusionMatrixDisplay.from_predictions(labels, y_pred)
            plt.title(f"Confusion Matrix for {model_name} on {split} set")
            plt.show()
            print("-" * 50)


def main():
    evaluate_nn_models()


if __name__ == "__main__":
    main()
