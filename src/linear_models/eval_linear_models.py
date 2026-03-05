from src.linear_models.train_linear_models import train_linear_models
from data.preprocessing import preprocess_ag_news
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
import matplotlib.pyplot as plt


def evaluate_linear_models() -> None:
    """Evaluate both Linear model, this is done by loading the preprocessed ag news dataset
    We than evaluate the model based on Macro-F1 and Accuracy score, plus a confusion matrix
    """
    datasets = preprocess_ag_news()
    models = train_linear_models(datasets["train"]["text"], datasets["train"]["label"])

    print("Evaluating linear models on the test set...")
    for model_name, model in models.items():
        y_pred = model.predict(datasets["test"]["text"])
        print(f"Model: {model_name}")
        print(f"Accuracy: {accuracy_score(datasets['test']['label'], y_pred)}")
        print(
            f"Macro-F1: {f1_score(datasets['test']['label'], y_pred, average='macro')}"
        )
        ConfusionMatrixDisplay.from_predictions(datasets["test"]["label"], y_pred)
        plt.title(f"Confusion Matrix for {model_name} on test set")
        plt.show()
        print("-" * 50)

    print("Evaluating linear models on the dev set...")
    for model_name, model in models.items():
        y_pred = model.predict(datasets["dev"]["text"])
        print(f"Model: {model_name}")
        print(f"Accuracy: {accuracy_score(datasets['dev']['label'], y_pred)}")
        print(
            f"Macro-F1: {f1_score(datasets['dev']['label'], y_pred, average='macro')}"
        )
        ConfusionMatrixDisplay.from_predictions(datasets["dev"]["label"], y_pred)
        plt.title(f"Confusion Matrix for {model_name} on dev set")
        plt.show()
        print("-" * 50)


def main():
    evaluate_linear_models()


if __name__ == "__main__":
    main()
