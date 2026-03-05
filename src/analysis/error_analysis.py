from data.preprocessing import preprocess_ag_news
from src.linear_models.train_linear_models import train_linear_models

def collect_misclassified_svm(n=20):
    data = preprocess_ag_news()
    models = train_linear_models(data["train"]["text"], data["train"]["label"])
    svm = models["svm"]

    X_test = data["test"]["text"]
    y_test = list(data["test"]["label"])
    raw_test = list(data["test"]["raw_text"])

    y_pred = svm.predict(X_test)

    print(f"Collecting first 20 misclassified examples from TEST:\n")

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

if __name__ == "__main__":
    collect_misclassified_svm(n=20)
