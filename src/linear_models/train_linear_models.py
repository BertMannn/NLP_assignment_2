from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def train_linear_models(X_train: list, y_train: list) -> dict:
    """Train both a linear SVM and a Logistic Regression model on the data provided

    Args:
        X_train (list): training data
        y_train (list): labels

    Returns:
        dict: return a dictionary with the trained models, with keys "logistic_regression" and "svm"
    """
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)

    svm_model = LinearSVC()
    svm_model.fit(X_train, y_train)

    return {"logistic_regression": lr_model, "svm": svm_model}
