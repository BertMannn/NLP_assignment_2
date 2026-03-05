from sklearn.feature_extraction.text import TfidfVectorizer
from data.load_data import load_ag_news


def preprocess_ag_news() -> dict:
    """Preprocess  and load the AG News dataset using TF-IDF vectorization

    Returns:
        dict: A dictionary containing the preprocessed datasets, with keys "train", "dev", and "test".
    """
    datasets = load_ag_news()

    vectorizer = TfidfVectorizer()
    vectorizer.fit(datasets["train"]["text"])

    # Datasets is of type Dataset and does not support direct assignment
    # so we create a new dictionary to store the vectorized datasets
    vectorized_datasets = {}

    for dataset in ["train", "dev", "test"]:
        vectorized_datasets[dataset] = {}
        vectorized_datasets[dataset]["text"] = vectorizer.transform(
            datasets[dataset]["text"]
        )
        vectorized_datasets[dataset]["label"] = datasets[dataset]["label"]
        if dataset == "test":
            vectorized_datasets[dataset]["raw_text"] = datasets[dataset]["text"]
    return vectorized_datasets
