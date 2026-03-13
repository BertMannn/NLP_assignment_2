from collections import Counter
from data.load_data import load_ag_news
import torch

MAX_LENGTH = 128


def tokenize(text: str):
    """Tokenizer for the neural models"""
    return text.lower().split()


def build_vocab(texts, min_freq: int = 2):
    """Create a vocabulary based on the training texts"""
    counter = Counter()

    for text in texts:
        tokens = tokenize(text)
        counter.update(tokens)

    vocab = {"<PAD>": 0, "<UNK>": 1}

    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)

    return vocab


def encode_text(text, vocab):
    """Convert a text string to a padded list of token ids"""
    tokens = tokenize(text)

    ids = []
    for token in tokens:
        if token in vocab:
            ids.append(vocab[token])
        else:
            ids.append(vocab["<UNK>"])

    # truncate if needed
    ids = ids[:MAX_LENGTH]

    # pad if sequence is shorter than max length
    if len(ids) < MAX_LENGTH:
        padding = [vocab["<PAD>"]] * (MAX_LENGTH - len(ids))
        ids = ids + padding

    return ids


def preprocess_ag_news() -> tuple[dict, dict]:
    """Preprocess and load the AG News dataset for neural models.

    Returns:
        dict: A dictionary containing the processed datasets,
        with keys "train", "dev", and "test".
    """
    datasets = load_ag_news()

    # Build vocabulary from the training split only
    vocab = build_vocab(datasets["train"]["text"])

    # Dataset object does not support direct assignment
    # so we store the processed data in a new dictionary
    processed_datasets = {}

    for dataset in ["train", "dev", "test"]:
        processed_datasets[dataset] = {}

        texts = datasets[dataset]["text"]

        encoded_texts = []
        for text in texts:
            encoded_texts.append(encode_text(text, vocab))

        processed_datasets[dataset]["text"] = torch.tensor(
            encoded_texts, dtype=torch.long
        )
        processed_datasets[dataset]["label"] = torch.tensor(
            datasets[dataset]["label"], dtype=torch.long
        )

        if dataset == "test":
            processed_datasets[dataset]["raw_text"] = texts

    return processed_datasets, vocab
