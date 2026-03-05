from datasets import load_dataset

SEED = 42


def load_ag_news() -> dict:
    """Load the AG News dataset from datasets

    Returns:
        dict: The train, dev and test splits of the dataset, with keys "train", "dev", and "test".
    """
    dataset = load_dataset("ag_news")

    train_dev = dataset["train"].train_test_split(test_size=0.1, seed=SEED)

    # Original train/test split is used. Additionally, 10% of all data is taken from the train set as the dev set.
    train_dataset = train_dev["train"]
    dev_dataset = train_dev["test"]
    test_dataset = dataset["test"]

    return {"train": train_dataset, "dev": dev_dataset, "test": test_dataset}
