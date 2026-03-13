import torch
from torch.utils.data import Dataset


class AGNewsDataset(Dataset):
    """PyTorch Dataset for AG News using preprocessed token IDs"""

    def __init__(self, texts, labels):
        """
        texts: list of lists of token IDs
        labels: list of integers
        """
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # convert to torch tensors
        x = torch.tensor(self.texts[idx], dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
