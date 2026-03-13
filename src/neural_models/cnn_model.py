import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNTextClassifier(nn.Module):
    """Simple 1D CNN for text classification"""

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        num_classes,
        kernel_sizes=[3, 4, 5],
        num_filters=100,
        dropout=0.3,
    ):
        super(CNNTextClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Create multiple Conv1D layers with different kernel sizes
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embedding_dim, out_channels=num_filters, kernel_size=k
                )
                for k in kernel_sizes
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        # x: (batch_size, seq_len)
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        x = x.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_len) for Conv1d

        conv_outs = []
        for conv in self.convs:
            c = F.relu(conv(x))  # (batch_size, num_filters, seq_len - k + 1)
            c = F.max_pool1d(c, kernel_size=c.shape[2])  # (batch_size, num_filters, 1)
            c = c.squeeze(2)  # (batch_size, num_filters)
            conv_outs.append(c)

        x = torch.cat(conv_outs, dim=1)  # (batch_size, num_filters * len(kernel_sizes))
        x = self.dropout(x)
        logits = self.fc(x)  # (batch_size, num_classes)
        return logits
