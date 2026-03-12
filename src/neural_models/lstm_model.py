import torch.nn as nn


class LSTMTextClassifier(nn.Module):
    """LSTM-based text classifier for AG News"""

    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, dropout=0.3):
        super(LSTMTextClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch_size, seq_len)

        x = self.embedding(x)  
        # (batch_size, seq_len, embedding_dim)

        lstm_out, (hidden, cell) = self.lstm(x)

        # hidden: (1, batch_size, hidden_size)
        hidden = hidden.squeeze(0)

        x = self.dropout(hidden)

        logits = self.fc(x)

        return logits