import torch
import torch.nn as nn


class BiLSTMModel(nn.Module):
    def __init__(
        self,
        input_size=63,
        hidden_size=64,
        num_layers=2,
        num_classes=10,
        dropout=0.3
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):

        # Input:
        # (batch, 30, 21, 3)

        batch_size = x.size(0)

        # Convert:
        # (batch, 30, 21, 3)
        # into:
        # (batch, 30, 63)

        x = x.reshape(
            batch_size,
            x.size(1),
            -1
        )

        # BiLSTM
        output, _ = self.lstm(x)

        # Use the final time step
        final_output = output[:, -1, :]

        # Classification
        logits = self.classifier(final_output)

        return logits