import torch
import torch.nn as nn

class MarketLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]
