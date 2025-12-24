#encoder.py
import torch
import torch.nn as nn

class MarketEncoder(nn.Module):
    def __init__(self, cnn, lstm, state_dim=128):
        super().__init__()

        self.cnn = cnn
        self.lstm = lstm

        self.fc = nn.Sequential(
            nn.Linear(64 + 64, state_dim),
            nn.ReLU()
        )

    def forward(self, x):
        cnn_out = self.cnn(x)
        lstm_out = self.lstm(x.permute(0, 2, 1))
        combined = torch.cat([cnn_out, lstm_out], dim=1)
        return self.fc(combined)
