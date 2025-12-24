#cnn.py
import torch
import torch.nn as nn

class MarketCNN(nn.Module):
    def __init__(self, in_channels, emb_dim=64):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Linear(64, emb_dim)

    def forward(self, x):
        x = self.cnn(x)
        x = x.squeeze(-1)
        return self.fc(x)
