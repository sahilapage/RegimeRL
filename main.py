import torch
from datasets.market_dataset import MarketDataset
from models.cnn import MarketCNN
from models.lstm import MarketLSTM
from models.encoder import MarketEncoder
from torch.utils.data import DataLoader

# Load data
windows_tensor = torch.load("data/windows_tensor.pt")

# Dataset & loader
dataset = MarketDataset(windows_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Models
num_features = windows_tensor.shape[1]
cnn = MarketCNN(in_channels=num_features)
lstm = MarketLSTM(input_dim=num_features)
encoder = MarketEncoder(cnn, lstm)

# Test forward pass
for batch in loader:
    state = encoder(batch)
    print("State shape:", state.shape)
    break
