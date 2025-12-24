from env.trading_env import TradingEnv
import torch

# Load data
windows_tensor = torch.load("data/windows_tensor.pt")

# Build models
from models.cnn import MarketCNN
from models.lstm import MarketLSTM
from models.encoder import MarketEncoder

num_features = windows_tensor.shape[1]
cnn = MarketCNN(in_channels=num_features)
lstm = MarketLSTM(input_dim=num_features)
encoder = MarketEncoder(cnn, lstm)

# Create env
env = TradingEnv(windows_tensor, encoder)

state = env.reset()
print("Initial state shape:", state.shape)

for _ in range(20):
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)
    print(f"Action: {action}, Reward: {reward:.4f}")
