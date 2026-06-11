import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.logger import Logger

from env.finiq_env import FinIQEnv
from models.cnn import MarketCNN
from models.lstm import MarketLSTM
from models.encoder import MarketEncoder

# ---------- LOAD DATA ----------
assets = ["AAPL", "MSFT", "NVDA", "TSLA"]
windows = {
    a: torch.load(f"data/train/{a}_train.pt")
    for a in assets
}

# ---------- MODEL ----------
num_features = windows[assets[0]].shape[1]

cnn = MarketCNN(in_channels=num_features)
lstm = MarketLSTM(input_dim=num_features)
encoder = MarketEncoder(cnn, lstm)

def make_env():
    return FinIQEnv(
        windows_dict=windows,
        encoder=encoder,
        action_type="continuous",
        regime_aware=True
    )

env = DummyVecEnv([make_env])

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    verbose=1
)

logger = Logger()

def callback(event):
    logger.log_metrics(event)

model.learn(total_timesteps=600_000, callback=callback)
model.save("ppo_finiq_final")

print("TRAINING COMPLETE")
logger.save_logs("training_logs")