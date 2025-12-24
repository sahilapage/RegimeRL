import os
import torch
import random
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from env.multi_asset_env import MultiAssetTradingEnv
from models.cnn import MarketCNN
from models.lstm import MarketLSTM
from models.encoder import MarketEncoder


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_DIR = "data/train"

asset_windows = {}

for file in os.listdir(DATA_DIR):
    if file.endswith(".pt"):
        asset_name = file.replace(".pt", "")
        asset_windows[asset_name] = torch.load(
            os.path.join(DATA_DIR, file)
        )

print(f"Loaded assets: {list(asset_windows.keys())}")


sample_tensor = next(iter(asset_windows.values()))
num_features = sample_tensor.shape[1]

cnn = MarketCNN(in_channels=num_features)
lstm = MarketLSTM(input_dim=num_features)
encoder = MarketEncoder(cnn, lstm)

encoder.train()


def make_env():
    return MultiAssetTradingEnv(
        asset_windows_dict=asset_windows,
        encoder=encoder
    )

env = DummyVecEnv([make_env])


model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1,
    seed=SEED
)


model.learn(total_timesteps=600_000)


model.save("ppo_finiq_phase3_multiasset")


