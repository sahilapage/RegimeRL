import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from env.trading_env import TradingEnv
from models.cnn import MarketCNN
from models.lstm import MarketLSTM
from models.encoder import MarketEncoder

windows_tensor = torch.load('data/windows_train.pt')

num_features = windows_tensor.shape[1]

cnn = MarketCNN(in_channels=num_features)
lstm = MarketLSTM(input_dim=num_features)
encoder = MarketEncoder(cnn, lstm)

def make_env():
    return TradingEnv(
        windows_tensor=windows_tensor,
        encoder=encoder
    )

env =  DummyVecEnv([make_env])

model = PPO(
    policy='MlpPolicy',
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64, 
    verbose=1,
)   

model.learn(total_timesteps=300_000)

model.save("ppo_finiq_trainonly")