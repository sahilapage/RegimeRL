import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import unittest

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

class TestRewardFunction(unittest.TestCase):
    def test_reward_function(self):
        env = make_env()
        obs = env.reset()
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        self.assertIsNotNone(reward)
        self.assertIsInstance(reward, float)

    def test_reward_function_multiple_steps(self):
        env = make_env()
        obs = env.reset()
        for _ in range(10):
            action = env.action_space.sample()
            next_obs, reward, done, info = env.step(action)
            self.assertIsNotNone(reward)
            self.assertIsInstance(reward, float)
            if done:
                break

    def test_reward_function_edge_cases(self):
        env = make_env()
        obs = env.reset()
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        self.assertIsNotNone(reward)
        self.assertIsInstance(reward, float)

        # Test with zero action
        action = [0.0] * env.action_space.shape[0]
        next_obs, reward, done, info = env.step(action)
        self.assertIsNotNone(reward)
        self.assertIsInstance(reward, float)

        # Test with maximum action
        action = [1.0] * env.action_space.shape[0]
        next_obs, reward, done, info = env.step(action)
        self.assertIsNotNone(reward)
        self.assertIsInstance(reward, float)

if __name__ == "__main__":
    unittest.main(exit=False)
    model.learn(total_timesteps=600_000)
    model.save("ppo_finiq_final")
    print("TRAINING COMPLETE")