import torch
import numpy as np
from stable_baselines3 import PPO
import unittest

from env.finiq_env import FinIQEnv
from models.cnn import MarketCNN
from models.lstm import MarketLSTM
from models.encoder import MarketEncoder

assets = ["AAPL", "MSFT", "NVDA", "TSLA"]
windows = {
    a: torch.load(f"data/test/{a}_test.pt")
    for a in assets
}

num_features = windows[assets[0]].shape[1]
cnn = MarketCNN(in_channels=num_features)
lstm = MarketLSTM(input_dim=num_features)
encoder = MarketEncoder(cnn, lstm)

env = FinIQEnv(
    windows_dict=windows,
    encoder=encoder,
    action_type="continuous",
    regime_aware=True
)

model = PPO.load("ppo_finiq_final")

obs, _ = env.reset()
done = False

balances = []

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)
    balances.append(info["balance"])

print("\n===== FINAL RESULTS =====")
print("Final Balance:", balances[-1])
print("Total Return:", balances[-1] - balances[0])

class TestRewardFunction(unittest.TestCase):

    def test_reward_function(self):
        env = FinIQEnv(
            windows_dict=windows,
            encoder=encoder,
            action_type="continuous",
            regime_aware=True
        )
        obs, _ = env.reset()
        done = False
        rewards = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            rewards.append(reward)
        self.assertIsNotNone(rewards)
        self.assertGreaterEqual(sum(rewards), 0)

    def test_reward_function_zero_action(self):
        env = FinIQEnv(
            windows_dict=windows,
            encoder=encoder,
            action_type="continuous",
            regime_aware=True
        )
        obs, _ = env.reset()
        done = False
        rewards = []
        while not done:
            action = np.zeros_like(model.predict(obs, deterministic=True)[0])
            obs, reward, done, _, _ = env.step(action)
            rewards.append(reward)
        self.assertIsNotNone(rewards)
        self.assertGreaterEqual(sum(rewards), 0)

    def test_reward_function_large_action(self):
        env = FinIQEnv(
            windows_dict=windows,
            encoder=encoder,
            action_type="continuous",
            regime_aware=True
        )
        obs, _ = env.reset()
        done = False
        rewards = []
        while not done:
            action = np.ones_like(model.predict(obs, deterministic=True)[0]) * 100
            obs, reward, done, _, _ = env.step(action)
            rewards.append(reward)
        self.assertIsNotNone(rewards)
        self.assertGreaterEqual(sum(rewards), 0)

if __name__ == '__main__':
    unittest.main(exit=False)