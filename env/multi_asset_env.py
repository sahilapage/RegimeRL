import random
import gym
from env.trading_env import TradingEnv

class MultiAssetTradingEnv(gym.Env):
    def __init__(self, asset_windows_dict, encoder):
        self.asset_windows = asset_windows_dict
        self.encoder = encoder
        self.asset_names = list(asset_windows_dict.keys())
        self.current_env = None

        sample_asset = self.asset_names[0]
        dummy_env = TradingEnv(
            windows_tensor=self.asset_windows[sample_asset],
            encoder=self.encoder
        )

        self.action_space = dummy_env.action_space
        self.observation_space = dummy_env.observation_space

    def seed(self, seed=None):
        random.seed(seed)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        asset = random.choice(self.asset_names)

        self.current_env = TradingEnv(
            windows_tensor=self.asset_windows[asset],
            encoder=self.encoder
        )

        return self.current_env.reset(seed=seed)


    def step(self, action):
      return self.current_env.step(action)

