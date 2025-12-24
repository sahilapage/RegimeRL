import gym
import numpy as np
import torch
from gym import spaces

class TradingEnv(gym.Env):

    def __init__(
        self,
        windows_tensor,
        encoder,
        initial_balance = 10000,
        transaction_cost = 0.0005,
    ):
        super().__init__()

        self.windows = windows_tensor
        self.encoder = encoder
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.peak_value = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.positions = 0
        self.entry_price = 0.0

        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell

        self.state_dim = 128
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.peak_value = self.initial_balance
        self.positions = 0
        self.entry_price = 0.0

        return self._get_state()

    def step(self, action):
      done = False

      prev_price = self._get_price(self.current_step)
      prev_value = self.balance + self.positions * prev_price

      if action == 1 and self.positions == 0:  # BUY
          self.positions = 1
          self.entry_price = prev_price
          self.balance -= self.transaction_cost

      elif action == 2 and self.positions == 1:  # SELL
          self.positions = 0
          self.entry_price = 0.0
          self.balance -= self.transaction_cost

      self.current_step += 1
      if self.current_step >= len(self.windows) - 1:
          done = True

      curr_price = self._get_price(self.current_step)
      curr_value = self.balance + self.positions * curr_price

      self.peak_value = max(self.peak_value, curr_value)
      drawdown = self.peak_value - curr_value

      LAMBDA_DD = 0.001  # risk aversion strength
      reward = (curr_value - prev_value) - LAMBDA_DD * drawdown

      return self._get_state(), reward, done, {}

    
    def _get_state(self):
        window = self.windows[self.current_step].unsqueeze(0)
        with torch.no_grad():
            state = self.encoder(window)
        return state.squeeze(0).numpy()
    
    def _get_price(self, step):
        return self.windows[step][3, -1].item()
    
    