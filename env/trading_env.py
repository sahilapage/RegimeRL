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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.balance = self.initial_balance
        self.positions = 0
        self.entry_price = 0.0
        self.peak_value = self.initial_balance

        obs = self._get_state()
        self.last_obs = obs
        info = {}

        return obs, info


    def step(self, action):
        terminated = False
        truncated = False

    # =====================================================
    # 1. TERMINATION GUARD (MUST BE FIRST)
    # =====================================================
        if self.current_step >= len(self.windows) - 1:
            terminated = True

    # Force liquidation if holding
            if self.positions == 1:
                final_price = self._get_price(self.current_step)
                self.balance += final_price
                self.positions = 0

            final_value = self.balance
            reward = final_value - self.peak_value  # final settlement reward

            return self.last_obs, reward, terminated, truncated, {}
    
    # =====================================================
    # 2. ACTION MASKING (CRITICAL FIX)
    # =====================================================
    # Cannot SELL if no position
        if self.positions == 0 and action == 2:
            action = 0  # HOLD

    # Cannot BUY if already holding
        if self.positions == 1 and action == 1:
            action = 0  # HOLD

    # =====================================================
    # 3. CURRENT PRICE & PORTFOLIO VALUE
    # =====================================================
        curr_price = self._get_price(self.current_step)
        prev_value = self.balance + self.positions * curr_price

    # =====================================================
    # 4. EXECUTE ACTION (ONLY IF VALID)
    # =====================================================
        if action == 1 and self.positions == 0:  # BUY
            self.positions = 1
            self.entry_price = curr_price
            self.balance -= self.transaction_cost

        elif action == 2 and self.positions == 1:  # SELL
            self.positions = 0
            self.entry_price = 0.0
            self.balance -= self.transaction_cost

    # =====================================================
    # 5. ADVANCE TIME
    # =====================================================
        self.current_step += 1

    # =====================================================
    # 6. NEXT PRICE & CURRENT VALUE
    # =====================================================
        next_price = self._get_price(self.current_step)
        curr_value = self.balance + self.positions * next_price

    # =====================================================
    # 7. DRAWDOWN TRACKING
    # =====================================================
        self.peak_value = max(self.peak_value, curr_value)
        drawdown = self.peak_value - curr_value

    # =====================================================
    # 8. RISK-AWARE REWARD
    # =====================================================
        LAMBDA_DD = 0.001
        reward = (curr_value - prev_value) - LAMBDA_DD * drawdown

    # Small inactivity penalty (prevents "do nothing forever")
        if action == 0:
            reward -= 1e-5

    # =====================================================
    # 9. NEXT OBSERVATION (SAFE)
    # =====================================================
        obs = self._get_state()
        self.last_obs = obs

        return obs, reward, terminated, truncated, {}

    def _get_state(self):
        window = self.windows[self.current_step].unsqueeze(0)
        with torch.no_grad():
            state = self.encoder(window)
        return state.squeeze(0).numpy()
    
    def _get_price(self, step):
        return self.windows[step][3, -1].item()
    
    