import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces


class FinIQEnv(gym.Env):
    """
    Unified Trading Environment (Gymnasium + SB3 compatible)
    Supports:
    - single / multi asset
    - discrete / continuous actions
    - regime-aware drawdown penalty
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        windows_dict,              # {"AAPL": tensor, ...}
        encoder,
        action_type="continuous",  # "discrete" | "continuous"
        regime_aware=True,
        initial_balance=10000,
        transaction_cost=0.0005,
        base_lambda_dd=0.001,
        min_hold_steps=5,
    ):
        super().__init__()

        # -------- Data --------
        self.assets = list(windows_dict.keys())
        self.windows = windows_dict
        self.encoder = encoder

        self.num_assets = len(self.assets)
        self.max_steps = min(w.shape[0] for w in windows_dict.values())

        # -------- Config --------
        self.action_type = action_type
        self.regime_aware = regime_aware
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.base_lambda_dd = base_lambda_dd
        self.min_hold_steps = min_hold_steps

        # -------- Regime lambdas --------
        self.REGIME_LAMBDA = {
            "trending": 0.0003,
            "sideways": 0.001,
            "volatile": 0.003,
        }

        # -------- Action Space --------
        if action_type == "discrete":
            self.action_space = spaces.Discrete(3)  # Hold, Buy, Sell
        else:
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.num_assets,),
                dtype=np.float32,
            )

        # -------- Observation Space --------
        self.state_dim = 128 * self.num_assets
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32,
        )

        self.reset()

    # -------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.step_idx = 0
        self.balance = self.initial_balance
        self.peak_balance = self.initial_balance

        self.positions = np.zeros(self.num_assets, dtype=np.float32)
        self.entry_prices = np.zeros(self.num_assets, dtype=np.float32)
        self.hold_steps = np.zeros(self.num_assets, dtype=np.int32)

        obs = self._get_state()
        return obs, {}

    # -------------------------------------------------
    def step(self, action):
        reward = 0.0

        terminated = False
        truncated = False

        if self.step_idx >= self.max_steps - 1:
            terminated = True
            return self._get_state(), 0.0, terminated, truncated, {}

        prices_now = self._get_prices(self.step_idx)
        prices_next = self._get_prices(self.step_idx + 1)

        # -------- ACTION --------
        if self.action_type == "discrete":
            reward += self._step_discrete(action, prices_now)
        else:
            reward += self._step_continuous(action, prices_now)

        # -------- PnL --------
        pnl = np.sum(self.positions * (prices_next - prices_now))
        reward += pnl
        self.balance += pnl

        # -------- DRAWDOWN --------
        self.peak_balance = max(self.peak_balance, self.balance)
        drawdown = self.peak_balance - self.balance

        regime = self._detect_regime()
        lambda_dd = self.REGIME_LAMBDA.get(regime, self.base_lambda_dd)

        reward -= lambda_dd * drawdown

        # -------- STEP --------
        self.step_idx += 1
        terminated = self.step_idx >= self.max_steps - 1

        info = {
            "balance": self.balance,
            "drawdown": drawdown,
            "regime": regime,
        }

        obs = self._get_state()
        return obs, reward, terminated, truncated, info

    # -------------------------------------------------
    def _step_discrete(self, action, prices):
        reward = 0.0

        for i in range(self.num_assets):
            if action == 1 and self.positions[i] == 0:
                self.positions[i] = 1.0
                self.entry_prices[i] = prices[i]
                reward -= self.transaction_cost

            elif action == 2 and self.positions[i] == 1:
                if self.hold_steps[i] >= self.min_hold_steps:
                    reward += prices[i] - self.entry_prices[i]
                    self.positions[i] = 0.0
                    self.entry_prices[i] = 0.0
                    self.hold_steps[i] = 0
                    reward -= self.transaction_cost

            if self.positions[i] != 0:
                self.hold_steps[i] += 1

        return reward

    # -------------------------------------------------
    def _step_continuous(self, action, prices):
        action = np.asarray(action, dtype=np.float32)
        target_pos = np.clip(action, -1.0, 1.0)

        delta = target_pos - self.positions
        cost = np.sum(np.abs(delta)) * self.transaction_cost

        self.positions = target_pos
        return -cost

    # -------------------------------------------------
    def _get_state(self):
        embeddings = []

        for asset in self.assets:
            window = self.windows[asset][self.step_idx].unsqueeze(0)
            with torch.no_grad():
                emb = self.encoder(window)
            embeddings.append(emb.squeeze(0))

        return torch.cat(embeddings).cpu().numpy().astype(np.float32)

    # -------------------------------------------------
    def _get_prices(self, step):
        prices = []
        for asset in self.assets:
            prices.append(self.windows[asset][step][3, -1].item())
        return np.array(prices, dtype=np.float32)

    # -------------------------------------------------
    def _detect_regime(self):
        lookback = 20
        if self.step_idx < lookback:
            return "sideways"

        prices = []
        for asset in self.assets:
            close = self.windows[asset][
                self.step_idx - lookback : self.step_idx + 1,
                3,
                -1
            ].cpu().numpy()
            prices.append(close)

        prices = np.mean(prices, axis=0)
        prices = np.clip(prices, 1e-6, None)

        returns = np.diff(np.log(prices))
        vol = np.std(returns)
        trend = np.abs(np.mean(returns))

        if vol > 0.015:
            return "volatile"
        elif trend > 0.002:
            return "trending"
        else:
            return "sideways"
