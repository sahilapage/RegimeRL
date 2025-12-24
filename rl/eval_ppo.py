import torch
import numpy as np
from collections import defaultdict
from stable_baselines3 import PPO

from env.trading_env import TradingEnv
from models.cnn import MarketCNN
from models.lstm import MarketLSTM
from models.encoder import MarketEncoder

import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


windows_test = torch.load("data/windows_test.pt")


close_prices = np.array([
    window[3, -1].item() for window in windows_test
])


close_prices = np.clip(close_prices, 1e-6, None)
log_returns = np.diff(np.log(close_prices))

window = 20
volatility = np.array([
    np.std(log_returns[max(0, i - window):i + 1])
    for i in range(len(log_returns))
])


min_len = min(len(log_returns), len(volatility))
log_returns = log_returns[:min_len]
volatility = volatility[:min_len]


ret_thresh = np.quantile(np.abs(log_returns), 0.6)
vol_thresh = np.quantile(volatility, 0.75)

regimes = []
for r, v in zip(log_returns, volatility):
    if v >= vol_thresh:
        regimes.append("volatile")
    elif abs(r) > ret_thresh:
        regimes.append("trending")
    else:
        regimes.append("sideways")


num_features = windows_test.shape[1]

cnn = MarketCNN(in_channels=num_features)
lstm = MarketLSTM(input_dim=num_features)
encoder = MarketEncoder(cnn, lstm)
encoder.eval()

env = TradingEnv(windows_test, encoder)
model = PPO.load("ppo_finiq_trainonly")


obs = env.reset()
done = False

history = []

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)

    step_idx = env.current_step - 1

    if 0 <= step_idx < len(regimes):
        history.append({
            "regime": regimes[step_idx],
            "balance": env.balance,
            "reward": reward,
            "action": int(action)
        })


stats = defaultdict(lambda: {
    "pnl": 0.0,
    "trades": 0,
    "rewards": []
})

initial_balance = env.initial_balance

for h in history:
    r = h["regime"]
    stats[r]["pnl"] += h["reward"]
    stats[r]["rewards"].append(h["reward"])
    if h["action"] == 1:  # BUY
        stats[r]["trades"] += 1


print("\n===== WALK-FORWARD REGIME RESULTS (TEST DATA) =====\n")

for regime, s in stats.items():
    print(f"Regime: {regime}")
    print(f"  Total PnL       : {s['pnl']:.4f}")
    print(f"  Number of Trades: {s['trades']}")
    print(f"  Avg Reward      : {np.mean(s['rewards']):.6f}")
    print("-" * 40)
