"""Simplified xLSTM-PPO reinforcement learning components."""

from __future__ import annotations

import gymnasium as gym
import torch
from torch import nn
import pytorch_lightning as pl


class TradingEnv(gym.Env):
    """Minimal trading environment for RL."""

    def __init__(self, prices: torch.Tensor):
        super().__init__()
        self.prices = prices
        self.action_space = gym.spaces.Discrete(3)  # short, hold, long
        self.observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(prices.shape[1],))
        self.step_idx = 0

    def reset(self, *, seed=None, options=None):  # type: ignore
        self.step_idx = 0
        return self.prices[self.step_idx], {}

    def step(self, action):
        self.step_idx += 1
        done = self.step_idx >= len(self.prices)
        reward = float(action - 1) * float(self.prices[self.step_idx] - self.prices[self.step_idx - 1]) if not done else 0.0
        obs = self.prices[self.step_idx] if not done else self.prices[-1]
        return obs, reward, done, False, {}


class xLSTMPPO(pl.LightningModule):
    """Tiny xLSTM policy trained with PPO."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.policy = nn.Linear(hidden_dim, 3)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        out, _ = self.lstm(x)
        h = out[:, -1]
        return self.policy(h), self.value(h)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def train_agent(env: gym.Env, **kwargs) -> xLSTMPPO:
    """Train PPO agent in the environment (dummy implementation)."""
    model = xLSTMPPO(input_dim=env.observation_space.shape[0], **kwargs)
    # Placeholder training loop
    return model
