"""Simple walk-forward backtesting framework."""

from __future__ import annotations

import pandas as pd
from typing import Callable, Tuple


class Backtester:
    """Run walk-forward backtests using a strategy callable."""

    def __init__(self, data: pd.DataFrame, strategy_fn: Callable[[pd.DataFrame], pd.DataFrame]):
        self.data = data
        self.strategy_fn = strategy_fn

    def run(self, train_window: int = 180, test_window: int = 30, step: int = 30) -> Tuple[pd.DataFrame, float]:
        equity = 1.0
        all_results = []
        for start in range(0, len(self.data) - train_window - test_window, step):
            train = self.data.iloc[start : start + train_window]
            test = self.data.iloc[start + train_window : start + train_window + test_window]
            signals = self.strategy_fn(train)
            # Very naive performance: sum of signals times returns
            returns = test['close'].pct_change().fillna(0)
            pnl = (signals['long_signal'] - signals['short_signal']) * returns
            equity *= (1 + pnl).prod()
            all_results.append(pnl)
        results = pd.concat(all_results, axis=0)
        return results, equity
