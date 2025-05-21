"""Adam Mancini inspired trading strategy for Nifty futures.

This module implements a simplified interpretation of the
levels and intraday signals often posted publicly by Adam
Mancini for the US S&P 500 market.  The logic here adapts
those concepts for the Indian Nifty index.

The strategy computes daily pivot levels using the previous
day's high, low and close.  It also tracks the opening range
for the first few minutes of trading.  Breakouts of the open
range in the direction of the pivot bias create trading
signals.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict


@dataclass
class Levels:
    """Container for daily pivot levels."""

    pivot: float
    r1: float
    r2: float
    r3: float
    s1: float
    s2: float
    s3: float


class AdamManciniNiftyStrategy:
    """Generate trading signals for Nifty based on Adam Mancini style levels."""

    def __init__(self, open_range_minutes: int = 15):
        self.open_range_minutes = open_range_minutes

    def compute_levels(self, daily_ohlc: pd.DataFrame) -> pd.DataFrame:
        """Compute pivot, support and resistance levels.

        Parameters
        ----------
        daily_ohlc : pd.DataFrame
            DataFrame with columns ``high``, ``low`` and ``close`` indexed by
            date.

        Returns
        -------
        pd.DataFrame
            DataFrame containing pivot, support and resistance levels for each
            day.
        """
        pivot = (daily_ohlc["high"] + daily_ohlc["low"] + daily_ohlc["close"]) / 3.0
        r1 = 2 * pivot - daily_ohlc["low"]
        s1 = 2 * pivot - daily_ohlc["high"]
        r2 = pivot + (daily_ohlc["high"] - daily_ohlc["low"])
        s2 = pivot - (daily_ohlc["high"] - daily_ohlc["low"])
        r3 = r1 + (daily_ohlc["high"] - daily_ohlc["low"])
        s3 = s1 - (daily_ohlc["high"] - daily_ohlc["low"])

        levels = pd.DataFrame(
            {
                "pivot": pivot,
                "r1": r1,
                "r2": r2,
                "r3": r3,
                "s1": s1,
                "s2": s2,
                "s3": s3,
            }
        )
        return levels

    def _open_range(self, intraday: pd.DataFrame) -> pd.Series:
        """Return the first ``open_range_minutes`` high and low."""
        subset = intraday.between_time("09:15", (pd.Timestamp("09:15") + pd.Timedelta(minutes=self.open_range_minutes)).time())
        return pd.Series({"high": subset["high"].max(), "low": subset["low"].min()})

    def generate_signals(self, intraday: pd.DataFrame) -> pd.DataFrame:
        """Generate intraday trading signals.

        Parameters
        ----------
        intraday : pd.DataFrame
            Minute level OHLC data for Nifty with a ``datetime`` index.

        Returns
        -------
        pd.DataFrame
            DataFrame containing ``long_signal`` and ``short_signal`` columns.
        """
        if not {"high", "low", "close"}.issubset(intraday.columns):
            raise ValueError("DataFrame must contain high, low and close columns")

        daily = intraday.resample("1D").agg({"high": "max", "low": "min", "close": "last"})
        levels = self.compute_levels(daily)
        open_range = intraday.resample("1D").apply(self._open_range)
        data = intraday.join(levels, how="left").fillna(method="ffill")
        data = data.join(open_range, rsuffix="_or", how="left").fillna(method="ffill")

        data["long_signal"] = np.where(
            (data["close"] > data["high_or"]) & (data["close"] > data["pivot"]), 1, 0
        )
        data["short_signal"] = np.where(
            (data["close"] < data["low_or"]) & (data["close"] < data["pivot"]), -1, 0
        )
        return data

