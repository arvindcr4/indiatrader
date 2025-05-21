"""Simple paper trading broker interface."""

from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PaperBroker:
    """Record trades without executing them on a real exchange."""

    def __init__(self):
        self.trades: List[dict] = []

    def place_order(self, symbol: str, quantity: int, side: str, price: float):
        trade = {"symbol": symbol, "qty": quantity, "side": side, "price": price}
        logger.info("Paper trade: %s", trade)
        self.trades.append(trade)

    def get_trade_log(self) -> List[dict]:
        return self.trades
