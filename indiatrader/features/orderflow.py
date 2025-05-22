"""Order flow feature generation for market data."""

import logging
import pandas as pd
from typing import Dict, List

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class OrderFlowFeatureGenerator:
    """Generate order-flow related features."""

    def generate_features(self, df: pd.DataFrame, feature_config: List[Dict]) -> pd.DataFrame:
        """Generate features based on configuration."""
        result_df = df.copy()
        for spec in feature_config:
            name = spec.get("name")
            params = spec.get("params", {})
            if name == "vwap":
                result_df = self._add_vwap(result_df, params.get("windows", [5]))
            elif name == "order_imbalance":
                result_df = self._add_order_imbalance(result_df)
            elif name == "delta_volume":
                result_df = self._add_delta_volume(result_df, params.get("windows", [5]))
            else:
                logger.warning(f"Unknown order flow feature: {name}")
        return result_df

    def _add_vwap(self, df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        if "close" not in df.columns or "volume" not in df.columns:
            logger.warning("Columns 'close' and 'volume' required for VWAP")
            return df
        price_volume = df["close"] * df["volume"]
        for window in windows:
            pv_sum = price_volume.rolling(window).sum()
            vol_sum = df["volume"].rolling(window).sum()
            df[f"vwap_{window}"] = pv_sum / vol_sum
        return df

    def _add_order_imbalance(self, df: pd.DataFrame) -> pd.DataFrame:
        if "bid_volume" in df.columns and "ask_volume" in df.columns:
            total = df["bid_volume"] + df["ask_volume"]
            total = total.replace(0, pd.NA)
            df["order_imbalance"] = (df["bid_volume"] - df["ask_volume"]) / total
        else:
            logger.warning("Order book columns not found; skipping order_imbalance")
        return df

    def _add_delta_volume(self, df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        if "volume" not in df.columns:
            logger.warning("Volume column not found; cannot calculate delta volume")
            return df
        for window in windows:
            df[f"delta_volume_{window}"] = df["volume"].diff(window)
        return df
