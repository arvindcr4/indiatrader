"""
Order flow feature engineering for market data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import logging

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class OrderFlowFeatureGenerator:
    """
    Generate order flow features from market data.
    """
    
    def __init__(self):
        """
        Initialize order flow feature generator.
        """
        pass
    
    def generate_features(self, 
                         df: pd.DataFrame, 
                         orderbook_df: Optional[pd.DataFrame] = None,
                         feature_config: Dict = None) -> pd.DataFrame:
        """
        Generate order flow features based on configuration.
        
        Args:
            df: DataFrame with OHLCV data
            orderbook_df: Optional DataFrame with order book data
            feature_config: Feature configuration dictionary
        
        Returns:
            DataFrame with added order flow features
        """
        # Make a copy to avoid modifying the original DataFrame
        result_df = df.copy()
        
        # Extract column names for OHLCV
        price_cols = self._get_price_columns(result_df)
        
        # Generate features for each configured feature type
        for feature_spec in feature_config:
            feature_type = feature_spec["name"]
            params = feature_spec.get("params", {})
            
            if feature_type == "vwap":
                result_df = self._add_vwap(result_df, price_cols, **params)
            
            elif feature_type == "order_imbalance" and orderbook_df is not None:
                result_df = self._add_order_imbalance(result_df, orderbook_df, **params)
            
            elif feature_type == "delta_volume":
                result_df = self._add_delta_volume(result_df, price_cols, **params)
            
            elif feature_type == "volume_profile":
                result_df = self._add_volume_profile(result_df, price_cols, **params)
            
            elif feature_type == "price_levels":
                result_df = self._add_price_levels(result_df, price_cols, **params)
            
            elif feature_type == "vpin":
                result_df = self._add_vpin(result_df, price_cols, **params)
            
            else:
                logger.warning(f"Unknown feature type: {feature_type}")
        
        return result_df
    
    def _get_price_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Extract OHLCV column names from DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            Dictionary mapping standardized column names to actual column names
        """
        # Common variations of column names
        open_candidates = ["open", "Open", "OPEN"]
        high_candidates = ["high", "High", "HIGH"]
        low_candidates = ["low", "Low", "LOW"]
        close_candidates = ["close", "Close", "CLOSE"]
        volume_candidates = ["volume", "Volume", "VOLUME", "vol", "Vol"]
        
        price_cols = {}
        
        # Find open column
        for col in open_candidates:
            if col in df.columns:
                price_cols["open"] = col
                break
        
        # Find high column
        for col in high_candidates:
            if col in df.columns:
                price_cols["high"] = col
                break
        
        # Find low column
        for col in low_candidates:
            if col in df.columns:
                price_cols["low"] = col
                break
        
        # Find close column
        for col in close_candidates:
            if col in df.columns:
                price_cols["close"] = col
                break
        
        # Find volume column
        for col in volume_candidates:
            if col in df.columns:
                price_cols["volume"] = col
                break
        
        # Validate that required columns were found
        required_cols = ["close", "volume"]
        for col in required_cols:
            if col not in price_cols:
                logger.warning(f"Required column not found: {col}")
        
        return price_cols
    
    def _add_vwap(self, 
                 df: pd.DataFrame, 
                 price_cols: Dict[str, str], 
                 windows: List[int] = [5, 15, 30, 60]) -> pd.DataFrame:
        """
        Add Volume Weighted Average Price (VWAP) features.
        
        Args:
            df: DataFrame with OHLCV data
            price_cols: Dictionary mapping standard column names to actual column names
            windows: List of window sizes for VWAP calculation
        
        Returns:
            DataFrame with added VWAP features
        """
        # Ensure required columns are available
        required_cols = ["high", "low", "close", "volume"]
        for col in required_cols:
            if col not in price_cols:
                logger.warning(f"Required column {col} not found. Cannot calculate VWAP.")
                return df
        
        high_col = price_cols["high"]
        low_col = price_cols["low"]
        close_col = price_cols["close"]
        volume_col = price_cols["volume"]
        
        # Calculate typical price
        df["typical_price"] = (df[high_col] + df[low_col] + df[close_col]) / 3
        
        # Calculate VWAP for each window
        for window in windows:
            # Calculate cumulative (typical price * volume)
            df[f"cum_tpv_{window}"] = (df["typical_price"] * df[volume_col]).rolling(window=window).sum()
            
            # Calculate cumulative volume
            df[f"cum_volume_{window}"] = df[volume_col].rolling(window=window).sum()
            
            # Calculate VWAP
            df[f"vwap_{window}"] = df[f"cum_tpv_{window}"] / df[f"cum_volume_{window}"]
            
            # Calculate percentage difference from VWAP
            df[f"vwap_{window}_pct"] = (df[close_col] - df[f"vwap_{window}"]) / df[f"vwap_{window}"] * 100
            
            # Drop intermediate columns
            df = df.drop(columns=[f"cum_tpv_{window}", f"cum_volume_{window}"])
        
        # Calculate VWAP crossover signals
        for window in windows:
            df[f"vwap_{window}_cross"] = np.where(
                (df[close_col] > df[f"vwap_{window}"]) & (df[close_col].shift(1) <= df[f"vwap_{window}"].shift(1)), 1,
                np.where((df[close_col] < df[f"vwap_{window}"]) & (df[close_col].shift(1) >= df[f"vwap_{window}"].shift(1)), -1, 0)
            )
        
        # Drop intermediate columns
        df = df.drop(columns=["typical_price"])
        
        return df
    
    def _add_order_imbalance(self, 
                            df: pd.DataFrame, 
                            orderbook_df: pd.DataFrame, 
                            levels: int = 5) -> pd.DataFrame:
        """
        Add order book imbalance features.
        
        Args:
            df: DataFrame with OHLCV data
            orderbook_df: DataFrame with order book data
            levels: Number of price levels to consider
        
        Returns:
            DataFrame with added order imbalance features
        """
        # Ensure orderbook_df has required columns
        required_cols = ["timestamp", "side", "price", "quantity"]
        for col in required_cols:
            if col not in orderbook_df.columns:
                logger.warning(f"Required column {col} not found in order book data. Cannot calculate order imbalance.")
                return df
        
        # Group order book data by timestamp
        grouped = orderbook_df.groupby("timestamp")
        
        # Initialize empty lists for each feature
        bid_ask_imbalance = []
        bid_ask_spread = []
        depth_imbalance = []
        
        # Process each timestamp
        for timestamp, group in grouped:
            # Separate bids and asks
            bids = group[group["side"] == "bid"].sort_values("price", ascending=False)
            asks = group[group["side"] == "ask"].sort_values("price")
            
            # Calculate bid-ask spread
            if not bids.empty and not asks.empty:
                best_bid = bids.iloc[0]["price"]
                best_ask = asks.iloc[0]["price"]
                spread = best_ask - best_bid
                spread_pct = spread / best_bid * 100
            else:
                spread = np.nan
                spread_pct = np.nan
            
            # Calculate top of book imbalance
            if not bids.empty and not asks.empty:
                best_bid_qty = bids.iloc[0]["quantity"]
                best_ask_qty = asks.iloc[0]["quantity"]
                tob_imbalance = (best_bid_qty - best_ask_qty) / (best_bid_qty + best_ask_qty)
            else:
                tob_imbalance = np.nan
            
            # Calculate depth imbalance (for specified number of levels)
            if not bids.empty and not asks.empty:
                bid_depth = bids.head(levels)["quantity"].sum()
                ask_depth = asks.head(levels)["quantity"].sum()
                depth_imb = (bid_depth - ask_depth) / (bid_depth + ask_depth)
            else:
                depth_imb = np.nan
            
            # Append values to lists
            bid_ask_imbalance.append(tob_imbalance)
            bid_ask_spread.append(spread_pct)
            depth_imbalance.append(depth_imb)
        
        # Create a DataFrame with calculated features
        imbalance_df = pd.DataFrame({
            "timestamp": list(grouped.groups.keys()),
            "bid_ask_imbalance": bid_ask_imbalance,
            "bid_ask_spread_pct": bid_ask_spread,
            "depth_imbalance": depth_imbalance
        })
        
        # Merge with the original DataFrame
        result_df = pd.merge(df, imbalance_df, on="timestamp", how="left")
        
        # Fill missing values
        result_df[["bid_ask_imbalance", "bid_ask_spread_pct", "depth_imbalance"]] = result_df[
            ["bid_ask_imbalance", "bid_ask_spread_pct", "depth_imbalance"]
        ].fillna(method="ffill")
        
        # Add derived signals
        result_df["strong_imbalance"] = np.where(abs(result_df["bid_ask_imbalance"]) > 0.7, 1, 0)
        result_df["wide_spread"] = np.where(result_df["bid_ask_spread_pct"] > result_df["bid_ask_spread_pct"].rolling(window=50).mean() * 2, 1, 0)
        
        return result_df
    
    def _add_delta_volume(self, 
                         df: pd.DataFrame, 
                         price_cols: Dict[str, str], 
                         windows: List[int] = [5, 15, 30]) -> pd.DataFrame:
        """
        Add Delta Volume features (buying vs. selling pressure).
        
        Args:
            df: DataFrame with OHLCV data
            price_cols: Dictionary mapping standard column names to actual column names
            windows: List of window sizes for calculations
        
        Returns:
            DataFrame with added Delta Volume features
        """
        # Ensure required columns are available
        required_cols = ["open", "close", "volume"]
        for col in required_cols:
            if col not in price_cols:
                logger.warning(f"Required column {col} not found. Cannot calculate Delta Volume.")
                return df
        
        open_col = price_cols["open"]
        close_col = price_cols["close"]
        volume_col = price_cols["volume"]
        
        # Calculate price change
        df["price_change"] = df[close_col] - df[open_col]
        
        # Determine if buying or selling volume
        df["up_volume"] = np.where(df["price_change"] >= 0, df[volume_col], 0)
        df["down_volume"] = np.where(df["price_change"] < 0, df[volume_col], 0)
        
        # Calculate delta volume (buying - selling)
        df["delta_volume"] = df["up_volume"] - df["down_volume"]
        
        # Calculate cumulative delta volume
        df["cum_delta_volume"] = df["delta_volume"].cumsum()
        
        # Calculate rolling delta volume for each window
        for window in windows:
            df[f"delta_volume_{window}"] = df["delta_volume"].rolling(window=window).sum()
            
            # Calculate buying/selling ratio
            df[f"up_volume_{window}"] = df["up_volume"].rolling(window=window).sum()
            df[f"down_volume_{window}"] = df["down_volume"].rolling(window=window).sum()
            df[f"vol_ratio_{window}"] = df[f"up_volume_{window}"] / df[f"down_volume_{window}"]
            
            # Calculate momentum based on delta volume
            df[f"delta_vol_momentum_{window}"] = df[f"delta_volume_{window}"] - df[f"delta_volume_{window}"].shift(window)
            
            # Calculate divergence signals
            df[f"price_up_{window}"] = df[close_col] > df[close_col].shift(window)
            df[f"delta_vol_down_{window}"] = df[f"delta_volume_{window}"] < df[f"delta_volume_{window}"].shift(window)
            df[f"delta_vol_divergence_{window}"] = (df[f"price_up_{window}"] & df[f"delta_vol_down_{window}"]).astype(int)
            
            # Drop intermediate columns
            df = df.drop(columns=[f"up_volume_{window}", f"down_volume_{window}", f"price_up_{window}", f"delta_vol_down_{window}"])
        
        # Drop intermediate columns
        df = df.drop(columns=["price_change", "up_volume", "down_volume"])
        
        return df
    
    def _add_volume_profile(self, 
                           df: pd.DataFrame, 
                           price_cols: Dict[str, str], 
                           num_bins: int = 10,
                           window: int = 20) -> pd.DataFrame:
        """
        Add Volume Profile features.
        
        Args:
            df: DataFrame with OHLCV data
            price_cols: Dictionary mapping standard column names to actual column names
            num_bins: Number of price bins for volume profile
            window: Rolling window size for volume profile calculation
        
        Returns:
            DataFrame with added Volume Profile features
        """
        # Ensure required columns are available
        required_cols = ["high", "low", "close", "volume"]
        for col in required_cols:
            if col not in price_cols:
                logger.warning(f"Required column {col} not found. Cannot calculate Volume Profile.")
                return df
        
        high_col = price_cols["high"]
        low_col = price_cols["low"]
        close_col = price_cols["close"]
        volume_col = price_cols["volume"]
        
        # Initialize columns for POC (Point of Control) and VAH/VAL (Value Area High/Low)
        df["vp_poc"] = np.nan
        df["vp_vah"] = np.nan
        df["vp_val"] = np.nan
        df["vp_above_poc"] = np.nan
        df["vp_below_poc"] = np.nan
        
        # Calculate Volume Profile for each row using a rolling window
        for i in range(window, len(df)):
            # Extract window of data
            window_data = df.iloc[i-window:i]
            
            # Determine price range for the window
            price_min = window_data[low_col].min()
            price_max = window_data[high_col].max()
            
            # Create price bins
            price_bins = np.linspace(price_min, price_max, num_bins + 1)
            bin_centers = (price_bins[:-1] + price_bins[1:]) / 2
            
            # Initialize volume counts for each bin
            bin_volumes = np.zeros(num_bins)
            
            # Distribute volume across price range for each bar
            for j in range(len(window_data)):
                bar = window_data.iloc[j]
                bar_low = bar[low_col]
                bar_high = bar[high_col]
                bar_volume = bar[volume_col]
                
                # Find bins that overlap with this bar's price range
                overlapping_bins = ((bin_centers >= bar_low) & (bin_centers <= bar_high))
                
                if np.any(overlapping_bins):
                    # Distribute volume equally among overlapping bins
                    bin_volumes[overlapping_bins] += bar_volume / np.sum(overlapping_bins)
            
            # Find POC (Point of Control) - price level with highest volume
            poc_idx = np.argmax(bin_volumes)
            poc_price = bin_centers[poc_idx]
            
            # Define value area (70% of total volume)
            total_volume = np.sum(bin_volumes)
            target_volume = total_volume * 0.7
            
            # Sort bins by volume (descending)
            sorted_idx = np.argsort(bin_volumes)[::-1]
            cumulative_volume = 0
            value_area_bins = set()
            
            # Add bins to value area until we reach target volume
            for idx in sorted_idx:
                value_area_bins.add(idx)
                cumulative_volume += bin_volumes[idx]
                if cumulative_volume >= target_volume:
                    break
            
            # Find VAH (Value Area High) and VAL (Value Area Low)
            vah_idx = max(value_area_bins)
            val_idx = min(value_area_bins)
            vah_price = bin_centers[vah_idx]
            val_price = bin_centers[val_idx]
            
            # Store values
            df.loc[df.index[i], "vp_poc"] = poc_price
            df.loc[df.index[i], "vp_vah"] = vah_price
            df.loc[df.index[i], "vp_val"] = val_price
            
            # Calculate relative position of current price to POC
            current_price = df.loc[df.index[i], close_col]
            df.loc[df.index[i], "vp_above_poc"] = 1 if current_price > poc_price else 0
            df.loc[df.index[i], "vp_below_poc"] = 1 if current_price < poc_price else 0
        
        # Forward fill missing values
        df[["vp_poc", "vp_vah", "vp_val", "vp_above_poc", "vp_below_poc"]] = df[
            ["vp_poc", "vp_vah", "vp_val", "vp_above_poc", "vp_below_poc"]
        ].fillna(method="ffill")
        
        # Calculate distance from current price to POC (percentage)
        df["vp_poc_distance"] = (df[close_col] - df["vp_poc"]) / df["vp_poc"] * 100
        
        # Add signal for price at value area boundaries
        df["vp_at_vah"] = np.where(abs(df[close_col] - df["vp_vah"]) / df["vp_vah"] < 0.005, 1, 0)
        df["vp_at_val"] = np.where(abs(df[close_col] - df["vp_val"]) / df["vp_val"] < 0.005, 1, 0)
        
        return df
    
    def _add_price_levels(self, 
                         df: pd.DataFrame, 
                         price_cols: Dict[str, str], 
                         window: int = 50,
                         min_touches: int = 2) -> pd.DataFrame:
        """
        Add Support and Resistance Price Level features.
        
        Args:
            df: DataFrame with OHLCV data
            price_cols: Dictionary mapping standard column names to actual column names
            window: Lookback window for identifying price levels
            min_touches: Minimum number of touches required to confirm a level
        
        Returns:
            DataFrame with added Price Level features
        """
        # Ensure required columns are available
        required_cols = ["high", "low", "close"]
        for col in required_cols:
            if col not in price_cols:
                logger.warning(f"Required column {col} not found. Cannot calculate Price Levels.")
                return df
        
        high_col = price_cols["high"]
        low_col = price_cols["low"]
        close_col = price_cols["close"]
        
        # Initialize columns for support and resistance levels
        df["support_level"] = np.nan
        df["resistance_level"] = np.nan
        df["at_support"] = 0
        df["at_resistance"] = 0
        
        # Calculate price levels for each row using a rolling window
        for i in range(window, len(df)):
            # Extract window of data
            window_data = df.iloc[i-window:i]
            
            # Find recent price levels
            # This is a simplified approach - real implementations would use more sophisticated methods
            # like detecting swing highs/lows or using clustering algorithms
            
            # Find potential supports (local minima in lows)
            diffs = window_data[low_col].diff()
            potential_supports = window_data[
                (diffs.shift(-1) > 0) & (diffs < 0)
            ][low_col].tolist()
            
            # Find potential resistances (local maxima in highs)
            diffs = window_data[high_col].diff()
            potential_resistances = window_data[
                (diffs.shift(-1) < 0) & (diffs > 0)
            ][high_col].tolist()
            
            # Count "touches" of each level
            support_levels = {}
            resistance_levels = {}
            
            # Define price proximity threshold (0.5% of current price)
            current_price = df.loc[df.index[i], close_col]
            threshold = current_price * 0.005
            
            # Count support touches
            for low_price in window_data[low_col]:
                for level in potential_supports:
                    if abs(low_price - level) < threshold:
                        support_levels[level] = support_levels.get(level, 0) + 1
            
            # Count resistance touches
            for high_price in window_data[high_col]:
                for level in potential_resistances:
                    if abs(high_price - level) < threshold:
                        resistance_levels[level] = resistance_levels.get(level, 0) + 1
            
            # Filter levels by minimum touches
            valid_supports = [level for level, touches in support_levels.items() if touches >= min_touches]
            valid_resistances = [level for level, touches in resistance_levels.items() if touches >= min_touches]
            
            # Find nearest support and resistance
            if valid_supports:
                nearest_support = max([s for s in valid_supports if s < current_price], default=None)
                if nearest_support:
                    df.loc[df.index[i], "support_level"] = nearest_support
                    # Check if price is at support
                    if abs(current_price - nearest_support) / current_price < 0.01:
                        df.loc[df.index[i], "at_support"] = 1
            
            if valid_resistances:
                nearest_resistance = min([r for r in valid_resistances if r > current_price], default=None)
                if nearest_resistance:
                    df.loc[df.index[i], "resistance_level"] = nearest_resistance
                    # Check if price is at resistance
                    if abs(current_price - nearest_resistance) / current_price < 0.01:
                        df.loc[df.index[i], "at_resistance"] = 1
        
        # Forward fill level values
        df[["support_level", "resistance_level"]] = df[["support_level", "resistance_level"]].fillna(method="ffill")
        
        # Calculate distance to support/resistance (percentage)
        df["support_distance"] = (df[close_col] - df["support_level"]) / df["support_level"] * 100
        df["resistance_distance"] = (df["resistance_level"] - df[close_col]) / df[close_col] * 100
        
        # Calculate support/resistance strength (risk/reward ratio)
        df["sr_ratio"] = abs(df["support_distance"]) / abs(df["resistance_distance"])
        
        return df
    
    def _add_vpin(self, 
                 df: pd.DataFrame, 
                 price_cols: Dict[str, str], 
                 num_buckets: int = 50,
                 window: int = 50) -> pd.DataFrame:
        """
        Add Volume-Synchronized Probability of Informed Trading (VPIN) features.
        
        Args:
            df: DataFrame with OHLCV data
            price_cols: Dictionary mapping standard column names to actual column names
            num_buckets: Number of volume buckets for VPIN calculation
            window: Rolling window size for VPIN calculation
        
        Returns:
            DataFrame with added VPIN features
        """
        # Ensure required columns are available
        required_cols = ["open", "close", "volume"]
        for col in required_cols:
            if col not in price_cols:
                logger.warning(f"Required column {col} not found. Cannot calculate VPIN.")
                return df
        
        open_col = price_cols["open"]
        close_col = price_cols["close"]
        volume_col = price_cols["volume"]
        
        # Calculate price change
        df["price_change"] = df[close_col] - df[open_col]
        
        # Calculate total volume
        total_volume = df[volume_col].sum()
        
        # Calculate bucket size (volume per bucket)
        bucket_size = total_volume / num_buckets
        
        # Initialize columns
        df["bucket_num"] = 0
        df["bucket_volume"] = 0
        df["buy_volume"] = 0
        df["sell_volume"] = 0
        
        # Assign bars to buckets based on cumulative volume
        current_bucket = 1
        current_bucket_volume = 0
        
        for i in range(len(df)):
            # Add volume to current bucket
            volume = df.loc[df.index[i], volume_col]
            df.loc[df.index[i], "bucket_num"] = current_bucket
            
            # Calculate buy/sell volume based on price change
            price_change = df.loc[df.index[i], "price_change"]
            price_change_ratio = price_change / df.loc[df.index[i], open_col]
            
            if price_change >= 0:
                buy_ratio = 0.5 + abs(price_change_ratio) / 2
                buy_ratio = min(buy_ratio, 1.0)
            else:
                buy_ratio = 0.5 - abs(price_change_ratio) / 2
                buy_ratio = max(buy_ratio, 0.0)
            
            buy_volume = volume * buy_ratio
            sell_volume = volume * (1 - buy_ratio)
            
            df.loc[df.index[i], "buy_volume"] = buy_volume
            df.loc[df.index[i], "sell_volume"] = sell_volume
            
            # Update bucket volume
            remaining_bucket_space = bucket_size - current_bucket_volume
            
            if volume <= remaining_bucket_space:
                # Volume fits in current bucket
                df.loc[df.index[i], "bucket_volume"] = current_bucket_volume + volume
                current_bucket_volume += volume
            else:
                # Volume spans multiple buckets
                df.loc[df.index[i], "bucket_volume"] = bucket_size
                
                # Move to next bucket
                current_bucket += 1
                current_bucket_volume = volume - remaining_bucket_space
        
        # Calculate VPIN
        df["vpin"] = np.nan
        
        # Group by bucket and aggregate
        bucket_data = df.groupby("bucket_num").agg({
            "buy_volume": "sum",
            "sell_volume": "sum",
            "volume": "sum"
        }).reset_index()
        
        # Calculate absolute imbalance for each bucket
        bucket_data["imbalance"] = abs(bucket_data["buy_volume"] - bucket_data["sell_volume"])
        
        # Calculate rolling VPIN
        for i in range(window, len(bucket_data) + 1):
            window_data = bucket_data.iloc[i-window:i]
            vpin_value = window_data["imbalance"].sum() / (window_data["volume"].sum())
            
            # Assign VPIN value to all rows in the last bucket of the window
            last_bucket = window_data.iloc[-1]["bucket_num"]
            df.loc[df["bucket_num"] == last_bucket, "vpin"] = vpin_value
        
        # Forward fill VPIN values
        df["vpin"] = df["vpin"].fillna(method="ffill")
        
        # Add VPIN signals
        df["vpin_high"] = np.where(df["vpin"] > df["vpin"].rolling(window=50).mean() * 1.5, 1, 0)
        df["vpin_spike"] = np.where(
            (df["vpin"] > df["vpin"].shift(1) * 1.2) & 
            (df["vpin"] > df["vpin"].rolling(window=20).mean() * 1.3), 
            1, 0
        )
        
        # Drop intermediate columns
        df = df.drop(columns=["price_change", "bucket_num", "bucket_volume", "buy_volume", "sell_volume"])
        
        return df