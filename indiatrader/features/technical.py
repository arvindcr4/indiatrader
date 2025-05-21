"""
Technical feature engineering for market data.
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


class TechnicalFeatureGenerator:
    """
    Generate technical features from market data.
    """
    
    def __init__(self):
        """
        Initialize technical feature generator.
        """
        pass
    
    def generate_features(self, df: pd.DataFrame, feature_config: Dict) -> pd.DataFrame:
        """
        Generate technical features based on configuration.
        
        Args:
            df: DataFrame with OHLCV data
            feature_config: Feature configuration dictionary
        
        Returns:
            DataFrame with added technical features
        """
        # Make a copy to avoid modifying the original DataFrame
        result_df = df.copy()
        
        # Extract column names for OHLCV
        price_cols = self._get_price_columns(result_df)
        
        # Generate features for each configured feature type
        for feature_spec in feature_config:
            feature_type = feature_spec["name"]
            params = feature_spec.get("params", {})
            
            if feature_type == "moving_averages":
                result_df = self._add_moving_averages(result_df, price_cols, **params)
            
            elif feature_type == "rsi":
                result_df = self._add_rsi(result_df, price_cols, **params)
            
            elif feature_type == "bollinger_bands":
                result_df = self._add_bollinger_bands(result_df, price_cols, **params)
            
            elif feature_type == "macd":
                result_df = self._add_macd(result_df, price_cols, **params)
            
            elif feature_type == "stochastic":
                result_df = self._add_stochastic(result_df, price_cols, **params)
            
            elif feature_type == "atr":
                result_df = self._add_atr(result_df, price_cols, **params)
            
            elif feature_type == "fibonacci_retracement":
                result_df = self._add_fibonacci_retracement(result_df, price_cols, **params)
            
            elif feature_type == "ichimoku":
                result_df = self._add_ichimoku(result_df, price_cols, **params)
            
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
        required_cols = ["open", "high", "low", "close"]
        for col in required_cols:
            if col not in price_cols:
                missing_col = col
                # Try to infer if possible
                if missing_col == "open" and "close" in price_cols:
                    logger.warning(f"Open column not found. Using close column as substitute.")
                    price_cols["open"] = price_cols["close"]
                else:
                    logger.warning(f"Required column not found: {col}")
        
        return price_cols
    
    def _add_moving_averages(self, 
                            df: pd.DataFrame, 
                            price_cols: Dict[str, str], 
                            windows: List[int] = [5, 10, 20, 50, 200]) -> pd.DataFrame:
        """
        Add moving average features.
        
        Args:
            df: DataFrame with OHLCV data
            price_cols: Dictionary mapping standard column names to actual column names
            windows: List of window sizes for moving averages
        
        Returns:
            DataFrame with added moving average features
        """
        if "close" not in price_cols:
            logger.warning("Close column not found. Cannot calculate moving averages.")
            return df
        
        close_col = price_cols["close"]
        
        # Calculate simple moving averages
        for window in windows:
            df[f"sma_{window}"] = df[close_col].rolling(window=window).mean()
            
            # Calculate percentage difference from current price
            df[f"sma_{window}_pct"] = (df[close_col] - df[f"sma_{window}"]) / df[f"sma_{window}"] * 100
        
        # Calculate exponential moving averages
        for window in windows:
            df[f"ema_{window}"] = df[close_col].ewm(span=window, adjust=False).mean()
            
            # Calculate percentage difference from current price
            df[f"ema_{window}_pct"] = (df[close_col] - df[f"ema_{window}"]) / df[f"ema_{window}"] * 100
        
        # Calculate moving average crossovers for adjacent window pairs
        for i in range(len(windows) - 1):
            short_window = windows[i]
            long_window = windows[i + 1]
            
            # SMA crossover
            df[f"sma_{short_window}_{long_window}_crossover"] = np.where(
                df[f"sma_{short_window}"] > df[f"sma_{long_window}"], 1, 
                np.where(df[f"sma_{short_window}"] < df[f"sma_{long_window}"], -1, 0)
            )
            
            # EMA crossover
            df[f"ema_{short_window}_{long_window}_crossover"] = np.where(
                df[f"ema_{short_window}"] > df[f"ema_{long_window}"], 1, 
                np.where(df[f"ema_{short_window}"] < df[f"ema_{long_window}"], -1, 0)
            )
        
        return df
    
    def _add_rsi(self, 
                df: pd.DataFrame, 
                price_cols: Dict[str, str], 
                window: int = 14) -> pd.DataFrame:
        """
        Add Relative Strength Index (RSI) features.
        
        Args:
            df: DataFrame with OHLCV data
            price_cols: Dictionary mapping standard column names to actual column names
            window: Window size for RSI calculation
        
        Returns:
            DataFrame with added RSI features
        """
        if "close" not in price_cols:
            logger.warning("Close column not found. Cannot calculate RSI.")
            return df
        
        close_col = price_cols["close"]
        
        # Calculate price changes
        delta = df[close_col].diff()
        
        # Separate gains and losses
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)
        
        # Calculate average gain and average loss
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        # Calculate relative strength and RSI
        rs = avg_gain / avg_loss
        df["rsi"] = 100 - (100 / (1 + rs))
        
        # Add RSI-based signals
        df["rsi_overbought"] = np.where(df["rsi"] > 70, 1, 0)
        df["rsi_oversold"] = np.where(df["rsi"] < 30, 1, 0)
        
        # RSI divergence (placeholder - full divergence detection requires more complex logic)
        # This is a simple implementation that detects potential divergence conditions
        df["price_higher_high"] = df[close_col] > df[close_col].shift(1)
        df["rsi_lower_high"] = df["rsi"] < df["rsi"].shift(1)
        df["bearish_divergence"] = (df["price_higher_high"] & df["rsi_lower_high"]).astype(int)
        
        df["price_lower_low"] = df[close_col] < df[close_col].shift(1)
        df["rsi_higher_low"] = df["rsi"] > df["rsi"].shift(1)
        df["bullish_divergence"] = (df["price_lower_low"] & df["rsi_higher_low"]).astype(int)
        
        return df
    
    def _add_bollinger_bands(self, 
                            df: pd.DataFrame, 
                            price_cols: Dict[str, str], 
                            window: int = 20, 
                            num_std: float = 2.0) -> pd.DataFrame:
        """
        Add Bollinger Bands features.
        
        Args:
            df: DataFrame with OHLCV data
            price_cols: Dictionary mapping standard column names to actual column names
            window: Window size for moving average
            num_std: Number of standard deviations for bands
        
        Returns:
            DataFrame with added Bollinger Bands features
        """
        if "close" not in price_cols:
            logger.warning("Close column not found. Cannot calculate Bollinger Bands.")
            return df
        
        close_col = price_cols["close"]
        
        # Calculate middle band (simple moving average)
        df["bb_middle"] = df[close_col].rolling(window=window).mean()
        
        # Calculate standard deviation
        df["bb_std"] = df[close_col].rolling(window=window).std()
        
        # Calculate upper and lower bands
        df["bb_upper"] = df["bb_middle"] + (df["bb_std"] * num_std)
        df["bb_lower"] = df["bb_middle"] - (df["bb_std"] * num_std)
        
        # Calculate bandwidth and %B
        df["bb_bandwidth"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        df["bb_percent_b"] = (df[close_col] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
        
        # Add band cross signals
        df["price_above_upper"] = np.where(df[close_col] > df["bb_upper"], 1, 0)
        df["price_below_lower"] = np.where(df[close_col] < df["bb_lower"], 1, 0)
        
        # Add squeeze conditions (tight bands)
        df["bb_squeeze"] = np.where(df["bb_bandwidth"] < df["bb_bandwidth"].rolling(window=50).quantile(0.2), 1, 0)
        
        return df
    
    def _add_macd(self, 
                df: pd.DataFrame, 
                price_cols: Dict[str, str], 
                fast: int = 12, 
                slow: int = 26, 
                signal: int = 9) -> pd.DataFrame:
        """
        Add Moving Average Convergence Divergence (MACD) features.
        
        Args:
            df: DataFrame with OHLCV data
            price_cols: Dictionary mapping standard column names to actual column names
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
        
        Returns:
            DataFrame with added MACD features
        """
        if "close" not in price_cols:
            logger.warning("Close column not found. Cannot calculate MACD.")
            return df
        
        close_col = price_cols["close"]
        
        # Calculate fast and slow EMAs
        ema_fast = df[close_col].ewm(span=fast, adjust=False).mean()
        ema_slow = df[close_col].ewm(span=slow, adjust=False).mean()
        
        # Calculate MACD line and signal line
        df["macd_line"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd_line"].ewm(span=signal, adjust=False).mean()
        
        # Calculate MACD histogram
        df["macd_histogram"] = df["macd_line"] - df["macd_signal"]
        
        # Calculate normalized MACD (for comparing across different price scales)
        df["macd_normalized"] = df["macd_line"] / df[close_col] * 100
        
        # Add MACD crossover signals
        df["macd_signal_crossover"] = np.where(
            df["macd_line"] > df["macd_signal"], 1, 
            np.where(df["macd_line"] < df["macd_signal"], -1, 0)
        )
        
        # Add zero line crossover signals
        df["macd_zero_crossover"] = np.where(
            df["macd_line"] > 0, 1, 
            np.where(df["macd_line"] < 0, -1, 0)
        )
        
        # Detect MACD divergence (placeholder - simplified version)
        df["price_higher_high"] = df[close_col] > df[close_col].shift(1)
        df["macd_lower_high"] = df["macd_line"] < df["macd_line"].shift(1)
        df["macd_bearish_divergence"] = (df["price_higher_high"] & df["macd_lower_high"]).astype(int)
        
        df["price_lower_low"] = df[close_col] < df[close_col].shift(1)
        df["macd_higher_low"] = df["macd_line"] > df["macd_line"].shift(1)
        df["macd_bullish_divergence"] = (df["price_lower_low"] & df["macd_higher_low"]).astype(int)
        
        return df
    
    def _add_stochastic(self, 
                       df: pd.DataFrame, 
                       price_cols: Dict[str, str], 
                       k_period: int = 14, 
                       d_period: int = 3, 
                       slowing: int = 3) -> pd.DataFrame:
        """
        Add Stochastic Oscillator features.
        
        Args:
            df: DataFrame with OHLCV data
            price_cols: Dictionary mapping standard column names to actual column names
            k_period: K period (lookback period)
            d_period: D period (signal line period)
            slowing: Slowing period
        
        Returns:
            DataFrame with added Stochastic Oscillator features
        """
        # Ensure required columns are available
        required_cols = ["high", "low", "close"]
        for col in required_cols:
            if col not in price_cols:
                logger.warning(f"Required column not found: {col}. Cannot calculate Stochastic Oscillator.")
                return df
        
        high_col = price_cols["high"]
        low_col = price_cols["low"]
        close_col = price_cols["close"]
        
        # Calculate %K (raw stochastic)
        lowest_low = df[low_col].rolling(window=k_period).min()
        highest_high = df[high_col].rolling(window=k_period).max()
        raw_k = 100 * (df[close_col] - lowest_low) / (highest_high - lowest_low)
        
        # Calculate %K (with slowing)
        df["stoch_k"] = raw_k.rolling(window=slowing).mean()
        
        # Calculate %D
        df["stoch_d"] = df["stoch_k"].rolling(window=d_period).mean()
        
        # Add overbought/oversold signals
        df["stoch_overbought"] = np.where(df["stoch_k"] > 80, 1, 0)
        df["stoch_oversold"] = np.where(df["stoch_k"] < 20, 1, 0)
        
        # Add crossover signals
        df["stoch_crossover"] = np.where(
            df["stoch_k"] > df["stoch_d"], 1, 
            np.where(df["stoch_k"] < df["stoch_d"], -1, 0)
        )
        
        return df
    
    def _add_atr(self, 
                df: pd.DataFrame, 
                price_cols: Dict[str, str], 
                window: int = 14) -> pd.DataFrame:
        """
        Add Average True Range (ATR) features.
        
        Args:
            df: DataFrame with OHLCV data
            price_cols: Dictionary mapping standard column names to actual column names
            window: Window size for ATR calculation
        
        Returns:
            DataFrame with added ATR features
        """
        # Ensure required columns are available
        required_cols = ["high", "low", "close"]
        for col in required_cols:
            if col not in price_cols:
                logger.warning(f"Required column not found: {col}. Cannot calculate ATR.")
                return df
        
        high_col = price_cols["high"]
        low_col = price_cols["low"]
        close_col = price_cols["close"]
        
        # Calculate true range
        tr1 = df[high_col] - df[low_col]
        tr2 = abs(df[high_col] - df[close_col].shift(1))
        tr3 = abs(df[low_col] - df[close_col].shift(1))
        
        df["true_range"] = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
        
        # Calculate ATR
        df["atr"] = df["true_range"].rolling(window=window).mean()
        
        # Calculate normalized ATR (ATR as percentage of price)
        df["atr_percent"] = df["atr"] / df[close_col] * 100
        
        # Add volatility signals
        df["atr_high_volatility"] = np.where(df["atr_percent"] > df["atr_percent"].rolling(window=50).mean() * 1.5, 1, 0)
        df["atr_low_volatility"] = np.where(df["atr_percent"] < df["atr_percent"].rolling(window=50).mean() * 0.5, 1, 0)
        
        return df
    
    def _add_fibonacci_retracement(self, 
                                 df: pd.DataFrame, 
                                 price_cols: Dict[str, str], 
                                 window: int = 100) -> pd.DataFrame:
        """
        Add Fibonacci Retracement levels.
        
        Args:
            df: DataFrame with OHLCV data
            price_cols: Dictionary mapping standard column names to actual column names
            window: Window size for high/low detection
        
        Returns:
            DataFrame with added Fibonacci Retracement features
        """
        # Ensure required columns are available
        if "close" not in price_cols:
            logger.warning("Close column not found. Cannot calculate Fibonacci Retracement levels.")
            return df
        
        close_col = price_cols["close"]
        
        # Define Fibonacci ratios
        fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
        
        # Calculate rolling high and low
        rolling_high = df[close_col].rolling(window=window).max()
        rolling_low = df[close_col].rolling(window=window).min()
        
        # Calculate range
        price_range = rolling_high - rolling_low
        
        # Calculate Fibonacci levels (both retracements and extensions)
        for ratio in fib_ratios:
            # Retracement levels (from high to low)
            df[f"fib_retr_{ratio:.3f}"] = rolling_high - (price_range * ratio)
            
            # Extension levels (projecting beyond the high)
            df[f"fib_ext_{ratio:.3f}"] = rolling_high + (price_range * ratio)
        
        # Add signals for when price crosses Fibonacci levels
        for ratio in fib_ratios:
            retr_level = f"fib_retr_{ratio:.3f}"
            df[f"{retr_level}_cross"] = np.where(
                (df[close_col] > df[retr_level]) & (df[close_col].shift(1) <= df[retr_level]), 1,
                np.where((df[close_col] < df[retr_level]) & (df[close_col].shift(1) >= df[retr_level]), -1, 0)
            )
        
        return df
    
    def _add_ichimoku(self, 
                     df: pd.DataFrame, 
                     price_cols: Dict[str, str], 
                     tenkan_period: int = 9, 
                     kijun_period: int = 26, 
                     senkou_b_period: int = 52) -> pd.DataFrame:
        """
        Add Ichimoku Cloud features.
        
        Args:
            df: DataFrame with OHLCV data
            price_cols: Dictionary mapping standard column names to actual column names
            tenkan_period: Tenkan-sen (Conversion Line) period
            kijun_period: Kijun-sen (Base Line) period
            senkou_b_period: Senkou Span B period
        
        Returns:
            DataFrame with added Ichimoku Cloud features
        """
        # Ensure required columns are available
        required_cols = ["high", "low", "close"]
        for col in required_cols:
            if col not in price_cols:
                logger.warning(f"Required column not found: {col}. Cannot calculate Ichimoku Cloud.")
                return df
        
        high_col = price_cols["high"]
        low_col = price_cols["low"]
        close_col = price_cols["close"]
        
        # Calculate Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        nine_period_high = df[high_col].rolling(window=tenkan_period).max()
        nine_period_low = df[low_col].rolling(window=tenkan_period).min()
        df["ichimoku_tenkan_sen"] = (nine_period_high + nine_period_low) / 2
        
        # Calculate Kijun-sen (Base Line): (26-period high + 26-period low)/2
        twenty_six_period_high = df[high_col].rolling(window=kijun_period).max()
        twenty_six_period_low = df[low_col].rolling(window=kijun_period).min()
        df["ichimoku_kijun_sen"] = (twenty_six_period_high + twenty_six_period_low) / 2
        
        # Calculate Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
        df["ichimoku_senkou_span_a"] = (df["ichimoku_tenkan_sen"] + df["ichimoku_kijun_sen"]) / 2
        
        # Calculate Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
        fifty_two_period_high = df[high_col].rolling(window=senkou_b_period).max()
        fifty_two_period_low = df[low_col].rolling(window=senkou_b_period).min()
        df["ichimoku_senkou_span_b"] = (fifty_two_period_high + fifty_two_period_low) / 2
        
        # Shift Senkou Span A and B forward by 26 periods (to create cloud)
        df["ichimoku_senkou_span_a_future"] = df["ichimoku_senkou_span_a"].shift(-kijun_period)
        df["ichimoku_senkou_span_b_future"] = df["ichimoku_senkou_span_b"].shift(-kijun_period)
        
        # Calculate Chikou Span (Lagging Span): Current closing price shifted back 26 periods
        df["ichimoku_chikou_span"] = df[close_col].shift(-kijun_period)
        
        # Add cloud signals
        df["price_above_cloud"] = np.where(
            (df[close_col] > df["ichimoku_senkou_span_a"]) & (df[close_col] > df["ichimoku_senkou_span_b"]), 1, 0
        )
        
        df["price_below_cloud"] = np.where(
            (df[close_col] < df["ichimoku_senkou_span_a"]) & (df[close_col] < df["ichimoku_senkou_span_b"]), 1, 0
        )
        
        df["price_in_cloud"] = np.where(
            ~df["price_above_cloud"].astype(bool) & ~df["price_below_cloud"].astype(bool), 1, 0
        )
        
        # Add Tenkan/Kijun cross signals
        df["tk_cross"] = np.where(
            (df["ichimoku_tenkan_sen"] > df["ichimoku_kijun_sen"]) & 
            (df["ichimoku_tenkan_sen"].shift(1) <= df["ichimoku_kijun_sen"].shift(1)), 1,
            np.where(
                (df["ichimoku_tenkan_sen"] < df["ichimoku_kijun_sen"]) & 
                (df["ichimoku_tenkan_sen"].shift(1) >= df["ichimoku_kijun_sen"].shift(1)), -1, 0
            )
        )
        
        return df