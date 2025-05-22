"""
Common trading strategies for backtesting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any

# Import technical indicators
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False


def moving_average_crossover(context: Dict[str, Any], 
                           data: pd.DataFrame, 
                           params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Moving average crossover strategy.
    
    Args:
        context: Trading context dictionary
        data: Market data DataFrame
        params: Strategy parameters (fast_ma, slow_ma, ma_type)
    
    Returns:
        Signal dictionary
    """
    # Get parameters with defaults
    fast_ma = params.get("fast_ma", 20)
    slow_ma = params.get("slow_ma", 50)
    ma_type = params.get("ma_type", "sma")
    
    # Get price series
    price_col = params.get("price_col", "close")
    prices = data[price_col]
    
    # Check if we have enough data
    if len(prices) < slow_ma + 1:
        return {"direction": 0}
    
    # Calculate moving averages
    if ma_type.lower() == "sma":
        fast_ma_values = prices.rolling(window=fast_ma).mean()
        slow_ma_values = prices.rolling(window=slow_ma).mean()
    elif ma_type.lower() == "ema":
        fast_ma_values = prices.ewm(span=fast_ma, adjust=False).mean()
        slow_ma_values = prices.ewm(span=slow_ma, adjust=False).mean()
    else:
        raise ValueError(f"Unknown MA type: {ma_type}")
    
    # Check for crossover
    current_fast = fast_ma_values.iloc[-1]
    current_slow = slow_ma_values.iloc[-1]
    prev_fast = fast_ma_values.iloc[-2]
    prev_slow = slow_ma_values.iloc[-2]
    
    # Generate signals
    if prev_fast <= prev_slow and current_fast > current_slow:
        # Bullish crossover
        return {
            "direction": 1,
            "sizing_params": {"percent": params.get("position_size", 1.0)}
        }
    elif prev_fast >= prev_slow and current_fast < current_slow:
        # Bearish crossover
        return {
            "direction": -1,
            "sizing_params": {"percent": params.get("position_size", 1.0)}
        }
    else:
        # No crossover
        return {"direction": 0}


def rsi_strategy(context: Dict[str, Any], 
                data: pd.DataFrame, 
                params: Dict[str, Any]) -> Dict[str, Any]:
    """
    RSI-based mean reversion strategy.
    
    Args:
        context: Trading context dictionary
        data: Market data DataFrame
        params: Strategy parameters (rsi_period, oversold, overbought)
    
    Returns:
        Signal dictionary
    """
    # Get parameters with defaults
    rsi_period = params.get("rsi_period", 14)
    oversold = params.get("oversold", 30)
    overbought = params.get("overbought", 70)
    
    # Get price series
    price_col = params.get("price_col", "close")
    prices = data[price_col]
    
    # Check if we have enough data
    if len(prices) < rsi_period + 1:
        return {"direction": 0}
    
    # Calculate RSI
    if TALIB_AVAILABLE:
        rsi = talib.RSI(prices.values, timeperiod=rsi_period)
    else:
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)
        
        # Calculate average gain and average loss
        avg_gain = gain.rolling(window=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period).mean()
        
        # Calculate relative strength and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
    
    # Get current RSI
    current_rsi = rsi[-1] if TALIB_AVAILABLE else rsi.iloc[-1]
    
    # Generate signals
    if current_rsi < oversold:
        # Oversold - buy signal
        return {
            "direction": 1,
            "sizing_params": {"percent": params.get("position_size", 1.0)}
        }
    elif current_rsi > overbought:
        # Overbought - sell signal
        return {
            "direction": -1,
            "sizing_params": {"percent": params.get("position_size", 1.0)}
        }
    else:
        # No signal
        return {"direction": 0}


def bollinger_band_strategy(context: Dict[str, Any], 
                          data: pd.DataFrame, 
                          params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Bollinger Band mean reversion strategy.
    
    Args:
        context: Trading context dictionary
        data: Market data DataFrame
        params: Strategy parameters (bb_period, num_std)
    
    Returns:
        Signal dictionary
    """
    # Get parameters with defaults
    bb_period = params.get("bb_period", 20)
    num_std = params.get("num_std", 2.0)
    
    # Get price series
    price_col = params.get("price_col", "close")
    prices = data[price_col]
    
    # Check if we have enough data
    if len(prices) < bb_period + 1:
        return {"direction": 0}
    
    # Calculate Bollinger Bands
    if TALIB_AVAILABLE:
        upper, middle, lower = talib.BBANDS(
            prices.values, 
            timeperiod=bb_period, 
            nbdevup=num_std, 
            nbdevdn=num_std
        )
    else:
        # Calculate middle band (simple moving average)
        middle = prices.rolling(window=bb_period).mean()
        
        # Calculate standard deviation
        std = prices.rolling(window=bb_period).std()
        
        # Calculate upper and lower bands
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
    
    # Get current price and bands
    current_price = prices.iloc[-1]
    current_upper = upper[-1] if TALIB_AVAILABLE else upper.iloc[-1]
    current_lower = lower[-1] if TALIB_AVAILABLE else lower.iloc[-1]
    
    # Generate signals
    if current_price <= current_lower:
        # Price below lower band - buy signal
        return {
            "direction": 1,
            "sizing_params": {"percent": params.get("position_size", 1.0)}
        }
    elif current_price >= current_upper:
        # Price above upper band - sell signal
        return {
            "direction": -1,
            "sizing_params": {"percent": params.get("position_size", 1.0)}
        }
    else:
        # No signal
        return {"direction": 0}


def macd_strategy(context: Dict[str, Any], 
                 data: pd.DataFrame, 
                 params: Dict[str, Any]) -> Dict[str, Any]:
    """
    MACD crossover strategy.
    
    Args:
        context: Trading context dictionary
        data: Market data DataFrame
        params: Strategy parameters (fast_period, slow_period, signal_period)
    
    Returns:
        Signal dictionary
    """
    # Get parameters with defaults
    fast_period = params.get("fast_period", 12)
    slow_period = params.get("slow_period", 26)
    signal_period = params.get("signal_period", 9)
    
    # Get price series
    price_col = params.get("price_col", "close")
    prices = data[price_col]
    
    # Check if we have enough data
    if len(prices) < slow_period + signal_period + 1:
        return {"direction": 0}
    
    # Calculate MACD
    if TALIB_AVAILABLE:
        macd, signal, hist = talib.MACD(
            prices.values, 
            fastperiod=fast_period, 
            slowperiod=slow_period, 
            signalperiod=signal_period
        )
    else:
        # Calculate fast and slow EMAs
        ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
        ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line and signal line
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        hist = macd - signal
    
    # Get current and previous MACD and signal values
    current_macd = macd[-1] if TALIB_AVAILABLE else macd.iloc[-1]
    current_signal = signal[-1] if TALIB_AVAILABLE else signal.iloc[-1]
    prev_macd = macd[-2] if TALIB_AVAILABLE else macd.iloc[-2]
    prev_signal = signal[-2] if TALIB_AVAILABLE else signal.iloc[-2]
    
    # Generate signals
    if prev_macd <= prev_signal and current_macd > current_signal:
        # Bullish crossover
        return {
            "direction": 1,
            "sizing_params": {"percent": params.get("position_size", 1.0)}
        }
    elif prev_macd >= prev_signal and current_macd < current_signal:
        # Bearish crossover
        return {
            "direction": -1,
            "sizing_params": {"percent": params.get("position_size", 1.0)}
        }
    else:
        # No signal
        return {"direction": 0}


def breakout_strategy(context: Dict[str, Any], 
                    data: pd.DataFrame, 
                    params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Breakout strategy based on price range.
    
    Args:
        context: Trading context dictionary
        data: Market data DataFrame
        params: Strategy parameters (period, atr_multiplier)
    
    Returns:
        Signal dictionary
    """
    # Get parameters with defaults
    period = params.get("period", 20)
    atr_multiplier = params.get("atr_multiplier", 1.0)
    
    # Get price series
    high_col = params.get("high_col", "high")
    low_col = params.get("low_col", "low")
    close_col = params.get("close_col", "close")
    
    # Check if we have enough data
    if len(data) < period + 1:
        return {"direction": 0}
    
    # Calculate breakout levels
    highest_high = data[high_col].rolling(window=period).max().iloc[-2]
    lowest_low = data[low_col].rolling(window=period).min().iloc[-2]
    
    # Calculate ATR if available in data or using talib
    if "atr" in data.columns:
        atr = data["atr"].iloc[-1]
    elif TALIB_AVAILABLE:
        atr = talib.ATR(
            data[high_col].values,
            data[low_col].values,
            data[close_col].values,
            timeperiod=period
        )[-1]
    else:
        # Calculate ATR manually
        high = data[high_col].values
        low = data[low_col].values
        close = data[close_col].values
        
        tr1 = high[-period:] - low[-period:]
        tr2 = np.abs(high[-period:] - np.roll(close, 1)[-period:])
        tr3 = np.abs(low[-period:] - np.roll(close, 1)[-period:])
        
        tr = np.vstack([tr1, tr2, tr3]).max(axis=0)
        atr = np.mean(tr)
    
    # Adjust breakout levels with ATR
    upper_level = highest_high + atr * atr_multiplier
    lower_level = lowest_low - atr * atr_multiplier
    
    # Get current price
    current_price = data[close_col].iloc[-1]
    
    # Generate signals
    if current_price > upper_level:
        # Price breaks above upper level - buy signal
        return {
            "direction": 1,
            "sizing_params": {"percent": params.get("position_size", 1.0)}
        }
    elif current_price < lower_level:
        # Price breaks below lower level - sell signal
        return {
            "direction": -1,
            "sizing_params": {"percent": params.get("position_size", 1.0)}
        }
    else:
        # No signal
        return {"direction": 0}


def trend_following_strategy(context: Dict[str, Any], 
                           data: pd.DataFrame, 
                           params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Trend following strategy with filters.
    
    Args:
        context: Trading context dictionary
        data: Market data DataFrame
        params: Strategy parameters (fast_ma, slow_ma, adx_period, adx_threshold)
    
    Returns:
        Signal dictionary
    """
    # Get parameters with defaults
    fast_ma = params.get("fast_ma", 20)
    slow_ma = params.get("slow_ma", 50)
    adx_period = params.get("adx_period", 14)
    adx_threshold = params.get("adx_threshold", 25)
    
    # Get price series
    price_col = params.get("price_col", "close")
    high_col = params.get("high_col", "high")
    low_col = params.get("low_col", "low")
    
    # Check if we have enough data
    min_period = max(slow_ma, adx_period) + 1
    if len(data) < min_period:
        return {"direction": 0}
    
    # Calculate moving averages
    fast_ma_values = data[price_col].rolling(window=fast_ma).mean()
    slow_ma_values = data[price_col].rolling(window=slow_ma).mean()
    
    # Calculate ADX
    if TALIB_AVAILABLE:
        adx = talib.ADX(
            data[high_col].values,
            data[low_col].values,
            data[price_col].values,
            timeperiod=adx_period
        )
    else:
        # This is a simplified ADX calculation
        # For a proper ADX, use talib or implement the full calculation
        # Calculate True Range
        tr1 = data[high_col] - data[low_col]
        tr2 = abs(data[high_col] - data[price_col].shift(1))
        tr3 = abs(data[low_col] - data[price_col].shift(1))
        tr = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
        
        # Calculate directional movement
        plus_dm = data[high_col].diff()
        minus_dm = -data[low_col].diff()
        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
        
        # Calculate smoothed values
        smoothed_tr = tr.rolling(window=adx_period).mean()
        smoothed_plus_dm = pd.Series(plus_dm).rolling(window=adx_period).mean()
        smoothed_minus_dm = pd.Series(minus_dm).rolling(window=adx_period).mean()
        
        # Calculate directional indexes
        plus_di = 100 * smoothed_plus_dm / smoothed_tr
        minus_di = 100 * smoothed_minus_dm / smoothed_tr
        
        # Calculate directional movement index
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # Calculate ADX
        adx = dx.rolling(window=adx_period).mean()
    
    # Get current values
    current_fast = fast_ma_values.iloc[-1]
    current_slow = slow_ma_values.iloc[-1]
    current_adx = adx[-1] if TALIB_AVAILABLE else adx.iloc[-1]
    
    # Check if there's a trend
    trend_strength = current_adx > adx_threshold
    
    # Generate signals
    if trend_strength:
        if current_fast > current_slow:
            # Strong uptrend
            return {
                "direction": 1,
                "sizing_params": {"percent": params.get("position_size", 1.0)}
            }
        elif current_fast < current_slow:
            # Strong downtrend
            return {
                "direction": -1,
                "sizing_params": {"percent": params.get("position_size", 1.0)}
            }
    
    # No signal
    return {"direction": 0}


def dual_momentum_strategy(context: Dict[str, Any], 
                         data: pd.DataFrame, 
                         params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dual momentum strategy (absolute and relative momentum).
    
    Args:
        context: Trading context dictionary
        data: Market data DataFrame
        params: Strategy parameters (momentum_period, lookback_period)
    
    Returns:
        Signal dictionary
    """
    # Get parameters with defaults
    momentum_period = params.get("momentum_period", 90)
    lookback_period = params.get("lookback_period", 10)
    
    # Get price series
    price_col = params.get("price_col", "close")
    prices = data[price_col]
    
    # Check if we have enough data
    if len(prices) < momentum_period + 1:
        return {"direction": 0}
    
    # Calculate absolute momentum (based on price change over period)
    current_price = prices.iloc[-1]
    past_price = prices.iloc[-momentum_period]
    absolute_momentum = current_price / past_price - 1
    
    # Calculate relative momentum (based on recent performance)
    recent_change = prices.iloc[-1] / prices.iloc[-lookback_period] - 1
    
    # Generate signals
    if absolute_momentum > 0 and recent_change > 0:
        # Both absolute and relative momentum positive
        return {
            "direction": 1,
            "sizing_params": {"percent": params.get("position_size", 1.0)}
        }
    elif absolute_momentum < 0 and recent_change < 0:
        # Both absolute and relative momentum negative
        return {
            "direction": -1,
            "sizing_params": {"percent": params.get("position_size", 1.0)}
        }
    else:
        # Conflicting signals
        return {"direction": 0}


def vwap_strategy(context: Dict[str, Any], 
                data: pd.DataFrame, 
                params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Strategy based on Volume Weighted Average Price (VWAP).
    
    Args:
        context: Trading context dictionary
        data: Market data DataFrame
        params: Strategy parameters (vwap_period, band_multiplier)
    
    Returns:
        Signal dictionary
    """
    # Get parameters with defaults
    vwap_period = params.get("vwap_period", 30)
    band_multiplier = params.get("band_multiplier", 2.0)
    
    # Get price and volume series
    high_col = params.get("high_col", "high")
    low_col = params.get("low_col", "low")
    close_col = params.get("close_col", "close")
    volume_col = params.get("volume_col", "volume")
    
    # Check if we have required columns
    required_cols = [high_col, low_col, close_col, volume_col]
    for col in required_cols:
        if col not in data.columns:
            return {"direction": 0}
    
    # Check if we have enough data
    if len(data) < vwap_period + 1:
        return {"direction": 0}
    
    # Get recent data for VWAP calculation
    recent_data = data.iloc[-vwap_period:]
    
    # Calculate typical price
    typical_price = (recent_data[high_col] + recent_data[low_col] + recent_data[close_col]) / 3
    
    # Calculate VWAP
    vwap = np.sum(typical_price * recent_data[volume_col]) / np.sum(recent_data[volume_col])
    
    # Calculate standard deviation of price from VWAP
    deviation = np.sqrt(np.sum(recent_data[volume_col] * (typical_price - vwap) ** 2) / np.sum(recent_data[volume_col]))
    
    # Calculate VWAP bands
    upper_band = vwap + deviation * band_multiplier
    lower_band = vwap - deviation * band_multiplier
    
    # Get current price
    current_price = data[close_col].iloc[-1]
    
    # Generate signals
    if current_price <= lower_band:
        # Price below lower band - buy signal
        return {
            "direction": 1,
            "sizing_params": {"percent": params.get("position_size", 1.0)}
        }
    elif current_price >= upper_band:
        # Price above upper band - sell signal
        return {
            "direction": -1,
            "sizing_params": {"percent": params.get("position_size", 1.0)}
        }
    else:
        # No signal
        return {"direction": 0}


def mean_reversion_with_zscore(context: Dict[str, Any], 
                             data: pd.DataFrame, 
                             params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mean reversion strategy based on z-score.
    
    Args:
        context: Trading context dictionary
        data: Market data DataFrame
        params: Strategy parameters (lookback_period, entry_zscore, exit_zscore)
    
    Returns:
        Signal dictionary
    """
    # Get parameters with defaults
    lookback_period = params.get("lookback_period", 20)
    entry_zscore = params.get("entry_zscore", 2.0)
    exit_zscore = params.get("exit_zscore", 0.5)
    
    # Get price series
    price_col = params.get("price_col", "close")
    prices = data[price_col]
    
    # Check if we have enough data
    if len(prices) < lookback_period + 1:
        return {"direction": 0}
    
    # Get recent prices for calculation
    recent_prices = prices.iloc[-lookback_period:]
    
    # Calculate mean and standard deviation
    mean_price = recent_prices.mean()
    std_price = recent_prices.std()
    
    # Calculate z-score
    current_price = prices.iloc[-1]
    zscore = (current_price - mean_price) / std_price if std_price > 0 else 0
    
    # Check current position
    current_position = context.get("position", 0)
    
    # Generate signals
    if current_position == 0:
        # No position - check for entry
        if zscore >= entry_zscore:
            # Price significantly above mean - sell signal
            return {
                "direction": -1,
                "sizing_params": {"percent": params.get("position_size", 1.0)}
            }
        elif zscore <= -entry_zscore:
            # Price significantly below mean - buy signal
            return {
                "direction": 1,
                "sizing_params": {"percent": params.get("position_size", 1.0)}
            }
    elif current_position > 0:
        # Long position - check for exit
        if zscore >= -exit_zscore:
            # Price reverted to or above mean - exit signal
            return {"direction": 0, "close_position": True}
    elif current_position < 0:
        # Short position - check for exit
        if zscore <= exit_zscore:
            # Price reverted to or below mean - exit signal
            return {"direction": 0, "close_position": True}
    
    # No signal
    return {"direction": 0}


def sentiment_based_strategy(context: Dict[str, Any], 
                           data: pd.DataFrame, 
                           params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Strategy based on sentiment indicators.
    
    Args:
        context: Trading context dictionary
        data: Market data DataFrame
        params: Strategy parameters (sentiment_period, threshold)
    
    Returns:
        Signal dictionary
    """
    # Get parameters with defaults
    sentiment_period = params.get("sentiment_period", 5)
    threshold = params.get("threshold", 0.5)
    
    # Check if we have sentiment scores
    if "sentiment_score" not in data.columns and "news_sentiment_score" not in data.columns:
        return {"direction": 0}
    
    # Use appropriate sentiment column
    sentiment_col = "news_sentiment_score" if "news_sentiment_score" in data.columns else "sentiment_score"
    
    # Check if we have enough data
    if len(data) < sentiment_period + 1:
        return {"direction": 0}
    
    # Calculate average sentiment
    avg_sentiment = data[sentiment_col].iloc[-sentiment_period:].mean()
    
    # Generate signals
    if avg_sentiment > threshold:
        # Positive sentiment - buy signal
        return {
            "direction": 1,
            "sizing_params": {"percent": params.get("position_size", 1.0)}
        }
    elif avg_sentiment < -threshold:
        # Negative sentiment - sell signal
        return {
            "direction": -1,
            "sizing_params": {"percent": params.get("position_size", 1.0)}
        }
    else:
        # Neutral sentiment - no signal
        return {"direction": 0}


def volatility_breakout_strategy(context: Dict[str, Any], 
                               data: pd.DataFrame, 
                               params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Volatility-based breakout strategy.
    
    Args:
        context: Trading context dictionary
        data: Market data DataFrame
        params: Strategy parameters (atr_period, atr_multiplier, ma_period)
    
    Returns:
        Signal dictionary
    """
    # Get parameters with defaults
    atr_period = params.get("atr_period", 14)
    atr_multiplier = params.get("atr_multiplier", 2.0)
    ma_period = params.get("ma_period", 20)
    
    # Get price series
    price_col = params.get("price_col", "close")
    high_col = params.get("high_col", "high")
    low_col = params.get("low_col", "low")
    
    # Check if we have enough data
    min_period = max(atr_period, ma_period) + 1
    if len(data) < min_period:
        return {"direction": 0}
    
    # Calculate moving average
    ma = data[price_col].rolling(window=ma_period).mean()
    
    # Calculate ATR
    if "atr" in data.columns:
        atr = data["atr"].iloc[-1]
    elif TALIB_AVAILABLE:
        atr = talib.ATR(
            data[high_col].values,
            data[low_col].values,
            data[price_col].values,
            timeperiod=atr_period
        )[-1]
    else:
        # Calculate ATR manually
        high = data[high_col].values
        low = data[low_col].values
        close = data[price_col].values
        
        tr1 = high[-atr_period:] - low[-atr_period:]
        tr2 = np.abs(high[-atr_period:] - np.roll(close, 1)[-atr_period:])
        tr3 = np.abs(low[-atr_period:] - np.roll(close, 1)[-atr_period:])
        
        tr = np.vstack([tr1, tr2, tr3]).max(axis=0)
        atr = np.mean(tr)
    
    # Calculate breakout levels
    current_ma = ma.iloc[-1]
    upper_level = current_ma + atr * atr_multiplier
    lower_level = current_ma - atr * atr_multiplier
    
    # Get current price
    current_price = data[price_col].iloc[-1]
    
    # Generate signals
    if current_price > upper_level:
        # Price breaks above upper level - buy signal
        return {
            "direction": 1,
            "sizing_params": {"percent": params.get("position_size", 1.0)}
        }
    elif current_price < lower_level:
        # Price breaks below lower level - sell signal
        return {
            "direction": -1,
            "sizing_params": {"percent": params.get("position_size", 1.0)}
        }
    else:
        # No signal
        return {"direction": 0}


def combined_strategy(context: Dict[str, Any], 
                    data: pd.DataFrame, 
                    params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combined strategy using multiple signals.
    
    Args:
        context: Trading context dictionary
        data: Market data DataFrame
        params: Strategy parameters (ma_fast, ma_slow, rsi_period, bb_period)
    
    Returns:
        Signal dictionary
    """
    # Get parameters with defaults
    ma_fast = params.get("ma_fast", 20)
    ma_slow = params.get("ma_slow", 50)
    rsi_period = params.get("rsi_period", 14)
    bb_period = params.get("bb_period", 20)
    
    # Create parameter dictionaries for each strategy
    ma_params = {
        "fast_ma": ma_fast,
        "slow_ma": ma_slow,
        "ma_type": params.get("ma_type", "sma"),
        "price_col": params.get("price_col", "close")
    }
    
    rsi_params = {
        "rsi_period": rsi_period,
        "oversold": params.get("oversold", 30),
        "overbought": params.get("overbought", 70),
        "price_col": params.get("price_col", "close")
    }
    
    bb_params = {
        "bb_period": bb_period,
        "num_std": params.get("num_std", 2.0),
        "price_col": params.get("price_col", "close")
    }
    
    # Get signals from each strategy
    ma_signal = moving_average_crossover(context, data, ma_params)
    rsi_signal = rsi_strategy(context, data, rsi_params)
    bb_signal = bollinger_band_strategy(context, data, bb_params)
    
    # Count the number of buy and sell signals
    buy_signals = sum(1 for signal in [ma_signal, rsi_signal, bb_signal] if signal["direction"] > 0)
    sell_signals = sum(1 for signal in [ma_signal, rsi_signal, bb_signal] if signal["direction"] < 0)
    
    # Calculate signal strength
    signal_strength = buy_signals - sell_signals
    
    # Generate signals
    if signal_strength >= params.get("min_buy_signals", 2):
        # Strong buy signal
        return {
            "direction": 1,
            "sizing_params": {"percent": params.get("position_size", 1.0)}
        }
    elif signal_strength <= -params.get("min_sell_signals", 2):
        # Strong sell signal
        return {
            "direction": -1,
            "sizing_params": {"percent": params.get("position_size", 1.0)}
        }
    else:
        # No strong signal
        return {"direction": 0}