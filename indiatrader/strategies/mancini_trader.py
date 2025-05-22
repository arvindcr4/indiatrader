"""Adam Mancini trading strategy implementation with broker integration.

This module implements a trading system based on Adam Mancini's trading strategies,
with support for both Dhan and ICICI Breeze brokers. It provides the infrastructure
to automatically detect trading signals and execute trades through the supported brokers.
"""

from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
from enum import Enum

from indiatrader.brokers.dhan import DhanClient
from indiatrader.brokers.icici import ICICIBreezeClient
from indiatrader.strategies.adam_mancini import AdamManciniNiftyStrategy, Levels

logger = logging.getLogger(__name__)


class BrokerType(Enum):
    """Supported broker types."""
    
    DHAN = "dhan"
    ICICI = "icici"


class SignalType(Enum):
    """Signal types for trading."""
    
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class ManciniTrader:
    """Trading system based on Adam Mancini's strategies with broker integration."""
    
    def __init__(
        self, 
        broker_type: BrokerType,
        broker_config: Dict[str, Any],
        symbol: str,
        exchange: str,
        product_type: str = None,
        quantity: int = 1,
        open_range_minutes: int = 15,
        is_futures: bool = False,
        expiry_date: str = None,
        backtest_mode: bool = False
    ):
        """Initialize the Mancini Trader.
        
        Parameters
        ----------
        broker_type : BrokerType
            Type of broker to use.
        broker_config : Dict[str, Any]
            Configuration for the broker client.
        symbol : str
            Trading symbol.
        exchange : str
            Exchange code.
        product_type : str, optional
            Product type for orders.
        quantity : int, optional
            Quantity for orders.
        open_range_minutes : int, optional
            Number of minutes for the opening range.
        is_futures : bool, optional
            Whether the symbol is a futures contract.
        expiry_date : str, optional
            Expiry date for futures/options.
        backtest_mode : bool, optional
            Whether to run in backtest mode.
        """
        self.broker_type = broker_type
        self.broker_config = broker_config
        self.symbol = symbol
        self.exchange = exchange
        self.product_type = product_type
        self.quantity = quantity
        self.is_futures = is_futures
        self.expiry_date = expiry_date
        self.backtest_mode = backtest_mode
        
        # Initialize Adam Mancini strategy
        self.strategy = AdamManciniNiftyStrategy(open_range_minutes=open_range_minutes)
        
        # Initialize broker client
        self.broker = self._initialize_broker()
        
        # Trading state
        self.position = 0
        self.trade_history = []
        self.current_signals = None
        self.is_running = False
        self.active_orders = []
        
        logger.info(f"Initialized ManciniTrader for {symbol} on {exchange}")
    
    def _initialize_broker(self):
        """Initialize broker client based on broker type.
        
        Returns
        -------
        Union[DhanClient, ICICIBreezeClient]
            Broker client.
        """
        if self.broker_type == BrokerType.DHAN:
            return DhanClient(
                client_id=self.broker_config.get('client_id'),
                access_token=self.broker_config.get('access_token')
            )
        elif self.broker_type == BrokerType.ICICI:
            return ICICIBreezeClient(
                api_key=self.broker_config.get('api_key'),
                api_secret=self.broker_config.get('api_secret'),
                session_token=self.broker_config.get('session_token')
            )
        else:
            raise ValueError(f"Unsupported broker type: {self.broker_type}")
    
    def _map_product_type(self):
        """Map generic product type to broker-specific product type.
        
        Returns
        -------
        str
            Broker-specific product type.
        """
        if self.broker_type == BrokerType.DHAN:
            product_map = {
                "intraday": self.broker.INTRA,
                "delivery": self.broker.DELIVERY,
                "margin": self.broker.MARGIN,
                "bo": self.broker.BO,
                "co": self.broker.CO,
            }
            return product_map.get(self.product_type.lower(), self.broker.INTRA)
        
        elif self.broker_type == BrokerType.ICICI:
            product_map = {
                "intraday": self.broker.INTRADAY,
                "delivery": self.broker.DELIVERY,
                "margin": self.broker.MARGIN,
                "bo": self.broker.BO,
                "co": self.broker.CO,
            }
            return product_map.get(self.product_type.lower(), self.broker.INTRADAY)
    
    def fetch_historical_data(
        self, 
        days: int = 10, 
        interval: str = "1minute"
    ) -> pd.DataFrame:
        """Fetch historical data for the trading symbol.
        
        Parameters
        ----------
        days : int, optional
            Number of days of data to fetch.
        interval : str, optional
            Data interval.
            
        Returns
        -------
        pd.DataFrame
            Historical OHLC data.
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        try:
            if self.broker_type == BrokerType.DHAN:
                df = self.broker.get_historical_data(
                    security_id=self.symbol,
                    exchange_segment=self.exchange,
                    instrument_type="EQ" if not self.is_futures else "FUT",
                    from_date=start_date.strftime("%Y-%m-%d"),
                    to_date=end_date.strftime("%Y-%m-%d"),
                    interval=interval
                )
            elif self.broker_type == BrokerType.ICICI:
                df = self.broker.get_historical_data(
                    exchange_code=self.exchange,
                    stock_code=self.symbol,
                    interval=interval,
                    from_date=start_date.strftime("%d-%m-%Y"),
                    to_date=end_date.strftime("%d-%m-%Y"),
                    product_type="Futures" if self.is_futures else None,
                    expiry_date=self.expiry_date if self.is_futures else None
                )
            
            # Ensure column names are standardized
            if 'o' in df.columns and 'open' not in df.columns:
                df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}, inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"Failed to fetch historical data: {e}")
            return pd.DataFrame()
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals using the Adam Mancini strategy.
        
        Parameters
        ----------
        data : pd.DataFrame
            Historical OHLC data.
            
        Returns
        -------
        pd.DataFrame
            Data with trading signals added.
        """
        try:
            signals_df = self.strategy.generate_signals(data)
            self.current_signals = signals_df
            return signals_df
        except Exception as e:
            logger.error(f"Failed to generate signals: {e}")
            return pd.DataFrame()
    
    def place_trade(self, signal_type: SignalType, price: float = 0) -> Dict:
        """Place a trade based on signal type.
        
        Parameters
        ----------
        signal_type : SignalType
            Type of trading signal.
        price : float, optional
            Price for limit orders.
            
        Returns
        -------
        Dict
            Trade response.
        """
        if self.backtest_mode:
            return self._place_backtest_trade(signal_type, price)
        
        mapped_product_type = self._map_product_type()
        
        try:
            if self.broker_type == BrokerType.DHAN:
                if signal_type == SignalType.LONG:
                    response = self.broker.place_order(
                        security_id=self.symbol,
                        exchange_segment=self.exchange,
                        transaction_type=self.broker.BUY,
                        quantity=self.quantity,
                        order_type=self.broker.MARKET,
                        product_type=mapped_product_type,
                        price=price
                    )
                    self.position += self.quantity
                elif signal_type == SignalType.SHORT:
                    response = self.broker.place_order(
                        security_id=self.symbol,
                        exchange_segment=self.exchange,
                        transaction_type=self.broker.SELL,
                        quantity=self.quantity,
                        order_type=self.broker.MARKET,
                        product_type=mapped_product_type,
                        price=price
                    )
                    self.position -= self.quantity
                else:
                    logger.warning("Neutral signal, no trade placed")
                    return {"status": "no_action", "message": "Neutral signal"}
            
            elif self.broker_type == BrokerType.ICICI:
                kwargs = {
                    'exchange_code': self.exchange,
                    'stock_code': self.symbol,
                    'quantity': self.quantity,
                    'price_type': self.broker.MARKET,
                    'product_type': mapped_product_type,
                    'price': price
                }
                
                if self.is_futures and self.expiry_date:
                    kwargs['product_type'] = 'Futures'
                    kwargs['expiry_date'] = self.expiry_date
                
                if signal_type == SignalType.LONG:
                    kwargs['action'] = self.broker.BUY
                    response = self.broker.place_order(**kwargs)
                    self.position += self.quantity
                elif signal_type == SignalType.SHORT:
                    kwargs['action'] = self.broker.SELL
                    response = self.broker.place_order(**kwargs)
                    self.position -= self.quantity
                else:
                    logger.warning("Neutral signal, no trade placed")
                    return {"status": "no_action", "message": "Neutral signal"}
            
            self.active_orders.append(response)
            return response
        
        except Exception as e:
            logger.error(f"Failed to place trade: {e}")
            return {"status": "error", "message": str(e)}
    
    def _place_backtest_trade(self, signal_type: SignalType, price: float = 0) -> Dict:
        """Place a backtest trade (simulated).
        
        Parameters
        ----------
        signal_type : SignalType
            Type of trading signal.
        price : float, optional
            Price for the trade.
            
        Returns
        -------
        Dict
            Simulated trade response.
        """
        timestamp = datetime.now()
        
        if signal_type == SignalType.LONG:
            self.position += self.quantity
            action = "BUY"
        elif signal_type == SignalType.SHORT:
            self.position -= self.quantity
            action = "SELL"
        else:
            logger.warning("Neutral signal, no trade placed")
            return {"status": "no_action", "message": "Neutral signal"}
        
        trade = {
            "timestamp": timestamp,
            "symbol": self.symbol,
            "exchange": self.exchange,
            "action": action,
            "quantity": self.quantity,
            "price": price if price > 0 else self._get_current_price(),
            "status": "COMPLETE",
            "order_id": f"BT-{int(timestamp.timestamp())}",
            "position": self.position
        }
        
        self.trade_history.append(trade)
        logger.info(f"Backtest trade placed: {trade}")
        
        return {
            "status": "success",
            "message": "Backtest trade placed",
            "trade": trade
        }
    
    def _get_current_price(self) -> float:
        """Get current price of the symbol.
        
        Returns
        -------
        float
            Current price.
        """
        try:
            if self.broker_type == BrokerType.DHAN:
                return self.broker.get_ltp(
                    security_id=self.symbol, 
                    exchange_segment=self.exchange
                )
            elif self.broker_type == BrokerType.ICICI:
                quotes = self.broker.get_quotes(
                    exchange_code=self.exchange,
                    stock_code=self.symbol,
                    product_type="Futures" if self.is_futures else None,
                    expiry_date=self.expiry_date if self.is_futures else None
                )
                
                if 'Success' in quotes and quotes['Success']:
                    return float(quotes['Success'][0]['ltp'])
                return 0.0
        except Exception as e:
            logger.error(f"Failed to get current price: {e}")
            return 0.0
    
    def check_for_signals(self, data: Optional[pd.DataFrame] = None) -> SignalType:
        """Check for trading signals.
        
        Parameters
        ----------
        data : pd.DataFrame, optional
            Historical data to use for signal generation.
            If None, will fetch latest data.
            
        Returns
        -------
        SignalType
            Detected signal type.
        """
        if data is None:
            data = self.fetch_historical_data(days=3)
        
        if data.empty:
            logger.warning("No data available for signal generation")
            return SignalType.NEUTRAL
        
        signals_df = self.generate_signals(data)
        
        if signals_df.empty:
            logger.warning("Failed to generate signals")
            return SignalType.NEUTRAL
        
        # Get the latest signals
        latest = signals_df.iloc[-1]
        
        if latest['long_signal'] == 1:
            return SignalType.LONG
        elif latest['short_signal'] == -1:
            return SignalType.SHORT
        else:
            return SignalType.NEUTRAL
    
    def run_once(self) -> Dict:
        """Run one iteration of the trading algorithm.
        
        Returns
        -------
        Dict
            Result of the iteration.
        """
        try:
            # Fetch latest data
            data = self.fetch_historical_data(days=3)
            
            if data.empty:
                return {"status": "error", "message": "No data available"}
            
            # Check for signals
            signal = self.check_for_signals(data)
            
            # Place trade if there's a signal
            if signal != SignalType.NEUTRAL:
                return self.place_trade(signal)
            else:
                return {"status": "no_action", "message": "No trading signal detected"}
        
        except Exception as e:
            logger.error(f"Error in run_once: {e}")
            return {"status": "error", "message": str(e)}
    
    def run_continuous(
        self, 
        interval_seconds: int = 60,
        start_time: str = "09:15",
        end_time: str = "15:30",
        max_trades_per_day: int = 5
    ) -> None:
        """Run the trading algorithm continuously.
        
        Parameters
        ----------
        interval_seconds : int, optional
            Interval between iterations in seconds.
        start_time : str, optional
            Trading start time (HH:MM).
        end_time : str, optional
            Trading end time (HH:MM).
        max_trades_per_day : int, optional
            Maximum number of trades per day.
            
        Returns
        -------
        None
        """
        self.is_running = True
        trades_today = 0
        last_trade_date = datetime.now().date()
        
        def is_trading_time():
            now = datetime.now()
            current_time = now.time()
            
            # Parse start and end times
            start = datetime.strptime(start_time, "%H:%M").time()
            end = datetime.strptime(end_time, "%H:%M").time()
            
            # Check if current time is within trading hours
            is_time_ok = start <= current_time <= end
            
            # Check if today is a weekday (0=Monday, 6=Sunday)
            is_weekday = now.weekday() < 5
            
            return is_time_ok and is_weekday
        
        try:
            logger.info(f"Starting continuous trading for {self.symbol}")
            
            while self.is_running:
                current_date = datetime.now().date()
                
                # Reset trade counter at the start of a new day
                if current_date != last_trade_date:
                    trades_today = 0
                    last_trade_date = current_date
                
                # Only trade during trading hours and if max trades not reached
                if is_trading_time() and trades_today < max_trades_per_day:
                    result = self.run_once()
                    
                    if result.get("status") not in ["error", "no_action"]:
                        trades_today += 1
                        logger.info(f"Trade executed: {result}. Trades today: {trades_today}/{max_trades_per_day}")
                
                # Sleep for the specified interval
                time.sleep(interval_seconds)
        
        except KeyboardInterrupt:
            logger.info("Trading stopped by user")
        except Exception as e:
            logger.error(f"Error in continuous trading: {e}")
        finally:
            self.is_running = False
    
    def stop(self) -> None:
        """Stop continuous trading."""
        self.is_running = False
        logger.info("Trading stopped")
    
    def get_trade_history(self) -> pd.DataFrame:
        """Get the trade history.
        
        Returns
        -------
        pd.DataFrame
            Trade history.
        """
        return pd.DataFrame(self.trade_history)
    
    def get_positions(self) -> Dict:
        """Get current positions.
        
        Returns
        -------
        Dict
            Current positions.
        """
        if self.backtest_mode:
            return {"position": self.position}
        
        try:
            if self.broker_type == BrokerType.DHAN:
                return self.broker.get_positions()
            elif self.broker_type == BrokerType.ICICI:
                # ICICI doesn't have a direct positions API, so we'll use order list
                return self.broker.get_order_list()
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return {}
    
    def create_ema_crossover_scanner(
        self, 
        symbols: List[str], 
        exchange: str, 
        fast_period: int = 8, 
        slow_period: int = 21
    ) -> Dict[str, SignalType]:
        """Create a scanner that checks for EMA crossovers across multiple symbols.
        
        Parameters
        ----------
        symbols : List[str]
            List of symbols to scan.
        exchange : str
            Exchange to use for all symbols.
        fast_period : int, optional
            Fast EMA period.
        slow_period : int, optional
            Slow EMA period.
            
        Returns
        -------
        Dict[str, SignalType]
            Dictionary of symbols and their signal types.
        """
        results = {}
        
        for symbol in symbols:
            try:
                # Fetch data for the symbol
                data = self.fetch_historical_data(
                    days=30, 
                    interval="1minute"
                )
                
                if data.empty:
                    results[symbol] = SignalType.NEUTRAL
                    continue
                
                # Calculate EMAs
                data['ema_fast'] = data['close'].ewm(span=fast_period, adjust=False).mean()
                data['ema_slow'] = data['close'].ewm(span=slow_period, adjust=False).mean()
                
                # Look for crossovers
                data['ema_cross'] = np.where(
                    data['ema_fast'] > data['ema_slow'], 
                    1, 
                    np.where(data['ema_fast'] < data['ema_slow'], -1, 0)
                )
                
                # Check for recent crossover (last 3 bars)
                recent_data = data.iloc[-3:]
                
                if 1 in recent_data['ema_cross'].values and -1 in recent_data['ema_cross'].values:
                    if recent_data['ema_cross'].iloc[-1] == 1:
                        results[symbol] = SignalType.LONG
                    else:
                        results[symbol] = SignalType.SHORT
                else:
                    results[symbol] = SignalType.NEUTRAL
            
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                results[symbol] = SignalType.NEUTRAL
        
        return results
    
    def create_failed_breakdown_scanner(
        self, 
        symbols: List[str], 
        exchange: str
    ) -> Dict[str, SignalType]:
        """Create a scanner that checks for failed breakdowns across multiple symbols.
        
        Parameters
        ----------
        symbols : List[str]
            List of symbols to scan.
        exchange : str
            Exchange to use for all symbols.
            
        Returns
        -------
        Dict[str, SignalType]
            Dictionary of symbols and their signal types.
        """
        results = {}
        
        for symbol in symbols:
            try:
                # Temporarily change the symbol and exchange
                original_symbol = self.symbol
                original_exchange = self.exchange
                
                self.symbol = symbol
                self.exchange = exchange
                
                # Check for signals
                signal = self.check_for_signals()
                results[symbol] = signal
                
                # Restore original symbol and exchange
                self.symbol = original_symbol
                self.exchange = original_exchange
            
            except Exception as e:
                logger.error(f"Error scanning {symbol} for failed breakdowns: {e}")
                results[symbol] = SignalType.NEUTRAL
        
        return results