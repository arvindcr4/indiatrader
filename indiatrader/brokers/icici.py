"""ICICI Breeze API client for IndiaTrader.

This module implements a client for the ICICI Breeze trading API,
providing functionality for authentication, order placement,
historical data retrieval, and real-time data streaming.
"""

from typing import Dict, List, Optional, Union, Any, Tuple
import os
import logging
import pandas as pd
import numpy as np
import urllib.parse
from datetime import datetime, timedelta
import time
import threading

# Import Breeze API client (lazy import to avoid network issues on module import)
BreezeConnect = None

def _get_breeze_connect():
    """Lazy import of BreezeConnect to avoid network calls on module import."""
    global BreezeConnect
    if BreezeConnect is None:
        try:
            from breeze_connect import BreezeConnect as _BreezeConnect
            BreezeConnect = _BreezeConnect
        except ImportError:
            raise ImportError(
                "breeze-connect package is required. Install it using 'pip install breeze-connect'"
            )
    return BreezeConnect

logger = logging.getLogger(__name__)


class ICICIBreezeClient:
    """Client for interacting with the ICICI Breeze trading API."""

    # Exchange Constants
    NSE = "NSE"
    BSE = "BSE"
    NFO = "NFO"
    MCX = "MCX"
    
    # Product Type Constants
    INTRADAY = "I"
    DELIVERY = "D"
    MARGIN = "M"
    CO = "CO"
    BO = "BO"
    
    # Order Type Constants
    MARKET = "M"
    LIMIT = "L"
    STOPMARKET = "SL-M"
    STOPLIMIT = "SL"
    
    # Action Type Constants
    BUY = "B"
    SELL = "S"
    
    # Order validity Constants
    DAY = "DAY"
    IMMEDIATE = "IOC"
    
    def __init__(
        self, 
        api_key: str, 
        api_secret: Optional[str] = None, 
        session_token: Optional[str] = None,
        paper_trade_mode: bool = False
    ):
        """Initialize the ICICI Breeze API client.
        
        Parameters
        ----------
        api_key : str
            ICICI Breeze API key.
        api_secret : str, optional
            ICICI Breeze API secret. If not provided, will attempt to 
            load from environment variable ICICI_API_SECRET.
        session_token : str, optional
            Session token for authentication. If not provided, will attempt to 
            load from environment variable ICICI_SESSION_TOKEN.
        paper_trade_mode : bool, optional
            Whether to run in paper trading mode (simulated API).
        """
        self.api_key = api_key
        self.api_secret = api_secret or os.environ.get("ICICI_API_SECRET", "demo_api_secret")
        self.session_token = session_token or os.environ.get("ICICI_SESSION_TOKEN", "demo_session_token")
        self.paper_trade_mode = paper_trade_mode
        self._ws_connected = False
        
        # In paper trade mode, we don't need actual credentials
        if not paper_trade_mode:
            if not self.api_secret:
                raise ValueError(
                    "API secret must be provided or set in ICICI_API_SECRET environment variable"
                )
            
            if not self.session_token:
                raise ValueError(
                    "Session token must be provided or set in ICICI_SESSION_TOKEN environment variable"
                )
        
        # Initialize client with lazy import
        breeze_connect_class = _get_breeze_connect()
        self.client = breeze_connect_class(api_key=self.api_key)
        
        # Only perform actual authentication in live mode
        if not paper_trade_mode:
            self.authenticate()
        else:
            logger.info("Running in paper trade mode (no actual authentication)")
            
        logger.info("Initialized ICICI Breeze client")
    
    def generate_login_url(self) -> str:
        """Generate login URL for obtaining session token.
        
        Returns
        -------
        str
            Login URL.
        """
        return f"https://api.icicidirect.com/apiuser/login?api_key={urllib.parse.quote_plus(self.api_key)}"
    
    def authenticate(self) -> None:
        """Authenticate with ICICI Breeze API."""
        try:
            # Skip authentication in paper trade mode
            if self.paper_trade_mode:
                logger.info("Running in paper trade mode - authentication skipped")
                return
                
            self.client.generate_session(
                api_secret=self.api_secret,
                session_token=self.session_token
            )
            logger.info("ICICI Breeze session generated successfully")
        except Exception as e:
            logger.error(f"Failed to generate ICICI Breeze session: {e}")
            raise
            
    # For backward compatibility
    def generate_session(self) -> None:
        """Generate session using API secret and session token (alias for authenticate)."""
        return self.authenticate()
    
    def connect_websocket(self, callbacks: Dict[str, callable] = None) -> None:
        """Connect to WebSocket for real-time data.
        
        Parameters
        ----------
        callbacks : Dict[str, callable], optional
            Dictionary of callback functions for WebSocket events.
            Supported keys: 'on_ticks', 'on_connect', 'on_close', 'on_error'.
        """
        try:
            # Skip actual WebSocket connection in paper trade mode
            if self.paper_trade_mode:
                logger.info("Running in paper trade mode - using simulated WebSocket")
                self._ws_connected = True
                
                # Simulate tick data for paper trading
                def paper_trade_simulation():
                    """Simulate tick data for paper trading."""
                    # Start with current price around NIFTY level
                    price = 24500.0
                    
                    while self._ws_connected:
                        # Simulate price movement
                        price_change = np.random.normal(0, 5)  # Random change with mean 0, std 5
                        price += price_change
                        
                        # Create simulated tick
                        current_time = datetime.now()
                        tick = {
                            'NIFTY': {
                                'last_price': price,
                                'ltp': price,
                                'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S')
                            }
                        }
                        
                        # Call on_ticks callback if available
                        if hasattr(self.client, 'on_ticks') and callable(self.client.on_ticks):
                            self.client.on_ticks(tick)
                            
                        time.sleep(1)  # Simulate 1-second updates
                
                # Start simulation thread
                simulation_thread = threading.Thread(target=paper_trade_simulation, daemon=True)
                simulation_thread.start()
                
                return
            
            # Set callbacks if provided
            if callbacks:
                if 'on_ticks' in callbacks and callable(callbacks['on_ticks']):
                    self.client.on_ticks = callbacks['on_ticks']
                if 'on_connect' in callbacks and callable(callbacks['on_connect']):
                    self.client.on_connect = callbacks['on_connect']
                if 'on_close' in callbacks and callable(callbacks['on_close']):
                    self.client.on_close = callbacks['on_close']
                if 'on_error' in callbacks and callable(callbacks['on_error']):
                    self.client.on_error = callbacks['on_error']
                    
            # Connect to WebSocket
            self.client.ws_connect()
            self._ws_connected = True
            
            # Keep connection alive
            def keep_alive():
                while self._ws_connected:
                    time.sleep(60)  # Sleep for 60 seconds
            
            # Start thread to keep connection alive
            thread = threading.Thread(target=keep_alive, daemon=True)
            thread.start()
            
            logger.info("Connected to ICICI Breeze WebSocket")
        except Exception as e:
            logger.error(f"Failed to connect to ICICI Breeze WebSocket: {e}")
            raise
    
    def disconnect_websocket(self) -> None:
        """Disconnect from WebSocket."""
        try:
            self._ws_connected = False
            self.client.ws_disconnect()
            logger.info("Disconnected from ICICI Breeze WebSocket")
        except Exception as e:
            logger.error(f"Failed to disconnect from ICICI Breeze WebSocket: {e}")
    
    def subscribe_feeds(
        self, 
        exchange_code: str, 
        stock_code: str, 
        product_type: str = None,
        expiry_date: str = None,
        strike_price: str = None,
        right: str = None,
        get_exchange_quotes: bool = True,
        get_market_depth: bool = False
    ) -> Dict:
        """Subscribe to real-time data feeds.
        
        Parameters
        ----------
        exchange_code : str
            Exchange code (NSE, BSE, NFO, etc).
        stock_code : str
            Stock/scrip code.
        product_type : str, optional
            Product type (Futures, Options, etc).
        expiry_date : str, optional
            Expiry date in DD-MMM-YYYY format.
        strike_price : str, optional
            Strike price.
        right : str, optional
            Right (Call or Put).
        get_exchange_quotes : bool, optional
            Whether to get exchange quotes.
        get_market_depth : bool, optional
            Whether to get market depth.
            
        Returns
        -------
        Dict
            Subscription response.
        """
        try:
            return self.client.subscribe_feeds(
                exchange_code=exchange_code,
                stock_code=stock_code,
                product_type=product_type,
                expiry_date=expiry_date,
                strike_price=strike_price,
                right=right,
                get_exchange_quotes=get_exchange_quotes,
                get_market_depth=get_market_depth
            )
        except Exception as e:
            logger.error(f"Failed to subscribe to feeds: {e}")
            raise
    
    def unsubscribe_feeds(
        self, 
        exchange_code: str, 
        stock_code: str, 
        product_type: str = None,
        expiry_date: str = None,
        strike_price: str = None,
        right: str = None,
        get_exchange_quotes: bool = True,
        get_market_depth: bool = False
    ) -> Dict:
        """Unsubscribe from real-time data feeds.
        
        Parameters
        ----------
        exchange_code : str
            Exchange code (NSE, BSE, NFO, etc).
        stock_code : str
            Stock/scrip code.
        product_type : str, optional
            Product type (Futures, Options, etc).
        expiry_date : str, optional
            Expiry date in DD-MMM-YYYY format.
        strike_price : str, optional
            Strike price.
        right : str, optional
            Right (Call or Put).
        get_exchange_quotes : bool, optional
            Whether to get exchange quotes.
        get_market_depth : bool, optional
            Whether to get market depth.
            
        Returns
        -------
        Dict
            Unsubscription response.
        """
        try:
            return self.client.unsubscribe_feeds(
                exchange_code=exchange_code,
                stock_code=stock_code,
                product_type=product_type,
                expiry_date=expiry_date,
                strike_price=strike_price,
                right=right,
                get_exchange_quotes=get_exchange_quotes,
                get_market_depth=get_market_depth
            )
        except Exception as e:
            logger.error(f"Failed to unsubscribe from feeds: {e}")
            raise
    
    def get_customer_details(self) -> Dict:
        """Get customer details.
        
        Returns
        -------
        Dict
            Customer details.
        """
        try:
            return self.client.get_customer_details()
        except Exception as e:
            logger.error(f"Failed to get customer details: {e}")
            raise
    
    def get_demat_holdings(self) -> Dict:
        """Get demat holdings.
        
        Returns
        -------
        Dict
            Demat holdings.
        """
        try:
            return self.client.get_demat_holdings()
        except Exception as e:
            logger.error(f"Failed to get demat holdings: {e}")
            raise
    
    def get_portfolio_holdings(self) -> Dict:
        """Get portfolio holdings.
        
        Returns
        -------
        Dict
            Portfolio holdings.
        """
        try:
            return self.client.get_portfolio_holdings()
        except Exception as e:
            logger.error(f"Failed to get portfolio holdings: {e}")
            raise
    
    def get_historical_data(
        self,
        exchange_code: str,
        stock_code: str,
        interval: str,
        from_date: str,
        to_date: str,
        product_type: str = None,
        expiry_date: str = None,
        strike_price: str = None,
        right: str = None,
    ) -> pd.DataFrame:
        """Get historical data.
        
        Parameters
        ----------
        exchange_code : str
            Exchange code (NSE, BSE, NFO, etc).
        stock_code : str
            Stock/scrip code.
        interval : str
            Candle interval (1minute, 5minute, 30minute, etc).
        from_date : str
            From date in DD-MM-YYYY format.
        to_date : str
            To date in DD-MM-YYYY format.
        product_type : str, optional
            Product type (Futures, Options, etc).
        expiry_date : str, optional
            Expiry date in DD-MMM-YYYY format.
        strike_price : str, optional
            Strike price.
        right : str, optional
            Right (Call or Put).
            
        Returns
        -------
        pd.DataFrame
            Historical data.
        """
        try:
            response = self.client.get_historical_data(
                exchange_code=exchange_code,
                stock_code=stock_code,
                interval=interval,
                from_date=from_date,
                to_date=to_date,
                product_type=product_type,
                expiry_date=expiry_date,
                strike_price=strike_price,
                right=right
            )
            
            if 'Success' in response and response['Success']:
                data = response['Success']
                df = pd.DataFrame(data)
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
                # Convert numerical columns to appropriate types
                numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col])
                return df
            else:
                logger.error(f"Failed to get historical data: {response}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            return pd.DataFrame()
    
    def get_quotes(
        self,
        exchange_code: str,
        stock_code: str,
        product_type: str = None,
        expiry_date: str = None,
        strike_price: str = None,
        right: str = None,
    ) -> Dict:
        """Get quotes.
        
        Parameters
        ----------
        exchange_code : str
            Exchange code (NSE, BSE, NFO, etc).
        stock_code : str
            Stock/scrip code.
        product_type : str, optional
            Product type (Futures, Options, etc).
        expiry_date : str, optional
            Expiry date in DD-MMM-YYYY format.
        strike_price : str, optional
            Strike price.
        right : str, optional
            Right (Call or Put).
            
        Returns
        -------
        Dict
            Quotes data.
        """
        try:
            return self.client.get_quotes(
                exchange_code=exchange_code,
                stock_code=stock_code,
                product_type=product_type,
                expiry_date=expiry_date,
                strike_price=strike_price,
                right=right
            )
        except Exception as e:
            logger.error(f"Failed to get quotes: {e}")
            raise
    
    def get_limits(self) -> Dict:
        """Get limits.
        
        Returns
        -------
        Dict
            Limits data.
        """
        try:
            return self.client.get_limits()
        except Exception as e:
            logger.error(f"Failed to get limits: {e}")
            raise
    
    def place_order(
        self,
        exchange_code: str,
        stock_code: str,
        action: str,
        quantity: int,
        price_type: str,
        product_type: str,
        price: float = 0,
        validity: str = "DAY",
        stop_loss_value: float = 0,
        target_value: float = 0,
        disclose_quantity: int = 0,
        expiry_date: str = None,
        right: str = None,
        strike_price: str = None,
    ) -> Dict:
        """Place an order.
        
        Parameters
        ----------
        exchange_code : str
            Exchange code (NSE, BSE, NFO, etc).
        stock_code : str
            Stock/scrip code.
        action : str
            Action (B for Buy, S for Sell).
        quantity : int
            Quantity.
        price_type : str
            Price type (L for Limit, M for Market, etc).
        product_type : str
            Product type (I for Intraday, D for Delivery, etc).
        price : float, optional
            Price.
        validity : str, optional
            Validity (DAY, IOC).
        stop_loss_value : float, optional
            Stop loss value.
        target_value : float, optional
            Target value.
        disclose_quantity : int, optional
            Disclosed quantity.
        expiry_date : str, optional
            Expiry date in DD-MMM-YYYY format.
        right : str, optional
            Right (Call or Put).
        strike_price : str, optional
            Strike price.
            
        Returns
        -------
        Dict
            Order response.
        """
        try:
            return self.client.place_order(
                exchange_code=exchange_code,
                stock_code=stock_code,
                action=action,
                quantity=quantity,
                price_type=price_type,
                product_type=product_type,
                price=price,
                validity=validity,
                stoploss=stop_loss_value,
                target=target_value,
                disclosed_quantity=disclose_quantity,
                expiry_date=expiry_date,
                right=right,
                strike_price=strike_price
            )
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            raise
    
    def modify_order(
        self,
        exchange_code: str,
        order_id: str,
        price_type: str,
        quantity: int = None,
        price: float = None,
        validity: str = None,
        stop_loss_value: float = None,
        target_value: float = None,
        disclosed_quantity: int = None,
    ) -> Dict:
        """Modify an order.
        
        Parameters
        ----------
        exchange_code : str
            Exchange code (NSE, BSE, NFO, etc).
        order_id : str
            Order ID.
        price_type : str
            Price type (L for Limit, M for Market, etc).
        quantity : int, optional
            Quantity.
        price : float, optional
            Price.
        validity : str, optional
            Validity (DAY, IOC).
        stop_loss_value : float, optional
            Stop loss value.
        target_value : float, optional
            Target value.
        disclosed_quantity : int, optional
            Disclosed quantity.
            
        Returns
        -------
        Dict
            Order modification response.
        """
        try:
            return self.client.modify_order(
                exchange_code=exchange_code,
                order_id=order_id,
                price_type=price_type,
                quantity=quantity,
                price=price,
                validity=validity,
                stoploss=stop_loss_value,
                target=target_value,
                disclosed_quantity=disclosed_quantity
            )
        except Exception as e:
            logger.error(f"Failed to modify order: {e}")
            raise
    
    def cancel_order(
        self,
        exchange_code: str,
        order_id: str
    ) -> Dict:
        """Cancel an order.
        
        Parameters
        ----------
        exchange_code : str
            Exchange code (NSE, BSE, NFO, etc).
        order_id : str
            Order ID.
            
        Returns
        -------
        Dict
            Order cancellation response.
        """
        try:
            return self.client.cancel_order(
                exchange_code=exchange_code,
                order_id=order_id
            )
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            raise
    
    def get_order_list(self) -> Dict:
        """Get order list.
        
        Returns
        -------
        Dict
            Order list.
        """
        try:
            return self.client.get_order_list()
        except Exception as e:
            logger.error(f"Failed to get order list: {e}")
            raise
    
    def get_trade_list(self) -> Dict:
        """Get trade list.
        
        Returns
        -------
        Dict
            Trade list.
        """
        try:
            return self.client.get_trade_list()
        except Exception as e:
            logger.error(f"Failed to get trade list: {e}")
            raise
    
    def place_mancini_trades(
        self,
        symbol: str,
        exchange: str,
        product_type: str = "I",
        quantity: int = 1,
        opening_range_minutes: int = 6,
        is_paper_trade: bool = True,
        expiry_date: str = None,
        is_futures: bool = False
    ) -> None:
        """Run Adam Mancini strategy in real-time and place trades.
        
        Parameters
        ----------
        symbol : str
            Trading symbol (e.g., NIFTY, BANKNIFTY).
        exchange : str
            Exchange code (NSE, BSE, NFO, etc).
        product_type : str, optional
            Product type (I for Intraday, D for Delivery, etc).
        quantity : int, optional
            Quantity of shares/contracts.
        opening_range_minutes : int, optional
            Number of minutes for the opening range period.
        is_paper_trade : bool, optional
            Whether to run in paper trading mode (no real orders).
        expiry_date : str, optional
            Expiry date in DD-MMM-YYYY format.
        is_futures : bool, optional
            Whether the instrument is a futures contract.
        """
        from indiatrader.strategies.mancini_trader import AdamManciniNiftyStrategy
        import numpy as np
        
        logger.info(f"Starting Adam Mancini strategy for {symbol} on {exchange}")
        logger.info(f"Opening range minutes: {opening_range_minutes}, Paper trade: {is_paper_trade}")
        
        # Initialize strategy
        strategy = AdamManciniNiftyStrategy(open_range_minutes=opening_range_minutes)
        
        # Current state
        position = 0  # 0: no position, 1: long, -1: short
        entry_price = 0
        daily_high_low = {}
        opening_range = {}
        
        # Convert exchange names if needed
        exchange_code = exchange
        if exchange == "NSE":
            exchange_code = self.NSE
        elif exchange == "BSE":
            exchange_code = self.BSE
        elif exchange == "NFO":
            exchange_code = self.NFO
            
        # Get stock code
        stock_code = symbol
            
        # Subscribe to real-time data
        self.subscribe_feeds(
            exchange_code=exchange_code,
            stock_code=stock_code,
            product_type="Futures" if is_futures else None,
            expiry_date=expiry_date if is_futures else None
        )
        
        # Real-time data processing
        def on_ticks(ticks):
            nonlocal position, entry_price, daily_high_low, opening_range
            
            if not ticks or symbol not in ticks:
                return
                
            tick = ticks[symbol]
            current_time = datetime.now()
            market_open_time = current_time.replace(hour=9, minute=15, second=0, microsecond=0)
            market_close_time = current_time.replace(hour=15, minute=30, second=0, microsecond=0)
            
            # Only trade during market hours
            if current_time < market_open_time or current_time > market_close_time:
                return
                
            # Get current date
            current_date = current_time.strftime('%Y-%m-%d')
            
            # Initialize daily high/low if needed
            if current_date not in daily_high_low:
                daily_high_low[current_date] = {
                    'high': float('-inf'),
                    'low': float('inf'),
                    'close': None
                }
                
            # Update daily high/low
            price = tick.get('last_price', tick.get('ltp', 0))
            daily_high_low[current_date]['high'] = max(daily_high_low[current_date]['high'], price)
            daily_high_low[current_date]['low'] = min(daily_high_low[current_date]['low'], price)
            daily_high_low[current_date]['close'] = price
            
            # Calculate opening range (first N minutes of trading)
            opening_range_end_time = market_open_time + timedelta(minutes=opening_range_minutes)
            
            if current_date not in opening_range and current_time >= market_open_time:
                opening_range[current_date] = {
                    'high': float('-inf'),
                    'low': float('inf')
                }
                
            if current_date in opening_range and current_time <= opening_range_end_time:
                opening_range[current_date]['high'] = max(opening_range[current_date]['high'], price)
                opening_range[current_date]['low'] = min(opening_range[current_date]['low'], price)
                
            # Trading logic - only after opening range is established
            if (current_date in opening_range and 
                current_time > opening_range_end_time and
                current_date in daily_high_low):
                
                # Calculate pivot level using previous day data
                prev_date = (datetime.strptime(current_date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
                
                if prev_date in daily_high_low:
                    prev_high = daily_high_low[prev_date]['high']
                    prev_low = daily_high_low[prev_date]['low']
                    prev_close = daily_high_low[prev_date]['close']
                    
                    pivot = (prev_high + prev_low + prev_close) / 3
                    
                    # Generate signals
                    long_signal = (price > opening_range[current_date]['high']) and (price > pivot)
                    short_signal = (price < opening_range[current_date]['low']) and (price < pivot)
                    
                    # Trading logic
                    if long_signal and position <= 0:
                        # Long signal - buy
                        if is_paper_trade:
                            logger.info(f"PAPER TRADE: BUY {symbol} at {price}")
                        else:
                            kwargs = {
                                'exchange_code': exchange_code,
                                'stock_code': stock_code,
                                'action': self.BUY,
                                'quantity': quantity,
                                'price_type': self.MARKET,
                                'product_type': product_type,
                                'price': 0,
                            }
                            
                            if is_futures and expiry_date:
                                kwargs['product_type'] = 'Futures'
                                kwargs['expiry_date'] = expiry_date
                            
                            try:
                                response = self.place_order(**kwargs)
                                logger.info(f"Placed LONG order for {symbol} at {price}: {response}")
                            except Exception as e:
                                logger.error(f"Error placing LONG order for {symbol}: {e}")
                                
                        position = 1
                        entry_price = price
                        
                    elif short_signal and position >= 0:
                        # Short signal - sell
                        if is_paper_trade:
                            logger.info(f"PAPER TRADE: SELL {symbol} at {price}")
                        else:
                            kwargs = {
                                'exchange_code': exchange_code,
                                'stock_code': stock_code,
                                'action': self.SELL,
                                'quantity': quantity,
                                'price_type': self.MARKET,
                                'product_type': product_type,
                                'price': 0,
                            }
                            
                            if is_futures and expiry_date:
                                kwargs['product_type'] = 'Futures'
                                kwargs['expiry_date'] = expiry_date
                            
                            try:
                                response = self.place_order(**kwargs)
                                logger.info(f"Placed SHORT order for {symbol} at {price}: {response}")
                            except Exception as e:
                                logger.error(f"Error placing SHORT order for {symbol}: {e}")
                                
                        position = -1
                        entry_price = price
                        
        # Register callback for real-time data
        self.client.on_ticks = on_ticks
        
        # Connect to WebSocket
        self.connect_websocket()
        
        logger.info("Connected to real-time data feed. Waiting for signals...")
        logger.info("Press Ctrl+C to stop the strategy.")
        
        try:
            # Keep the main thread running
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Strategy stopped by user.")
        finally:
            self.disconnect_websocket()
            
    def place_mancini_trades_backtest(
        self, 
        signals: pd.DataFrame, 
        exchange_code: str,
        stock_code: str, 
        quantity: int,
        product_type: str = "I",
        is_futures: bool = False,
        expiry_date: str = None
    ) -> List[Dict]:
        """Place trades based on Adam Mancini strategy signals for backtesting.
        
        Parameters
        ----------
        signals : pd.DataFrame
            DataFrame with signals from AdamManciniNiftyStrategy.
        exchange_code : str
            Exchange code (NSE, BSE, NFO, etc).
        stock_code : str
            Stock/scrip code.
        quantity : int
            Quantity of shares/contracts.
        product_type : str, optional
            Product type (I for Intraday, D for Delivery, etc).
        is_futures : bool, optional
            Whether the instrument is a futures contract.
        expiry_date : str, optional
            Expiry date in DD-MMM-YYYY format.
            
        Returns
        -------
        List[Dict]
            List of order responses.
        """
        orders = []
        
        # Filter for only rows with active signals
        long_signals = signals[signals['long_signal'] == 1]
        short_signals = signals[signals['short_signal'] == -1]
        
        # Place long orders
        for idx, row in long_signals.iterrows():
            try:
                kwargs = {
                    'exchange_code': exchange_code,
                    'stock_code': stock_code,
                    'action': self.BUY,
                    'quantity': quantity,
                    'price_type': self.MARKET,
                    'product_type': product_type,
                    'price': 0,
                }
                
                if is_futures and expiry_date:
                    kwargs['product_type'] = 'Futures'
                    kwargs['expiry_date'] = expiry_date
                
                response = self.place_order(**kwargs)
                orders.append({
                    'timestamp': idx,
                    'type': 'LONG',
                    'response': response
                })
                logger.info(f"Placed LONG order for {stock_code} at {idx}: {response}")
            except Exception as e:
                logger.error(f"Error placing LONG order for {stock_code} at {idx}: {e}")
        
        # Place short orders
        for idx, row in short_signals.iterrows():
            try:
                kwargs = {
                    'exchange_code': exchange_code,
                    'stock_code': stock_code,
                    'action': self.SELL,
                    'quantity': quantity,
                    'price_type': self.MARKET,
                    'product_type': product_type,
                    'price': 0,
                }
                
                if is_futures and expiry_date:
                    kwargs['product_type'] = 'Futures'
                    kwargs['expiry_date'] = expiry_date
                
                response = self.place_order(**kwargs)
                orders.append({
                    'timestamp': idx,
                    'type': 'SHORT',
                    'response': response
                })
                logger.info(f"Placed SHORT order for {stock_code} at {idx}: {response}")
            except Exception as e:
                logger.error(f"Error placing SHORT order for {stock_code} at {idx}: {e}")
                
        return orders