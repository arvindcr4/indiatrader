"""Dhan API client for IndiaTrader.

This module implements a client for the Dhan trading API,
providing functionality for authentication, order placement,
historical data retrieval, and real-time data streaming.
"""

from typing import Dict, List, Optional, Union, Any
import os
import logging
import pandas as pd
from datetime import datetime, timedelta

# Import Dhan API client
try:
    from dhanhq import dhanhq
except ImportError:
    raise ImportError(
        "dhanhq package is required. Install it using 'pip install dhanhq'"
    )

logger = logging.getLogger(__name__)


class DhanClient:
    """Client for interacting with the Dhan trading API."""

    # Exchange Constants
    NSE = "NSE"
    BSE = "BSE"
    NFO = "NFO"
    MCX = "MCX"
    
    # Order Type Constants
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"
    SLM = "SLM"
    
    # Transaction Type Constants
    BUY = "BUY"
    SELL = "SELL"
    
    # Product Type Constants
    INTRA = "INTRADAY"
    DELIVERY = "DELIVERY"
    MARGIN = "MARGIN"
    BO = "BO"
    CO = "CO"

    def __init__(self, client_id: str, access_token: str = None):
        """Initialize the Dhan API client.
        
        Parameters
        ----------
        client_id : str
            Your Dhan client ID.
        access_token : str, optional
            Your Dhan access token. If not provided, will attempt to load 
            from environment variable DHAN_ACCESS_TOKEN.
        """
        self.client_id = client_id
        self.access_token = access_token or os.environ.get("DHAN_ACCESS_TOKEN")
        
        if not self.access_token:
            raise ValueError(
                "Access token must be provided or set in DHAN_ACCESS_TOKEN environment variable"
            )
            
        self.client = dhanhq(client_id, self.access_token)
        logger.info(f"Initialized Dhan client for {client_id}")
        
    def get_profile(self) -> Dict:
        """Get user profile information.
        
        Returns
        -------
        Dict
            User profile data.
        """
        return self.client.get_user_details()
    
    def get_funds(self) -> Dict:
        """Get account fund details.
        
        Returns
        -------
        Dict
            Account fund information.
        """
        return self.client.get_funds()
    
    def get_positions(self) -> Dict:
        """Get current day's positions.
        
        Returns
        -------
        Dict
            Current day's position information.
        """
        return self.client.get_positions()
    
    def get_holdings(self) -> Dict:
        """Get holdings information.
        
        Returns
        -------
        Dict
            Holdings information.
        """
        return self.client.get_holdings()

    def get_order_book(self) -> Dict:
        """Get order book for the day.
        
        Returns
        -------
        Dict
            Order book information.
        """
        return self.client.get_order_book()
    
    def get_trade_book(self) -> Dict:
        """Get trade book for the day.
        
        Returns
        -------
        Dict
            Trade book information.
        """
        return self.client.get_trade_book()

    def place_order(
        self,
        security_id: str,
        exchange_segment: str,
        transaction_type: str,
        quantity: int,
        order_type: str,
        price: float = 0,
        trigger_price: float = 0,
        stop_loss_value: float = 0,
        square_off_value: float = 0,
        trailing_stop_loss: float = 0,
        disclosed_quantity: int = 0,
        validity: str = "DAY",
        product_type: str = "INTRADAY",
        source: str = "WEB"
    ) -> Dict:
        """Place an order.
        
        Parameters
        ----------
        security_id : str
            Security ID of the scrip.
        exchange_segment : str
            Exchange segment (NSE, BSE, NFO, etc).
        transaction_type : str
            Transaction type (BUY or SELL).
        quantity : int
            Quantity of shares/contracts.
        order_type : str
            Order type (MARKET, LIMIT, SL, SLM).
        price : float, optional
            Order price, required for LIMIT orders.
        trigger_price : float, optional
            Trigger price, required for SL and SLM orders.
        stop_loss_value : float, optional
            Stop loss value for BO orders.
        square_off_value : float, optional
            Square off value for BO orders.
        trailing_stop_loss : float, optional
            Trailing stop loss for BO orders.
        disclosed_quantity : int, optional
            Disclosed quantity.
        validity : str, optional
            Order validity (DAY, IOC).
        product_type : str, optional
            Product type (INTRADAY, DELIVERY, MARGIN, BO, CO).
        source : str, optional
            Order source.
            
        Returns
        -------
        Dict
            Order response information.
        """
        return self.client.place_order(
            security_id=security_id,
            exchange_segment=exchange_segment,
            transaction_type=transaction_type,
            quantity=quantity,
            order_type=order_type,
            product_type=product_type,
            price=price,
            trigger_price=trigger_price,
            stop_loss_value=stop_loss_value,
            square_off_value=square_off_value,
            trailing_stop_loss=trailing_stop_loss,
            disclosed_quantity=disclosed_quantity,
            validity=validity,
            source=source
        )
    
    def modify_order(self, order_id: str, **kwargs) -> Dict:
        """Modify an existing order.
        
        Parameters
        ----------
        order_id : str
            Order ID to modify.
        **kwargs
            Parameters to modify.
            
        Returns
        -------
        Dict
            Modify order response.
        """
        return self.client.modify_order(order_id=order_id, **kwargs)
    
    def cancel_order(self, order_id: str) -> Dict:
        """Cancel an existing order.
        
        Parameters
        ----------
        order_id : str
            Order ID to cancel.
            
        Returns
        -------
        Dict
            Cancel order response.
        """
        return self.client.cancel_order(order_id=order_id)
    
    def get_historical_data(
        self,
        security_id: str,
        exchange_segment: str,
        instrument_type: str,
        from_date: str,
        to_date: str,
        interval: str = "1minute"
    ) -> pd.DataFrame:
        """Get historical data for a security.
        
        Parameters
        ----------
        security_id : str
            Security ID of the scrip.
        exchange_segment : str
            Exchange segment (NSE, BSE, NFO, etc).
        instrument_type : str
            Instrument type (EQ, FUT, OPT, etc).
        from_date : str
            From date in YYYY-MM-DD format.
        to_date : str
            To date in YYYY-MM-DD format.
        interval : str, optional
            Candle interval.
            
        Returns
        -------
        pd.DataFrame
            Historical data as DataFrame.
        """
        response = self.client.get_historical_data(
            security_id=security_id,
            exchange_segment=exchange_segment,
            instrument_type=instrument_type,
            from_date=from_date,
            to_date=to_date,
            interval=interval
        )
        
        if 'data' in response and 'candles' in response['data']:
            data = response['data']['candles']
            df = pd.DataFrame(data, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            return df
        else:
            logger.error(f"Failed to get historical data: {response}")
            return pd.DataFrame()
    
    def get_ltp(self, security_id: str, exchange_segment: str) -> float:
        """Get last traded price for a security.
        
        Parameters
        ----------
        security_id : str
            Security ID of the scrip.
        exchange_segment : str
            Exchange segment (NSE, BSE, NFO, etc).
            
        Returns
        -------
        float
            Last traded price.
        """
        response = self.client.get_quotes(
            security_id=security_id, 
            exchange_segment=exchange_segment,
            quote_type="LTP"
        )
        
        if 'data' in response and 'ltp' in response['data']:
            return float(response['data']['ltp'])
        else:
            logger.error(f"Failed to get LTP: {response}")
            return 0.0
    
    def search_scrip(self, search_string: str) -> List[Dict]:
        """Search for a scrip.
        
        Parameters
        ----------
        search_string : str
            String to search for.
            
        Returns
        -------
        List[Dict]
            List of matching scrips.
        """
        response = self.client.search_scrip(search_string=search_string)
        
        if 'data' in response:
            return response['data']
        else:
            logger.error(f"Failed to search scrip: {response}")
            return []
            
    def place_mancini_trades(
        self, 
        signals: pd.DataFrame, 
        symbol: str, 
        exchange: str, 
        quantity: int,
        product_type: str = "INTRADAY"
    ) -> List[Dict]:
        """Place trades based on Adam Mancini strategy signals.
        
        Parameters
        ----------
        signals : pd.DataFrame
            DataFrame with signals from AdamManciniNiftyStrategy.
        symbol : str
            Security ID of the scrip.
        exchange : str
            Exchange segment (NSE, BSE, NFO, etc).
        quantity : int
            Quantity of shares/contracts.
        product_type : str, optional
            Product type (INTRADAY, DELIVERY, MARGIN, BO, CO).
            
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
                response = self.place_order(
                    security_id=symbol,
                    exchange_segment=exchange,
                    transaction_type=self.BUY,
                    quantity=quantity,
                    order_type=self.MARKET,
                    product_type=product_type,
                    price=0
                )
                orders.append({
                    'timestamp': idx,
                    'type': 'LONG',
                    'response': response
                })
                logger.info(f"Placed LONG order for {symbol} at {idx}: {response}")
            except Exception as e:
                logger.error(f"Error placing LONG order for {symbol} at {idx}: {e}")
        
        # Place short orders
        for idx, row in short_signals.iterrows():
            try:
                response = self.place_order(
                    security_id=symbol,
                    exchange_segment=exchange,
                    transaction_type=self.SELL,
                    quantity=quantity,
                    order_type=self.MARKET,
                    product_type=product_type,
                    price=0
                )
                orders.append({
                    'timestamp': idx,
                    'type': 'SHORT',
                    'response': response
                })
                logger.info(f"Placed SHORT order for {symbol} at {idx}: {response}")
            except Exception as e:
                logger.error(f"Error placing SHORT order for {symbol} at {idx}: {e}")
                
        return orders