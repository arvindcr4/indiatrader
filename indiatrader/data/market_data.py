"""
Market data connectors for Indian exchanges using Dhan and ICICI Breeze APIs.
"""

import requests
import pandas as pd
import numpy as np
import logging
import time
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import pyarrow as pa
import pyarrow.parquet as pq
import os

from indiatrader.data.config import get_api_credentials, get_market_symbols

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class DhanConnector:
    """
    Connector for Dhan API.
    """
    
    def __init__(self, client_id: Optional[str] = None, access_token: Optional[str] = None):
        """
        Initialize Dhan connector.
        
        Args:
            client_id: Dhan client ID. If None, will be loaded from config.
            access_token: Dhan access token. If None, will be loaded from config.
        """
        credentials = get_api_credentials("market_data", "dhan")
        self.client_id = client_id or credentials.get("client_id")
        self.access_token = access_token or credentials.get("access_token") or os.environ.get("DHAN_ACCESS_TOKEN")
        
        # Try to import Dhan API client
        try:
            from dhanhq import dhanhq
            self.client = dhanhq(self.client_id, self.access_token)
            logger.info(f"Initialized Dhan API client for {self.client_id}")
        except ImportError:
            logger.warning("dhanhq package not found. Install it using 'pip install dhanhq'")
            self.client = None
        except Exception as e:
            logger.error(f"Failed to initialize Dhan API client: {str(e)}")
            self.client = None
    
    def get_market_status(self) -> Dict:
        """
        Get current market status.
        
        Returns:
            Dict containing market status information
        """
        try:
            # Dhan doesn't have a specific market status API
            # We'll check if we can get a quote as a proxy for market status
            quote = self.get_quote("NIFTY")
            if quote:
                return {"status": "open", "message": "Market is open"}
            else:
                return {"status": "closed", "message": "Market might be closed"}
        except Exception as e:
            logger.error(f"Failed to get market status: {str(e)}")
            return {"status": "unknown", "message": f"Error: {str(e)}"}
    
    def get_indices(self) -> pd.DataFrame:
        """
        Get current index values.
        
        Returns:
            DataFrame containing index information
        """
        try:
            if not self.client:
                return pd.DataFrame()
                
            # Get quotes for major indices
            indices = ["NIFTY 50", "BANKNIFTY", "NIFTY IT", "NIFTY AUTO", "NIFTY PHARMA"]
            data = []
            
            for idx in indices:
                try:
                    # Try to get index data using quote_data
                    quote_response = self.client.quote_data(
                        exchange="NSE",
                        symbol=idx
                    )
                    
                    if isinstance(quote_response, dict) and "quote" in quote_response:
                        quote_data = quote_response["quote"]
                        
                        data.append({
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'open': float(quote_data.get("open_price", 0)),
                            'high': float(quote_data.get("high_price", 0)),
                            'low': float(quote_data.get("low_price", 0)),
                            'close': float(quote_data.get("last_price", 0)),
                            'volume': int(quote_data.get("total_volume", 0)),
                            'change': float(quote_data.get("net_change", 0)),
                            'change_pct': float(quote_data.get("net_change_percentage", 0)),
                            'symbol': idx
                        })
                except Exception as e:
                    logger.warning(f"Failed to get quote data for index {idx}: {str(e)}")
                    
                    # Try OHLC data as fallback
                    try:
                        ohlc_response = self.client.ohlc_data(
                            exchange="NSE",
                            symbol=idx
                        )
                        
                        if isinstance(ohlc_response, dict) and "ohlc" in ohlc_response:
                            ohlc_data = ohlc_response["ohlc"]
                            
                            data.append({
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'open': float(ohlc_data.get("open", 0)),
                                'high': float(ohlc_data.get("high", 0)),
                                'low': float(ohlc_data.get("low", 0)),
                                'close': float(ohlc_data.get("close", 0)),
                                'volume': int(ohlc_data.get("volume", 0)),
                                'change': 0,  # May not be available
                                'change_pct': 0,  # May not be available
                                'symbol': idx
                            })
                    except Exception as ohlc_error:
                        logger.warning(f"Failed to get OHLC data for index {idx}: {str(ohlc_error)}")
                        
                        # Try historical daily data as a last resort
                        try:
                            # Get today's data
                            end_date = datetime.now().strftime("%Y-%m-%d")
                            start_date = end_date  # Same day
                            
                            historical_response = self.client.historical_daily_data(
                                exchange="NSE",
                                symbol=idx,
                                start_date=start_date,
                                end_date=end_date
                            )
                            
                            if isinstance(historical_response, dict) and "candles" in historical_response:
                                if len(historical_response["candles"]) > 0:
                                    latest_data = historical_response["candles"][-1]
                                    
                                    data.append({
                                        'timestamp': latest_data[0],
                                        'open': float(latest_data[1]),
                                        'high': float(latest_data[2]),
                                        'low': float(latest_data[3]),
                                        'close': float(latest_data[4]),
                                        'volume': int(latest_data[5]),
                                        'change': 0,  # Need previous day data to calculate
                                        'change_pct': 0,  # Need previous day data to calculate
                                        'symbol': idx
                                    })
                        except Exception as hist_error:
                            logger.warning(f"Failed to get historical data for index {idx}: {str(hist_error)}")
            
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Failed to get indices: {str(e)}")
            return pd.DataFrame()
    
    def get_quote(self, symbol: str) -> Dict:
        """
        Get quote for a specific symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'RELIANCE')
        
        Returns:
            Dict containing quote information
        """
        try:
            if not self.client:
                return {}
                
            # Determine exchange based on symbol
            exchange = "NSE"
            if symbol in ["NIFTY", "BANKNIFTY", "NIFTY 50", "NIFTY BANK"]:
                exchange = "NSE"
                
            # Try the quote_data method available in dhanhq 2.0.2
            try:
                quote_response = self.client.quote_data(
                    exchange=exchange,
                    symbol=symbol
                )
                
                if isinstance(quote_response, dict) and "quote" in quote_response:
                    quote_data = quote_response["quote"]
                    
                    # Create a standardized result
                    result = {
                        "tradingsymbol": symbol,
                        "exchange": exchange,
                        "open": float(quote_data.get("open_price", 0)),
                        "high": float(quote_data.get("high_price", 0)),
                        "low": float(quote_data.get("low_price", 0)),
                        "lastPrice": float(quote_data.get("last_price", 0)),
                        "close": float(quote_data.get("close_price", 0)),
                        "totalTradedVolume": int(quote_data.get("total_volume", 0)),
                        "change": float(quote_data.get("net_change", 0)),
                        "pChange": float(quote_data.get("net_change_percentage", 0)),
                        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    return result
            except Exception as quote_error:
                logger.warning(f"Error getting quote data: {str(quote_error)}")
                
            # Try OHLC data as a fallback
            try:
                ohlc_response = self.client.ohlc_data(
                    exchange=exchange,
                    symbol=symbol
                )
                
                if isinstance(ohlc_response, dict) and "ohlc" in ohlc_response:
                    ohlc_data = ohlc_response["ohlc"]
                    
                    # Create a standardized result
                    result = {
                        "tradingsymbol": symbol,
                        "exchange": exchange,
                        "open": float(ohlc_data.get("open", 0)),
                        "high": float(ohlc_data.get("high", 0)),
                        "low": float(ohlc_data.get("low", 0)),
                        "lastPrice": float(ohlc_data.get("close", 0)),  # Use close as lastPrice
                        "close": float(ohlc_data.get("close", 0)),
                        "totalTradedVolume": int(ohlc_data.get("volume", 0)),
                        "change": 0,  # May not be available
                        "pChange": 0,  # May not be available
                        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    return result
            except Exception as ohlc_error:
                logger.warning(f"Error getting OHLC data: {str(ohlc_error)}")
                
            # Try historical data as a last resort
            try:
                # Get today's data
                end_date = datetime.now().strftime("%Y-%m-%d")
                start_date = end_date  # Same day
                
                historical_response = self.client.historical_daily_data(
                    exchange=exchange,
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if isinstance(historical_response, dict) and "candles" in historical_response:
                    if len(historical_response["candles"]) > 0:
                        latest_data = historical_response["candles"][-1]
                        
                        # Create a standardized result from historical data
                        result = {
                            "tradingsymbol": symbol,
                            "exchange": exchange,
                            "open": float(latest_data[1]),  # Open is at index 1
                            "high": float(latest_data[2]),  # High is at index 2
                            "low": float(latest_data[3]),   # Low is at index 3
                            "lastPrice": float(latest_data[4]),  # Close is at index 4
                            "close": float(latest_data[4]),
                            "totalTradedVolume": int(latest_data[5]),  # Volume is at index 5
                            "change": 0,  # Calculate if possible
                            "pChange": 0,  # Calculate if possible
                            "timestamp": latest_data[0]  # Timestamp is at index 0
                        }
                        return result
            except Exception as hist_error:
                logger.warning(f"Error getting historical data: {str(hist_error)}")
            
            logger.error(f"All quote methods failed for {symbol}")
            return {}
        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {str(e)}")
            return {}
    
    def get_historical_data(
        self, 
        symbol: str, 
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Get historical data for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'RELIANCE')
            start_date: Start date in YYYY-MM-DD format or datetime
            end_date: End date in YYYY-MM-DD format or datetime
            interval: Data interval ('1m', '5m', '15m', '30m', '1h', '1d')
        
        Returns:
            DataFrame with historical price/volume data
        """
        try:
            if not self.client:
                return pd.DataFrame()
                
            # Convert dates to string format if datetime objects
            if isinstance(start_date, datetime):
                start_date = start_date.strftime("%Y-%m-%d")
            
            if isinstance(end_date, datetime):
                end_date = end_date.strftime("%Y-%m-%d")
                
            # Determine exchange
            exchange = "NSE"
            if any(idx in symbol for idx in ["NIFTY", "BANKNIFTY"]):
                exchange = "NSE"
                
            # Use historical_daily_data for daily data (available in dhanhq 2.0.2)
            if interval == "1d":
                try:
                    logger.info(f"Fetching daily historical data for {symbol}")
                    response = self.client.historical_daily_data(
                        exchange=exchange,
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    # Process the response
                    if isinstance(response, dict) and "candles" in response:
                        candles = response["candles"]
                        
                        # Create DataFrame from candles
                        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
                        
                        # Convert types
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                        for col in ["open", "high", "low", "close"]:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        df["volume"] = pd.to_numeric(df["volume"], errors='coerce').astype(int)
                        
                        # Calculate change and percentage
                        df["change"] = df["close"].diff()
                        df["change_pct"] = df["close"].pct_change() * 100
                        
                        return df
                except Exception as daily_error:
                    logger.warning(f"Error fetching daily historical data: {str(daily_error)}")
            
            # Use intraday_minute_data for intraday data
            if interval in ["1m", "5m", "15m", "30m", "1h"]:
                try:
                    logger.info(f"Fetching intraday data for {symbol}")
                    
                    # Map interval to minutes
                    minutes_map = {
                        "1m": 1,
                        "5m": 5,
                        "15m": 15,
                        "30m": 30,
                        "1h": 60
                    }
                    minutes = minutes_map.get(interval, 1)
                    
                    response = self.client.intraday_minute_data(
                        exchange=exchange,
                        symbol=symbol,
                        minutes=minutes
                    )
                    
                    # Process the response
                    if isinstance(response, dict) and "candles" in response:
                        candles = response["candles"]
                        
                        # Create DataFrame from candles
                        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
                        
                        # Convert types
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                        for col in ["open", "high", "low", "close"]:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        df["volume"] = pd.to_numeric(df["volume"], errors='coerce').astype(int)
                        
                        # Calculate change and percentage
                        df["change"] = df["close"].diff()
                        df["change_pct"] = df["close"].pct_change() * 100
                        
                        return df
                except Exception as intraday_error:
                    logger.warning(f"Error fetching intraday data: {str(intraday_error)}")
            
            # No data available
            logger.warning(f"No historical data available for {symbol} with interval {interval}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {str(e)}")
            return pd.DataFrame()


class ICICIBreezeConnector:
    """
    Connector for ICICI Breeze API.
    """
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, session_token: Optional[str] = None):
        """
        Initialize ICICI Breeze connector.
        
        Args:
            api_key: ICICI Breeze API key. If None, will be loaded from config.
            api_secret: ICICI Breeze API secret. If None, will be loaded from config.
            session_token: ICICI Breeze session token. If None, will be loaded from config.
        """
        credentials = get_api_credentials("market_data", "icici")
        self.api_key = api_key or credentials.get("api_key")
        self.api_secret = api_secret or credentials.get("api_secret") or os.environ.get("ICICI_API_SECRET")
        self.session_token = session_token or credentials.get("session_token") or os.environ.get("ICICI_SESSION_TOKEN")
        
        # Try to import ICICI Breeze API client
        try:
            from breeze_connect import BreezeConnect
            self.client = BreezeConnect(api_key=self.api_key)
            
            # Generate session
            self.client.generate_session(
                api_secret=self.api_secret,
                session_token=self.session_token
            )
            logger.info("ICICI Breeze session generated successfully")
        except ImportError:
            logger.warning("breeze-connect package not found. Install it using 'pip install breeze-connect'")
            self.client = None
        except Exception as e:
            logger.error(f"Failed to initialize ICICI Breeze API client: {str(e)}")
            self.client = None
    
    def get_market_status(self) -> Dict:
        """
        Get current market status.
        
        Returns:
            Dict containing market status information
        """
        try:
            # ICICI Breeze doesn't have a specific market status API
            # Try to get customer details as a proxy for connectivity
            if not self.client:
                return {"status": "closed", "message": "API client not initialized"}
                
            customer_details = self.client.get_customer_details()
            if customer_details and "Success" in customer_details:
                return {"status": "open", "message": "Market is open"}
            else:
                return {"status": "closed", "message": "Market might be closed"}
        except Exception as e:
            logger.error(f"Failed to get market status: {str(e)}")
            return {"status": "unknown", "message": f"Error: {str(e)}"}
    
    def get_indices(self) -> pd.DataFrame:
        """
        Get current index values.
        
        Returns:
            DataFrame containing index information
        """
        try:
            if not self.client:
                return pd.DataFrame()
                
            # Get quotes for major indices
            indices = [
                {"code": "NIFTY 50", "exchange": "NSE"},
                {"code": "BANKNIFTY", "exchange": "NSE"},
                {"code": "NIFTY IT", "exchange": "NSE"},
                {"code": "NIFTY AUTO", "exchange": "NSE"},
                {"code": "NIFTY PHARMA", "exchange": "NSE"}
            ]
            
            data = []
            for idx in indices:
                try:
                    quote = self.client.get_quotes(
                        exchange_code=idx["exchange"],
                        stock_code=idx["code"]
                    )
                    
                    if quote and "Success" in quote:
                        quote_data = quote["Success"]
                        data.append({
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'open': float(quote_data.get("open", 0)),
                            'high': float(quote_data.get("high", 0)),
                            'low': float(quote_data.get("low", 0)),
                            'close': float(quote_data.get("close", 0)),
                            'volume': int(quote_data.get("volume", 0)),
                            'change': float(quote_data.get("change", 0)),
                            'change_pct': float(quote_data.get("change_percentage", 0)),
                            'symbol': idx["code"]
                        })
                except Exception as e:
                    logger.warning(f"Failed to get quote for index {idx['code']}: {str(e)}")
            
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Failed to get indices: {str(e)}")
            return pd.DataFrame()
    
    def get_quote(self, symbol: str) -> Dict:
        """
        Get quote for a specific symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'RELIANCE')
        
        Returns:
            Dict containing quote information
        """
        try:
            if not self.client:
                return {}
                
            # Determine exchange based on symbol
            exchange = "NSE"
            if symbol in ["NIFTY", "BANKNIFTY", "NIFTY 50", "NIFTY BANK"]:
                exchange = "NSE"
                
            quote = self.client.get_quotes(
                exchange_code=exchange,
                stock_code=symbol
            )
            
            # Check if we got a valid response
            if quote and "Success" in quote:
                # Transform to standardized format
                quote_data = quote["Success"]
                result = {
                    "tradingsymbol": symbol,
                    "exchange": exchange,
                    "open": float(quote_data.get("open", 0)),
                    "high": float(quote_data.get("high", 0)),
                    "low": float(quote_data.get("low", 0)),
                    "lastPrice": float(quote_data.get("ltp", 0)),
                    "close": float(quote_data.get("close", 0)),
                    "totalTradedVolume": int(quote_data.get("volume", 0)),
                    "change": float(quote_data.get("change", 0)),
                    "pChange": float(quote_data.get("change_percentage", 0)),
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                return result
            
            return {}
        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {str(e)}")
            return {}
    
    def get_historical_data(
        self, 
        symbol: str, 
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Get historical data for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'RELIANCE')
            start_date: Start date in YYYY-MM-DD format or datetime
            end_date: End date in YYYY-MM-DD format or datetime
            interval: Data interval ('1m', '5m', '15m', '30m', '1h', '1d')
        
        Returns:
            DataFrame with historical price/volume data
        """
        try:
            if not self.client:
                return pd.DataFrame()
                
            # Convert dates to ICICI Breeze format (DD-MM-YYYY)
            if isinstance(start_date, datetime):
                start_date_str = start_date.strftime("%d-%m-%Y")
            else:
                # Convert from YYYY-MM-DD to DD-MM-YYYY
                parts = start_date.split('-')
                if len(parts) == 3:
                    start_date_str = f"{parts[2]}-{parts[1]}-{parts[0]}"
                else:
                    start_date_str = start_date
            
            if isinstance(end_date, datetime):
                end_date_str = end_date.strftime("%d-%m-%Y")
            else:
                # Convert from YYYY-MM-DD to DD-MM-YYYY
                parts = end_date.split('-')
                if len(parts) == 3:
                    end_date_str = f"{parts[2]}-{parts[1]}-{parts[0]}"
                else:
                    end_date_str = end_date
                
            # Map interval to ICICI Breeze format
            interval_mapping = {
                "1m": "1minute",
                "5m": "5minute",
                "15m": "15minute",
                "30m": "30minute",
                "1h": "1hour",
                "1d": "1day",
            }
            
            icici_interval = interval_mapping.get(interval, "1day")
            
            # Determine exchange
            exchange = "NSE"
            if any(idx in symbol for idx in ["NIFTY", "BANKNIFTY"]):
                exchange = "NSE"
            
            # Special handling for indices
            if symbol in ["NIFTY", "BANKNIFTY", "NIFTY 50", "NIFTY BANK"]:
                # Map to correct ICICI symbol format for indices
                if symbol == "NIFTY" or symbol == "NIFTY 50":
                    icici_symbol = "NIFTY 50"
                elif symbol == "BANKNIFTY" or symbol == "NIFTY BANK":
                    icici_symbol = "NIFTY BANK"
                else:
                    icici_symbol = symbol
                
                try:
                    # Specific method for index data
                    response = self.client.get_historical_data(
                        interval=icici_interval,
                        from_date=start_date_str,
                        to_date=end_date_str,
                        stock_code=icici_symbol,
                        exchange_code=exchange,
                        product_type="INDEX"
                    )
                except Exception as idx_error:
                    logger.warning(f"Error fetching index data: {str(idx_error)}. Trying standard method.")
                    response = None
            else:
                response = None
                
            # Try standard equity if index-specific method failed or not needed
            if response is None:
                try:
                    # Get historical data
                    response = self.client.get_historical_data(
                        exchange_code=exchange,
                        stock_code=symbol,
                        interval=icici_interval,
                        from_date=start_date_str,
                        to_date=end_date_str
                    )
                except Exception as std_error:
                    logger.error(f"Standard historical data method failed: {str(std_error)}")
                    response = None
            
            # Process ICICI response
            if response and isinstance(response, dict) and "Success" in response:
                data = response["Success"]
                df = pd.DataFrame(data)
            elif isinstance(response, pd.DataFrame):
                df = response
            else:
                # No fallback to yfinance, only use real API data
                df = pd.DataFrame()
            
            # Rename columns if needed
            column_map = {
                "datetime": "timestamp",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume"
            }
            
            df.rename(columns=column_map, inplace=True, errors='ignore')
            
            # Convert columns to appropriate types
            if not df.empty:
                # Convert timestamp to datetime
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                elif "datetime" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["datetime"])
                    df.drop("datetime", axis=1, inplace=True, errors='ignore')
                
                # Convert numeric columns
                numeric_cols = ["open", "high", "low", "close", "volume"]
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col])
            
                # Calculate change and change_pct if not present
                if "change" not in df.columns and "close" in df.columns:
                    df["change"] = df["close"].diff()
                    
                if "change_pct" not in df.columns and "close" in df.columns and "change" in df.columns:
                    df["change_pct"] = (df["change"] / df["close"].shift(1) * 100)
            
            return df
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {str(e)}")
            return pd.DataFrame()

# Backward compatibility aliases
NSEConnector = DhanConnector
BSEConnector = ICICIBreezeConnector

def get_api_credentials(section: str, provider: str) -> Dict:
    """
    Get API credentials from config file or environment variables.
    
    Args:
        section: Config section (e.g., 'market_data')
        provider: API provider (e.g., 'dhan', 'icici')
    
    Returns:
        Dict containing API credentials
    """
    try:
        # Try to load from config file
        config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config.yaml")
        
        if os.path.exists(config_file):
            try:
                import yaml
                with open(config_file, "r") as f:
                    config = yaml.safe_load(f)
                
                if config and "data_sources" in config and section in config["data_sources"] and provider in config["data_sources"][section]:
                    return config["data_sources"][section][provider]
            except Exception as e:
                logger.warning(f"Failed to load credentials from config file: {str(e)}")
        
        # Try environment variables as fallback
        if provider == "dhan":
            return {
                "client_id": os.environ.get("DHAN_CLIENT_ID", ""),
                "access_token": os.environ.get("DHAN_ACCESS_TOKEN", "")
            }
        elif provider == "icici":
            return {
                "api_key": os.environ.get("ICICI_API_KEY", ""),
                "api_secret": os.environ.get("ICICI_API_SECRET", ""),
                "session_token": os.environ.get("ICICI_SESSION_TOKEN", "")
            }
        else:
            return {}
    except Exception as e:
        logger.error(f"Error getting API credentials: {str(e)}")
        return {}