"""
Market data connectors for Indian exchanges.
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

from indiatrader.data.config import get_api_credentials, get_market_symbols

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class NSEConnector:
    """
    Connector for National Stock Exchange (NSE) India.
    """
    
    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None):
        """
        Initialize NSE connector.
        
        Args:
            api_key: NSE API key. If None, will be loaded from config.
            api_url: NSE API base URL. If None, will be loaded from config.
        """
        credentials = get_api_credentials("market_data", "nse")
        self.api_key = api_key or credentials.get("api_key")
        self.api_url = api_url or credentials.get("api_url")
        
        # Check if credentials are available
        if not self.api_key:
            logger.warning("NSE API key not found in configuration. Some functionality may be limited.")
        
        # Default headers
        self.headers = {
            "Accept": "application/json",
            "X-API-KEY": self.api_key,
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
        }
    
    def get_market_status(self) -> Dict:
        """
        Get current market status.
        
        Returns:
            Dict containing market status information
        """
        url = f"{self.api_url}/marketStatus"
        response = self._make_request(url)
        return response
    
    def get_indices(self) -> pd.DataFrame:
        """
        Get current index values.
        
        Returns:
            DataFrame containing index information
        """
        url = f"{self.api_url}/allIndices"
        response = self._make_request(url)
        
        if response and "data" in response:
            return pd.DataFrame(response["data"])
        
        return pd.DataFrame()
    
    def get_quote(self, symbol: str) -> Dict:
        """
        Get quote for a specific symbol.
        
        Args:
            symbol: NSE symbol (e.g., 'RELIANCE')
        
        Returns:
            Dict containing quote information
        """
        url = f"{self.api_url}/quote-equity?symbol={symbol}"
        response = self._make_request(url)
        return response
    
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
            symbol: NSE symbol (e.g., 'RELIANCE')
            start_date: Start date in YYYY-MM-DD format or datetime
            end_date: End date in YYYY-MM-DD format or datetime
            interval: Data interval ('1m', '5m', '15m', '30m', '1h', '1d')
        
        Returns:
            DataFrame with historical price/volume data
        """
        # Convert dates to string format if datetime objects
        if isinstance(start_date, datetime):
            start_date = start_date.strftime("%Y-%m-%d")
        
        if isinstance(end_date, datetime):
            end_date = end_date.strftime("%Y-%m-%d")
        
        interval_mapping = {
            "1m": "MINUTE",
            "5m": "5MINUTE",
            "15m": "15MINUTE",
            "30m": "30MINUTE",
            "1h": "HOUR",
            "1d": "DAY",
        }
        
        nse_interval = interval_mapping.get(interval, "DAY")
        
        url = f"{self.api_url}/historical/cm/equity?symbol={symbol}&from={start_date}&to={end_date}&series=EQ&interval={nse_interval}"
        response = self._make_request(url)
        
        if response and "data" in response:
            data = response["data"]
            df = pd.DataFrame(data)
            
            # Convert columns to appropriate types
            if not df.empty:
                # Convert timestamp to datetime
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                
                # Convert numeric columns
                numeric_cols = ["open", "high", "low", "close", "volume"]
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col])
            
            return df
        
        return pd.DataFrame()
    
    def get_order_book(self, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get Level II order book data for a symbol.
        
        Args:
            symbol: NSE symbol (e.g., 'RELIANCE')
        
        Returns:
            Tuple of (bids_df, asks_df) DataFrames containing order book data
        """
        url = f"{self.api_url}/market-depth?symbol={symbol}"
        response = self._make_request(url)
        
        bids_df = pd.DataFrame()
        asks_df = pd.DataFrame()
        
        if response and "data" in response:
            data = response["data"]
            
            if "bids" in data:
                bids_df = pd.DataFrame(data["bids"])
            
            if "asks" in data:
                asks_df = pd.DataFrame(data["asks"])
        
        return bids_df, asks_df
    
    def save_to_parquet(
        self, 
        symbol: str, 
        data: pd.DataFrame, 
        data_type: str,
        interval: str = "1d",
        base_path: str = "./data"
    ) -> str:
        """
        Save market data to Parquet format.
        
        Args:
            symbol: Symbol name
            data: DataFrame to save
            data_type: Type of data ('ohlcv', 'quotes', 'orderbook')
            interval: Data interval for OHLCV data
            base_path: Base directory to save files
        
        Returns:
            Path to saved file
        """
        import os
        
        # Create directory structure if it doesn't exist
        os.makedirs(f"{base_path}/nse/{data_type}/{interval}", exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f"{base_path}/nse/{data_type}/{interval}/{symbol}_{timestamp}.parquet"
        
        # Convert to PyArrow Table and write to Parquet
        table = pa.Table.from_pandas(data)
        pq.write_table(table, filename)
        
        logger.info(f"Saved {symbol} {data_type} data to {filename}")
        return filename
    
    def _make_request(self, url: str, max_retries: int = 3, retry_delay: int = 2) -> Dict:
        """
        Make HTTP request with retry logic.
        
        Args:
            url: API endpoint URL
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        
        Returns:
            API response as dictionary
        """
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                response.raise_for_status()  # Raise exception for 4XX/5XX responses
                
                return response.json()
            
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt+1}/{max_retries}): {str(e)}")
                
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed after {max_retries} attempts: {str(e)}")
        
        return {}


class BSEConnector:
    """
    Connector for Bombay Stock Exchange (BSE) India.
    """
    
    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None):
        """
        Initialize BSE connector.
        
        Args:
            api_key: BSE API key. If None, will be loaded from config.
            api_url: BSE API base URL. If None, will be loaded from config.
        """
        credentials = get_api_credentials("market_data", "bse")
        self.api_key = api_key or credentials.get("api_key")
        self.api_url = api_url or credentials.get("api_url")
        
        # Default headers
        self.headers = {
            "Accept": "application/json",
            "X-API-KEY": self.api_key,
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
        }
    
    def get_quote(self, symbol: str) -> Dict:
        """
        Get quote for a specific symbol.
        
        Args:
            symbol: BSE scrip code (e.g., '500325' for Reliance)
        
        Returns:
            Dict containing quote information
        """
        url = f"{self.api_url}/quote?scripcode={symbol}"
        response = self._make_request(url)
        return response
    
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
            symbol: BSE scrip code
            start_date: Start date in YYYY-MM-DD format or datetime
            end_date: End date in YYYY-MM-DD format or datetime
            interval: Data interval ('1d' only for BSE)
        
        Returns:
            DataFrame with historical price/volume data
        """
        # Convert dates to string format if datetime objects
        if isinstance(start_date, datetime):
            start_date = start_date.strftime("%Y%m%d")
        else:
            # Convert YYYY-MM-DD to YYYYMMDD
            start_date = start_date.replace("-", "")
        
        if isinstance(end_date, datetime):
            end_date = end_date.strftime("%Y%m%d")
        else:
            # Convert YYYY-MM-DD to YYYYMMDD
            end_date = end_date.replace("-", "")
        
        url = f"{self.api_url}/history?scripcode={symbol}&fromdate={start_date}&todate={end_date}&flag=0"
        response = self._make_request(url)
        
        if response and "data" in response:
            data = response["data"]
            df = pd.DataFrame(data)
            
            # Convert columns to appropriate types
            if not df.empty:
                # Convert date to datetime
                df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
                
                # Convert numeric columns
                numeric_cols = ["open", "high", "low", "close", "volume"]
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col])
            
            return df
        
        return pd.DataFrame()
    
    def _make_request(self, url: str, max_retries: int = 3, retry_delay: int = 2) -> Dict:
        """
        Make HTTP request with retry logic.
        
        Args:
            url: API endpoint URL
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        
        Returns:
            API response as dictionary
        """
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                response.raise_for_status()  # Raise exception for 4XX/5XX responses
                
                return response.json()
            
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt+1}/{max_retries}): {str(e)}")
                
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed after {max_retries} attempts: {str(e)}")
        
        return {}