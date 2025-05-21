"""
Feature processing pipeline for combining multiple feature types.
"""

import pandas as pd
import numpy as np
import logging
import os
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import pyarrow as pa
import pyarrow.parquet as pq

from indiatrader.data.config import load_config
from indiatrader.features.technical import TechnicalFeatureGenerator
from indiatrader.features.order_flow import OrderFlowFeatureGenerator
from indiatrader.features.nlp import NLPFeatureGenerator

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class FeatureProcessor:
    """
    Process and combine multiple feature types for model training.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize feature processor.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        
        # Initialize feature generators
        self.technical_generator = TechnicalFeatureGenerator()
        self.order_flow_generator = OrderFlowFeatureGenerator()
        self.nlp_generator = NLPFeatureGenerator()
        
        # Configure data and feature directories
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.data_dir = os.path.join(root_dir, "data")
        self.feature_dir = os.path.join(root_dir, "features")
        
        os.makedirs(self.feature_dir, exist_ok=True)
    
    def process_market_data(self, 
                           symbol: str, 
                           exchange: str = "nse",
                           interval: str = "1d",
                           start_date: Optional[Union[str, datetime]] = None,
                           end_date: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """
        Process market data to generate technical and order flow features.
        
        Args:
            symbol: Symbol to process
            exchange: Exchange name ('nse' or 'bse')
            interval: Data interval ('1m', '5m', '15m', '1h', '1d')
            start_date: Start date for data processing
            end_date: End date for data processing
        
        Returns:
            DataFrame with combined features
        """
        # Load market data
        market_data = self._load_market_data(symbol, exchange, interval, start_date, end_date)
        
        if market_data.empty:
            logger.warning(f"No market data found for {symbol} ({exchange}) with interval {interval}")
            return pd.DataFrame()
        
        # Load order book data if available
        orderbook_data = self._load_orderbook_data(symbol, exchange, interval, start_date, end_date)
        
        # Generate technical features
        tech_config = self.config["features"]["technical"]
        market_data = self.technical_generator.generate_features(market_data, tech_config)
        
        # Generate order flow features if order book data is available
        order_flow_config = self.config["features"]["order_flow"]
        market_data = self.order_flow_generator.generate_features(
            market_data, orderbook_data, order_flow_config
        )
        
        # Clean up data (remove rows with too many NaN values)
        threshold = 0.5  # At least 50% of columns should have values
        min_non_na = int(threshold * market_data.shape[1])
        market_data = market_data.dropna(thresh=min_non_na)
        
        # Forward fill remaining NaN values
        market_data = market_data.fillna(method="ffill")
        
        # Backward fill any remaining NaNs at the beginning
        market_data = market_data.fillna(method="bfill")
        
        # Replace any remaining NaNs with zeros
        market_data = market_data.fillna(0)
        
        return market_data
    
    def process_news_data(self, 
                         symbols: List[str], 
                         start_date: Optional[Union[str, datetime]] = None,
                         end_date: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """
        Process news data to generate NLP features.
        
        Args:
            symbols: List of symbols to filter news for
            start_date: Start date for data processing
            end_date: End date for data processing
        
        Returns:
            DataFrame with NLP features
        """
        # Load news data
        news_data = self._load_news_data(start_date, end_date)
        
        if news_data.empty:
            logger.warning("No news data found")
            return pd.DataFrame()
        
        # Filter news related to specified symbols
        if symbols:
            filtered_news = []
            
            for _, row in news_data.iterrows():
                text = row["title"] + " " + (row["description"] if "description" in row else "")
                
                # Check if any symbol is mentioned in the text
                if any(symbol.lower() in text.lower() for symbol in symbols):
                    filtered_news.append(row)
            
            if filtered_news:
                news_data = pd.DataFrame(filtered_news)
            else:
                logger.warning(f"No news found for symbols: {symbols}")
                return pd.DataFrame()
        
        # Generate NLP features
        nlp_config = self.config["features"]["nlp"]
        
        # Define text column (could be title, description, or combined)
        if "description" in news_data.columns:
            news_data["text"] = news_data["title"] + " " + news_data["description"].fillna("")
            text_column = "text"
        else:
            text_column = "title"
        
        # Generate NLP features
        news_data = self.nlp_generator.generate_features(news_data, nlp_config, text_column=text_column)
        
        # Clean up data
        news_data = news_data.fillna(0)
        
        return news_data
    
    def merge_market_and_news_data(self, 
                                  market_data: pd.DataFrame, 
                                  news_data: pd.DataFrame) -> pd.DataFrame:
        """
        Merge market data with news data based on timestamps.
        
        Args:
            market_data: DataFrame with market features
            news_data: DataFrame with news features
        
        Returns:
            DataFrame with combined features
        """
        if market_data.empty or news_data.empty:
            logger.warning("Cannot merge data: market or news data is empty")
            return market_data  # Return market data as fallback
        
        # Ensure timestamp columns are datetime type
        if "timestamp" not in market_data.columns:
            logger.warning("Timestamp column not found in market data")
            return market_data
        
        if "timestamp" not in news_data.columns:
            logger.warning("Timestamp column not found in news data")
            return market_data
        
        market_data["timestamp"] = pd.to_datetime(market_data["timestamp"])
        news_data["timestamp"] = pd.to_datetime(news_data["timestamp"])
        
        # Sort data by timestamp
        market_data = market_data.sort_values("timestamp")
        news_data = news_data.sort_values("timestamp")
        
        # Select relevant news features to merge
        news_features = news_data[["timestamp", "sentiment_score", "sentiment_positive", 
                                   "sentiment_negative", "sentiment_neutral"]]
        
        # For each market data point, find the most recent news
        merged_data = market_data.copy()
        
        # Add news features with forward fill
        for feature in news_features.columns:
            if feature != "timestamp":
                merged_data[f"news_{feature}"] = np.nan
        
        # Loop through market data points
        for i, market_row in merged_data.iterrows():
            market_time = market_row["timestamp"]
            
            # Find news before this market data point
            relevant_news = news_features[news_features["timestamp"] <= market_time]
            
            if not relevant_news.empty:
                # Get most recent news
                most_recent = relevant_news.iloc[-1]
                
                # Add news features to market data
                for feature in news_features.columns:
                    if feature != "timestamp":
                        merged_data.loc[i, f"news_{feature}"] = most_recent[feature]
        
        # Forward fill news features
        news_cols = [f"news_{feature}" for feature in news_features.columns if feature != "timestamp"]
        merged_data[news_cols] = merged_data[news_cols].fillna(method="ffill")
        
        # Fill any remaining NaNs with neutral values
        merged_data["news_sentiment_score"] = merged_data["news_sentiment_score"].fillna(0)
        merged_data["news_sentiment_positive"] = merged_data["news_sentiment_positive"].fillna(0)
        merged_data["news_sentiment_negative"] = merged_data["news_sentiment_negative"].fillna(0)
        merged_data["news_sentiment_neutral"] = merged_data["news_sentiment_neutral"].fillna(1)
        
        return merged_data
    
    def create_model_ready_features(self, 
                                   symbol: str, 
                                   exchange: str = "nse",
                                   interval: str = "1d",
                                   include_news: bool = True,
                                   start_date: Optional[Union[str, datetime]] = None,
                                   end_date: Optional[Union[str, datetime]] = None,
                                   target_horizon: int = 1,
                                   save_to_file: bool = True) -> pd.DataFrame:
        """
        Create model-ready features by combining all feature types and adding target variables.
        
        Args:
            symbol: Symbol to process
            exchange: Exchange name ('nse' or 'bse')
            interval: Data interval ('1m', '5m', '15m', '1h', '1d')
            include_news: Whether to include news features
            start_date: Start date for data processing
            end_date: End date for data processing
            target_horizon: Forecast horizon in terms of intervals
            save_to_file: Whether to save the features to a file
        
        Returns:
            DataFrame with all features and target variables
        """
        # Process market data
        market_data = self.process_market_data(symbol, exchange, interval, start_date, end_date)
        
        if market_data.empty:
            logger.error(f"Failed to process market data for {symbol}")
            return pd.DataFrame()
        
        # Add news data if requested
        if include_news:
            news_data = self.process_news_data([symbol], start_date, end_date)
            
            if not news_data.empty:
                market_data = self.merge_market_and_news_data(market_data, news_data)
        
        # Add target variables
        if "close" in market_data.columns:
            close_col = "close"
        elif "Close" in market_data.columns:
            close_col = "Close"
        else:
            logger.warning("Close price column not found. Cannot create target variables.")
            close_col = None
        
        if close_col:
            # Future return
            market_data[f"target_return_{target_horizon}"] = market_data[close_col].pct_change(target_horizon).shift(-target_horizon)
            
            # Future direction (1 for up, 0 for down)
            market_data[f"target_direction_{target_horizon}"] = np.where(
                market_data[f"target_return_{target_horizon}"] > 0, 1, 0
            )
            
            # Future volatility (absolute return)
            market_data[f"target_volatility_{target_horizon}"] = market_data[f"target_return_{target_horizon}"].abs()
        
        # Save to file if requested
        if save_to_file:
            os.makedirs(os.path.join(self.feature_dir, exchange, interval), exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d")
            file_path = os.path.join(self.feature_dir, exchange, interval, f"{symbol}_features_{timestamp}.parquet")
            
            # Convert to PyArrow Table and write to Parquet
            table = pa.Table.from_pandas(market_data)
            pq.write_table(table, file_path)
            
            logger.info(f"Saved features for {symbol} to {file_path}")
        
        return market_data
    
    def _load_market_data(self, 
                         symbol: str, 
                         exchange: str,
                         interval: str,
                         start_date: Optional[Union[str, datetime]] = None,
                         end_date: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """
        Load market data from Parquet files.
        
        Args:
            symbol: Symbol to load
            exchange: Exchange name
            interval: Data interval
            start_date: Start date filter
            end_date: End date filter
        
        Returns:
            DataFrame with market data
        """
        market_data_path = os.path.join(self.data_dir, exchange.lower(), "ohlcv", interval)
        
        if not os.path.exists(market_data_path):
            logger.warning(f"Market data path does not exist: {market_data_path}")
            return pd.DataFrame()
        
        # Find Parquet files for the symbol
        import glob
        symbol_files = glob.glob(os.path.join(market_data_path, f"{symbol}_*.parquet"))
        
        if not symbol_files:
            logger.warning(f"No Parquet files found for {symbol} in {market_data_path}")
            return pd.DataFrame()
        
        # Load and combine all files
        dfs = []
        
        for file_path in symbol_files:
            try:
                df = pd.read_parquet(file_path)
                dfs.append(df)
            except Exception as e:
                logger.error(f"Failed to read Parquet file {file_path}: {str(e)}")
        
        if not dfs:
            logger.warning(f"No valid data loaded for {symbol}")
            return pd.DataFrame()
        
        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Drop duplicates based on timestamp
        if "timestamp" in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset=["timestamp"])
            combined_df = combined_df.sort_values("timestamp")
        
        # Apply date filters if provided
        if start_date is not None or end_date is not None:
            if "timestamp" not in combined_df.columns:
                logger.warning("Cannot apply date filters: timestamp column not found")
            else:
                # Ensure timestamp is datetime
                combined_df["timestamp"] = pd.to_datetime(combined_df["timestamp"])
                
                # Apply start date filter
                if start_date is not None:
                    if isinstance(start_date, str):
                        start_date = pd.to_datetime(start_date)
                    
                    combined_df = combined_df[combined_df["timestamp"] >= start_date]
                
                # Apply end date filter
                if end_date is not None:
                    if isinstance(end_date, str):
                        end_date = pd.to_datetime(end_date)
                    
                    combined_df = combined_df[combined_df["timestamp"] <= end_date]
        
        return combined_df
    
    def _load_orderbook_data(self, 
                            symbol: str, 
                            exchange: str,
                            interval: str,
                            start_date: Optional[Union[str, datetime]] = None,
                            end_date: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """
        Load order book data from Parquet files.
        
        Args:
            symbol: Symbol to load
            exchange: Exchange name
            interval: Data interval
            start_date: Start date filter
            end_date: End date filter
        
        Returns:
            DataFrame with order book data
        """
        orderbook_path = os.path.join(self.data_dir, exchange.lower(), "orderbook")
        
        if not os.path.exists(orderbook_path):
            logger.info(f"Order book path does not exist: {orderbook_path}")
            return None
        
        # Find Parquet files for the symbol
        import glob
        symbol_files = glob.glob(os.path.join(orderbook_path, f"{symbol}_*.parquet"))
        
        if not symbol_files:
            logger.info(f"No order book Parquet files found for {symbol} in {orderbook_path}")
            return None
        
        # Load and combine all files
        dfs = []
        
        for file_path in symbol_files:
            try:
                df = pd.read_parquet(file_path)
                dfs.append(df)
            except Exception as e:
                logger.error(f"Failed to read order book Parquet file {file_path}: {str(e)}")
        
        if not dfs:
            logger.info(f"No valid order book data loaded for {symbol}")
            return None
        
        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Apply date filters if provided
        if start_date is not None or end_date is not None:
            if "timestamp" not in combined_df.columns:
                logger.warning("Cannot apply date filters: timestamp column not found in order book data")
            else:
                # Ensure timestamp is datetime
                combined_df["timestamp"] = pd.to_datetime(combined_df["timestamp"])
                
                # Apply start date filter
                if start_date is not None:
                    if isinstance(start_date, str):
                        start_date = pd.to_datetime(start_date)
                    
                    combined_df = combined_df[combined_df["timestamp"] >= start_date]
                
                # Apply end date filter
                if end_date is not None:
                    if isinstance(end_date, str):
                        end_date = pd.to_datetime(end_date)
                    
                    combined_df = combined_df[combined_df["timestamp"] <= end_date]
        
        return combined_df
    
    def _load_news_data(self, 
                       start_date: Optional[Union[str, datetime]] = None,
                       end_date: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """
        Load news data from Parquet files.
        
        Args:
            start_date: Start date filter
            end_date: End date filter
        
        Returns:
            DataFrame with news data
        """
        news_path = os.path.join(self.data_dir, "news")
        
        if not os.path.exists(news_path):
            logger.warning(f"News data path does not exist: {news_path}")
            return pd.DataFrame()
        
        # Find all Parquet files
        import glob
        news_files = glob.glob(os.path.join(news_path, "*.parquet"))
        
        if not news_files:
            logger.warning(f"No news Parquet files found in {news_path}")
            return pd.DataFrame()
        
        # Load and combine all files
        dfs = []
        
        for file_path in news_files:
            try:
                df = pd.read_parquet(file_path)
                dfs.append(df)
            except Exception as e:
                logger.error(f"Failed to read news Parquet file {file_path}: {str(e)}")
        
        if not dfs:
            logger.warning("No valid news data loaded")
            return pd.DataFrame()
        
        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Drop duplicates based on title or URL if available
        if "title" in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset=["title"])
        elif "url" in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset=["url"])
        
        # Apply date filters if provided
        if start_date is not None or end_date is not None:
            if "timestamp" not in combined_df.columns and "publishedAt" in combined_df.columns:
                combined_df["timestamp"] = combined_df["publishedAt"]
            
            if "timestamp" not in combined_df.columns:
                logger.warning("Cannot apply date filters: timestamp column not found in news data")
            else:
                # Ensure timestamp is datetime
                combined_df["timestamp"] = pd.to_datetime(combined_df["timestamp"])
                
                # Apply start date filter
                if start_date is not None:
                    if isinstance(start_date, str):
                        start_date = pd.to_datetime(start_date)
                    
                    combined_df = combined_df[combined_df["timestamp"] >= start_date]
                
                # Apply end date filter
                if end_date is not None:
                    if isinstance(end_date, str):
                        end_date = pd.to_datetime(end_date)
                    
                    combined_df = combined_df[combined_df["timestamp"] <= end_date]
        
        return combined_df


def main():
    """
    Main entry point for feature processing.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Feature Processing for Indian Stock Market")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--symbol", type=str, required=True, help="Symbol to process")
    parser.add_argument("--exchange", type=str, choices=["nse", "bse"], default="nse",
                        help="Exchange name")
    parser.add_argument("--interval", type=str, choices=["1m", "5m", "15m", "1h", "1d"], default="1d",
                        help="Data interval")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--target-horizon", type=int, default=1,
                        help="Forecast horizon in terms of intervals")
    parser.add_argument("--no-news", action="store_true", help="Exclude news features")
    
    args = parser.parse_args()
    
    # Configure logging to file and console
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Initialize processor
    processor = FeatureProcessor(args.config)
    
    try:
        # Process features
        features = processor.create_model_ready_features(
            symbol=args.symbol,
            exchange=args.exchange,
            interval=args.interval,
            include_news=not args.no_news,
            start_date=args.start_date,
            end_date=args.end_date,
            target_horizon=args.target_horizon,
            save_to_file=True
        )
        
        if features.empty:
            logger.error(f"Failed to create features for {args.symbol}")
        else:
            logger.info(f"Successfully created features for {args.symbol} with shape {features.shape}")
    
    except Exception as e:
        logger.error(f"Error in feature processing: {str(e)}")
        raise


if __name__ == "__main__":
    main()