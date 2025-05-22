"""
Data ingestion pipeline for Indian stock market data.
"""

import argparse
import logging
import time
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import os
import yaml

from indiatrader.data.config import load_config, get_market_symbols
from indiatrader.data.market_data import NSEConnector, BSEConnector
from indiatrader.data.alternative_data import NewsConnector, SocialMediaConnector

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class DataIngestionPipeline:
    """
    Pipeline for ingesting market data and alternative data.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize data ingestion pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        
        # Initialize data connectors
        self.nse = NSEConnector()
        self.bse = BSEConnector()
        self.news = NewsConnector()
        self.twitter = SocialMediaConnector("twitter")
        self.reddit = SocialMediaConnector("reddit")
        
        # Configure data directory
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Track ingestion stats
        self.stats = {
            "started_at": datetime.now(),
            "market_data_files": 0,
            "news_data_files": 0,
            "social_data_files": 0,
            "errors": 0
        }
    
    def ingest_market_data(self, 
                          exchange: str = "nse", 
                          symbols: Optional[List[str]] = None,
                          intervals: Optional[List[str]] = None,
                          days_back: int = 1) -> Dict[str, Any]:
        """
        Ingest market data for specified symbols and intervals.
        
        Args:
            exchange: Exchange to get data from ('nse' or 'bse')
            symbols: List of symbols to get data for. If None, uses symbols from config.
            intervals: List of intervals to get data for. If None, uses intervals from config.
            days_back: Number of days of historical data to ingest
        
        Returns:
            Dictionary of ingestion statistics
        """
        # Default to symbols from config if not specified
        if symbols is None:
            symbols = get_market_symbols(exchange.lower())
        
        # Default to intervals from config if not specified
        if intervals is None:
            intervals = self.config["data_sources"]["market_data"][exchange.lower()].get("timeframes", ["1d"])
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        logger.info(f"Ingesting {exchange.upper()} data for {len(symbols)} symbols, {len(intervals)} intervals, from {start_date.date()} to {end_date.date()}")
        
        for symbol in symbols:
            for interval in intervals:
                try:
                    # Get connector based on exchange
                    if exchange.lower() == "nse":
                        connector = self.nse
                    elif exchange.lower() == "bse":
                        connector = self.bse
                    else:
                        logger.error(f"Unsupported exchange: {exchange}")
                        continue
                    
                    # Get historical data
                    df = connector.get_historical_data(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        interval=interval
                    )
                    
                    if df.empty:
                        logger.warning(f"No data retrieved for {symbol} ({interval})")
                        continue
                    
                    # Save to Parquet
                    file_path = os.path.join(self.data_dir, exchange.lower(), "ohlcv", interval)
                    os.makedirs(file_path, exist_ok=True)
                    
                    timestamp = datetime.now().strftime("%Y%m%d")
                    file_name = f"{symbol}_{timestamp}.parquet"
                    full_path = os.path.join(file_path, file_name)
                    
                    df.to_parquet(full_path, index=False)
                    logger.info(f"Saved {symbol} ({interval}) data to {full_path}")
                    
                    self.stats["market_data_files"] += 1
                    
                    # Add slight delay to avoid API rate limits
                    time.sleep(0.5)
                
                except Exception as e:
                    logger.error(f"Error ingesting data for {symbol} ({interval}): {str(e)}")
                    self.stats["errors"] += 1
        
        # Update statistics
        self.stats["market_data_completed_at"] = datetime.now()
        return self.stats
    
    def ingest_news_data(self, sources: Optional[List[str]] = None, limit: int = 100) -> Dict[str, Any]:
        """
        Ingest news data from specified sources.
        
        Args:
            sources: List of news sources to ingest from
            limit: Maximum number of articles to ingest per source
        
        Returns:
            Dictionary of ingestion statistics
        """
        # Default to sources from config if not specified
        if sources is None:
            sources = self.config["data_sources"]["alternative_data"]["news"].get("sources", ["moneycontrol"])
        
        logger.info(f"Ingesting news data from {sources}")
        
        for source in sources:
            try:
                # Get latest news
                df = self.news.get_latest_news(source=source, limit=limit)
                
                if df.empty:
                    logger.warning(f"No news data retrieved from {source}")
                    continue
                
                # Save to Parquet
                file_path = os.path.join(self.data_dir, "news")
                os.makedirs(file_path, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d")
                file_name = f"{source}_{timestamp}.parquet"
                full_path = os.path.join(file_path, file_name)
                
                df.to_parquet(full_path, index=False)
                logger.info(f"Saved {source} news data to {full_path}")
                
                self.stats["news_data_files"] += 1
            
            except Exception as e:
                logger.error(f"Error ingesting news data from {source}: {str(e)}")
                self.stats["errors"] += 1
        
        # Update statistics
        self.stats["news_data_completed_at"] = datetime.now()
        return self.stats
    
    def ingest_social_media_data(self, platform: str = "twitter") -> Dict[str, Any]:
        """
        Ingest social media data.
        
        Args:
            platform: Social media platform ('twitter' or 'reddit')
        
        Returns:
            Dictionary of ingestion statistics
        """
        logger.info(f"Ingesting {platform} data")
        
        try:
            if platform.lower() == "twitter":
                # Get keywords from config
                keywords = self.config["data_sources"]["alternative_data"]["social"].get("twitter", {}).get("keywords", [])
                
                if not keywords:
                    logger.warning("No Twitter keywords configured")
                    return self.stats
                
                # Get Twitter data
                df = self.twitter.get_twitter_data(keywords=keywords, limit=100, days_back=1)
                
                if df.empty:
                    logger.warning("No Twitter data retrieved")
                    return self.stats
                
                # Save to Parquet
                file_path = os.path.join(self.data_dir, "social", "twitter")
                os.makedirs(file_path, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d")
                file_name = f"twitter_{timestamp}.parquet"
                full_path = os.path.join(file_path, file_name)
                
                df.to_parquet(full_path, index=False)
                logger.info(f"Saved Twitter data to {full_path}")
                
                self.stats["social_data_files"] += 1
            
            elif platform.lower() == "reddit":
                # Get subreddits from config
                subreddits = self.config["data_sources"]["alternative_data"]["social"].get("reddit", {}).get("subreddits", [])
                
                if not subreddits:
                    logger.warning("No Reddit subreddits configured")
                    return self.stats
                
                # Get Reddit data
                df = self.reddit.get_reddit_data(subreddits=subreddits, limit=100, timeframe="day")
                
                if df.empty:
                    logger.warning("No Reddit data retrieved")
                    return self.stats
                
                # Save to Parquet
                file_path = os.path.join(self.data_dir, "social", "reddit")
                os.makedirs(file_path, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d")
                file_name = f"reddit_{timestamp}.parquet"
                full_path = os.path.join(file_path, file_name)
                
                df.to_parquet(full_path, index=False)
                logger.info(f"Saved Reddit data to {full_path}")
                
                self.stats["social_data_files"] += 1
            
            else:
                logger.error(f"Unsupported social media platform: {platform}")
        
        except Exception as e:
            logger.error(f"Error ingesting {platform} data: {str(e)}")
            self.stats["errors"] += 1
        
        # Update statistics
        self.stats["social_data_completed_at"] = datetime.now()
        return self.stats
    
    def run_full_ingestion(self) -> Dict[str, Any]:
        """
        Run full data ingestion pipeline.
        
        Returns:
            Dictionary of ingestion statistics
        """
        logger.info("Starting full data ingestion pipeline")
        
        # Ingest market data
        self.ingest_market_data(exchange="nse")
        self.ingest_market_data(exchange="bse")
        
        # Ingest news data
        self.ingest_news_data()
        
        # Ingest social media data
        self.ingest_social_media_data("twitter")
        self.ingest_social_media_data("reddit")
        
        # Update statistics
        self.stats["completed_at"] = datetime.now()
        self.stats["total_duration"] = (self.stats["completed_at"] - self.stats["started_at"]).total_seconds()
        
        logger.info(f"Data ingestion completed in {self.stats['total_duration']:.2f} seconds")
        logger.info(f"Files created: {self.stats['market_data_files'] + self.stats['news_data_files'] + self.stats['social_data_files']}")
        logger.info(f"Errors: {self.stats['errors']}")
        
        return self.stats


def main():
    """
    Main entry point for data ingestion.
    """
    parser = argparse.ArgumentParser(description="Indian Stock Market Data Ingestion")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--exchange", type=str, choices=["nse", "bse", "all"], default="all",
                        help="Exchange to ingest data from")
    parser.add_argument("--data-types", type=str, nargs="+", 
                        choices=["market", "news", "social", "all"], default=["all"],
                        help="Types of data to ingest")
    parser.add_argument("--days", type=int, default=1,
                        help="Number of days of historical data to ingest")
    
    args = parser.parse_args()
    
    # Configure logging to file and console
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"ingestion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Initialize pipeline
    pipeline = DataIngestionPipeline(args.config)
    
    try:
        if "all" in args.data_types:
            # Run full ingestion
            stats = pipeline.run_full_ingestion()
        else:
            # Run selective ingestion
            stats = {}
            
            if "market" in args.data_types:
                if args.exchange == "all" or args.exchange == "nse":
                    stats["nse"] = pipeline.ingest_market_data(exchange="nse", days_back=args.days)
                
                if args.exchange == "all" or args.exchange == "bse":
                    stats["bse"] = pipeline.ingest_market_data(exchange="bse", days_back=args.days)
            
            if "news" in args.data_types:
                stats["news"] = pipeline.ingest_news_data()
            
            if "social" in args.data_types:
                stats["twitter"] = pipeline.ingest_social_media_data("twitter")
                stats["reddit"] = pipeline.ingest_social_media_data("reddit")
        
        # Save statistics to file
        stats_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stats")
        os.makedirs(stats_dir, exist_ok=True)
        
        stats_file = os.path.join(stats_dir, f"ingestion_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml")
        
        # Convert datetime objects to strings for YAML serialization
        for key, value in stats.items():
            if isinstance(value, datetime):
                stats[key] = value.isoformat()
        
        with open(stats_file, "w") as f:
            yaml.dump(stats, f)
        
        logger.info(f"Statistics saved to {stats_file}")
    
    except Exception as e:
        logger.error(f"Error in data ingestion pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    main()