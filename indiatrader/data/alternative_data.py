"""
Alternative data connectors for news and social media.
"""

import requests
import pandas as pd
import logging
import time
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import json
import re

from indiatrader.data.config import get_api_credentials

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class NewsConnector:
    """
    Connector for financial news sources.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize news connector.
        
        Args:
            api_key: API key for news service. If None, will be loaded from config.
        """
        credentials = get_api_credentials("alternative_data", "news")
        self.api_key = api_key or credentials.get("api_key")
        
        # Check if credentials are available
        if not self.api_key:
            logger.warning("News API key not found in configuration. Some functionality may be limited.")
        
        # Supported news sources
        self.supported_sources = ["moneycontrol", "economictimes", "businessstandard"]
    
    def get_latest_news(self, source: str = "moneycontrol", limit: int = 10) -> pd.DataFrame:
        """
        Get latest news articles from a specific source.
        
        Args:
            source: News source name
            limit: Maximum number of articles to retrieve
        
        Returns:
            DataFrame containing news articles
        """
        if source not in self.supported_sources:
            logger.warning(f"Unsupported news source: {source}. Using moneycontrol instead.")
            source = "moneycontrol"
        
        # API endpoint would vary by provider
        # This is a placeholder implementation
        url = f"https://api.newsapi.example/{source}/latest?limit={limit}&apikey={self.api_key}"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if "articles" in data:
                # Convert to DataFrame
                df = pd.DataFrame(data["articles"])
                
                # Add timestamp and source columns
                df["timestamp"] = pd.to_datetime(df["publishedAt"])
                df["source"] = source
                
                return df
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch news from {source}: {str(e)}")
        
        # Return empty DataFrame if request failed
        return pd.DataFrame()
    
    def search_news(self, 
                   query: str, 
                   sources: Optional[List[str]] = None,
                   start_date: Optional[Union[str, datetime]] = None,
                   end_date: Optional[Union[str, datetime]] = None,
                   limit: int = 50) -> pd.DataFrame:
        """
        Search for news articles by keyword.
        
        Args:
            query: Search query string
            sources: List of news sources to search
            start_date: Start date for search range
            end_date: End date for search range
            limit: Maximum number of articles to retrieve
        
        Returns:
            DataFrame containing matching news articles
        """
        # Default to all supported sources if not specified
        if sources is None:
            sources = self.supported_sources
        else:
            # Filter out any unsupported sources
            sources = [s for s in sources if s in self.supported_sources]
            
            if not sources:
                logger.warning("No supported news sources specified. Using all available sources.")
                sources = self.supported_sources
        
        # Convert dates to string format if datetime objects
        if start_date is not None and isinstance(start_date, datetime):
            start_date = start_date.strftime("%Y-%m-%d")
        
        if end_date is not None and isinstance(end_date, datetime):
            end_date = end_date.strftime("%Y-%m-%d")
        
        # API endpoint would vary by provider
        # This is a placeholder implementation
        url = f"https://api.newsapi.example/search"
        
        params = {
            "q": query,
            "sources": ",".join(sources),
            "limit": limit,
            "apikey": self.api_key
        }
        
        if start_date:
            params["from"] = start_date
        
        if end_date:
            params["to"] = end_date
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if "articles" in data:
                # Convert to DataFrame
                df = pd.DataFrame(data["articles"])
                
                # Add timestamp column
                df["timestamp"] = pd.to_datetime(df["publishedAt"])
                
                return df
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to search news: {str(e)}")
        
        # Return empty DataFrame if request failed
        return pd.DataFrame()
    
    def extract_stock_mentions(self, text: str) -> List[str]:
        """
        Extract stock symbols mentioned in news text.
        
        Args:
            text: News article text
        
        Returns:
            List of stock symbols mentioned
        """
        # Pattern for NSE symbols: mostly uppercase letters 2-10 chars long
        # This is a simplified pattern and may need refinement
        pattern = r'\b[A-Z]{2,10}\b'
        
        # Extract matches
        matches = re.findall(pattern, text)
        
        # Filter out common words that might match the pattern
        common_words = {"CEO", "CFO", "NSE", "BSE", "SEBI", "RBI", "USD", "INR"}
        symbols = [match for match in matches if match not in common_words]
        
        return symbols


class SocialMediaConnector:
    """
    Connector for social media platforms.
    """
    
    def __init__(self, platform: str = "twitter"):
        """
        Initialize social media connector.
        
        Args:
            platform: Social media platform ('twitter' or 'reddit')
        """
        self.platform = platform.lower()
        
        if self.platform == "twitter":
            credentials = get_api_credentials("alternative_data", "twitter")
            self.api_key = credentials.get("api_key", "")
            self.api_secret = credentials.get("api_secret", "")
        
        elif self.platform == "reddit":
            credentials = get_api_credentials("alternative_data", "reddit")
            self.client_id = credentials.get("client_id", "")
            self.client_secret = credentials.get("client_secret", "")
        
        else:
            logger.warning(f"Unsupported platform: {platform}. Supported platforms are: twitter, reddit")
    
    def get_twitter_data(self, 
                        keywords: List[str], 
                        limit: int = 100, 
                        lang: str = "en",
                        days_back: int = 1) -> pd.DataFrame:
        """
        Get Twitter data for specified keywords.
        
        Args:
            keywords: List of keywords to search for
            limit: Maximum number of tweets to retrieve
            lang: Language filter
            days_back: Number of days to look back
        
        Returns:
            DataFrame containing tweets
        """
        if self.platform != "twitter":
            logger.error("This method is only available for Twitter platform.")
            return pd.DataFrame()
        
        if not self.api_key or not self.api_secret:
            logger.error("Twitter API credentials not configured.")
            return pd.DataFrame()
        
        # Twitter API v2 endpoint (placeholder)
        url = "https://api.twitter.com/2/tweets/search/recent"
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Format dates for API
        start_str = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Build query
        query = " OR ".join(keywords)
        
        params = {
            "query": query,
            "max_results": limit,
            "start_time": start_str,
            "tweet.fields": "created_at,public_metrics,entities",
            "expansions": "author_id",
            "user.fields": "name,username,public_metrics"
        }
        
        headers = {
            "Authorization": f"Bearer {self._get_twitter_bearer_token()}"
        }
        
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if "data" in data:
                # Convert to DataFrame
                tweets_df = pd.DataFrame(data["data"])
                
                # Add timestamp column
                tweets_df["timestamp"] = pd.to_datetime(tweets_df["created_at"])
                
                # Add user information
                if "includes" in data and "users" in data["includes"]:
                    users_df = pd.DataFrame(data["includes"]["users"])
                    # Merge user data with tweets
                    tweets_df = tweets_df.merge(users_df, left_on="author_id", right_on="id", suffixes=("", "_user"))
                
                return tweets_df
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch Twitter data: {str(e)}")
        
        # Return empty DataFrame if request failed
        return pd.DataFrame()
    
    def get_reddit_data(self, 
                       subreddits: List[str], 
                       limit: int = 100,
                       timeframe: str = "day") -> pd.DataFrame:
        """
        Get Reddit posts from specified subreddits.
        
        Args:
            subreddits: List of subreddit names
            limit: Maximum number of posts to retrieve per subreddit
            timeframe: Time frame filter ('hour', 'day', 'week', 'month', 'year', 'all')
        
        Returns:
            DataFrame containing Reddit posts
        """
        if self.platform != "reddit":
            logger.error("This method is only available for Reddit platform.")
            return pd.DataFrame()
        
        if not self.client_id or not self.client_secret:
            logger.error("Reddit API credentials not configured.")
            return pd.DataFrame()
        
        # Validate timeframe
        valid_timeframes = ["hour", "day", "week", "month", "year", "all"]
        if timeframe not in valid_timeframes:
            logger.warning(f"Invalid timeframe: {timeframe}. Using 'day' instead.")
            timeframe = "day"
        
        all_posts = []
        
        # Get posts from each subreddit
        for subreddit in subreddits:
            # Reddit API endpoint
            url = f"https://www.reddit.com/r/{subreddit}/top.json"
            
            params = {
                "t": timeframe,
                "limit": limit
            }
            
            headers = {
                "User-Agent": "IndiaTrader/0.1"
            }
            
            try:
                response = requests.get(url, params=params, headers=headers, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                
                if "data" in data and "children" in data["data"]:
                    posts = data["data"]["children"]
                    
                    # Extract post data
                    for post in posts:
                        post_data = post["data"]
                        all_posts.append({
                            "id": post_data.get("id"),
                            "title": post_data.get("title"),
                            "text": post_data.get("selftext"),
                            "score": post_data.get("score"),
                            "num_comments": post_data.get("num_comments"),
                            "created_utc": post_data.get("created_utc"),
                            "subreddit": post_data.get("subreddit"),
                            "permalink": post_data.get("permalink"),
                            "url": post_data.get("url")
                        })
            
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to fetch Reddit data for r/{subreddit}: {str(e)}")
        
        # Convert to DataFrame
        if all_posts:
            df = pd.DataFrame(all_posts)
            
            # Add timestamp column
            df["timestamp"] = pd.to_datetime(df["created_utc"], unit="s")
            
            return df
        
        # Return empty DataFrame if no data was collected
        return pd.DataFrame()
    
    def _get_twitter_bearer_token(self) -> str:
        """
        Get Twitter API bearer token using API key and secret.
        
        Returns:
            Bearer token string
        """
        if not self.api_key or not self.api_secret:
            logger.error("Twitter API credentials not configured.")
            return ""
        
        url = "https://api.twitter.com/oauth2/token"
        
        # Encode credentials
        import base64
        credentials = f"{self.api_key}:{self.api_secret}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        
        headers = {
            "Authorization": f"Basic {encoded_credentials}",
            "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"
        }
        
        data = "grant_type=client_credentials"
        
        try:
            response = requests.post(url, headers=headers, data=data, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if "access_token" in data:
                return data["access_token"]
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get Twitter bearer token: {str(e)}")
        
        return ""