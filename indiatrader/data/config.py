"""
Configuration utilities for data ingestion.
"""

import os
import yaml
from typing import Dict, Any, List, Optional


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file. If None, will look for config.yaml 
                    in the project root directory.
    
    Returns:
        Dict containing configuration parameters.
    """
    if config_path is None:
        # Determine path relative to project root
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(root_dir, "config.yaml")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def get_market_symbols(exchange: str) -> List[str]:
    """
    Get list of symbols configured for a specific exchange.
    
    Args:
        exchange: Exchange name (e.g., 'nse', 'bse')
    
    Returns:
        List of symbol strings
    """
    config = load_config()
    return config["data_sources"]["market_data"].get(exchange, {}).get("symbols", [])


def get_api_credentials(service: str, provider: Optional[str] = None) -> Dict[str, str]:
    """
    Get API credentials for a specific service.
    
    Args:
        service: Service type (e.g., 'market_data', 'news', 'social')
        provider: Provider name within the service (e.g., 'nse', 'twitter')
    
    Returns:
        Dict containing API credentials
    """
    config = load_config()
    
    if service == "market_data":
        if provider is None:
            raise ValueError("Provider must be specified for market_data service")
        
        return {
            "api_key": config["data_sources"]["market_data"].get(provider, {}).get("api_key", ""),
            "api_url": config["data_sources"]["market_data"].get(provider, {}).get("api_url", ""),
        }
    
    elif service == "alternative_data":
        if provider == "news":
            return {"api_key": config["data_sources"]["alternative_data"].get("news", {}).get("api_key", "")}
        
        elif provider == "twitter":
            twitter_config = config["data_sources"]["alternative_data"].get("social", {}).get("twitter", {})
            return {
                "api_key": twitter_config.get("api_key", ""),
                "api_secret": twitter_config.get("api_secret", ""),
            }
        
        elif provider == "reddit":
            reddit_config = config["data_sources"]["alternative_data"].get("social", {}).get("reddit", {})
            return {
                "client_id": reddit_config.get("client_id", ""),
                "client_secret": reddit_config.get("client_secret", ""),
            }
    
    # Default empty credentials if not found
    return {}