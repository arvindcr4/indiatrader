#!/usr/bin/env python3
"""Command-line interface for running the Adam Mancini trading algorithm.

This script provides a CLI for running the ManciniTrader with various options
for broker selection, trading parameters, and execution modes.
"""

import os
import argparse
import logging
import json
from datetime import datetime
import pandas as pd

from indiatrader.strategies.mancini_trader import ManciniTrader, BrokerType, SignalType

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"mancini_trade_{datetime.now().strftime('%Y%m%d')}.log")
    ]
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments.
    
    Returns
    -------
    argparse.Namespace
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Run Adam Mancini trading algorithm")
    
    # Required arguments
    parser.add_argument(
        "--symbol", 
        type=str, 
        required=True,
        help="Trading symbol (e.g., NIFTY, BANKNIFTY)"
    )
    parser.add_argument(
        "--exchange", 
        type=str, 
        required=True,
        help="Exchange (e.g., NSE, BSE, NFO)"
    )
    
    # Broker arguments
    broker_group = parser.add_argument_group("Broker options")
    broker_group.add_argument(
        "--broker", 
        type=str, 
        choices=["dhan", "icici"],
        default="dhan",
        help="Broker to use (default: dhan)"
    )
    
    # Dhan broker arguments
    dhan_group = parser.add_argument_group("Dhan broker options")
    dhan_group.add_argument(
        "--dhan-client-id", 
        type=str,
        help="Dhan client ID (if using Dhan)"
    )
    dhan_group.add_argument(
        "--dhan-access-token", 
        type=str,
        help="Dhan access token (if using Dhan)"
    )
    
    # ICICI broker arguments
    icici_group = parser.add_argument_group("ICICI broker options")
    icici_group.add_argument(
        "--icici-api-key", 
        type=str,
        help="ICICI Breeze API key (if using ICICI)"
    )
    icici_group.add_argument(
        "--icici-api-secret", 
        type=str,
        help="ICICI Breeze API secret (if using ICICI)"
    )
    icici_group.add_argument(
        "--icici-session-token", 
        type=str,
        help="ICICI Breeze session token (if using ICICI)"
    )
    
    # Trading parameters
    trading_group = parser.add_argument_group("Trading parameters")
    trading_group.add_argument(
        "--quantity", 
        type=int, 
        default=1,
        help="Trading quantity (default: 1)"
    )
    trading_group.add_argument(
        "--product-type", 
        type=str, 
        default="intraday",
        choices=["intraday", "delivery", "margin", "bo", "co"],
        help="Product type (default: intraday)"
    )
    trading_group.add_argument(
        "--is-futures", 
        action="store_true",
        help="Specify if the symbol is a futures contract"
    )
    trading_group.add_argument(
        "--expiry-date", 
        type=str,
        help="Expiry date for futures/options in DD-MMM-YYYY format (e.g., 28-Jun-2023)"
    )
    trading_group.add_argument(
        "--open-range-minutes", 
        type=int, 
        default=15,
        help="Number of minutes for the opening range (default: 15)"
    )
    
    # Execution mode
    execution_group = parser.add_argument_group("Execution mode")
    execution_group.add_argument(
        "--backtest-mode", 
        action="store_true",
        help="Run in backtest mode (simulated trades)"
    )
    execution_group.add_argument(
        "--continuous", 
        action="store_true",
        help="Run in continuous mode"
    )
    execution_group.add_argument(
        "--interval", 
        type=int, 
        default=60,
        help="Interval between iterations in seconds (default: 60)"
    )
    execution_group.add_argument(
        "--start-time", 
        type=str, 
        default="09:15",
        help="Trading start time in HH:MM format (default: 09:15)"
    )
    execution_group.add_argument(
        "--end-time", 
        type=str, 
        default="15:30",
        help="Trading end time in HH:MM format (default: 15:30)"
    )
    execution_group.add_argument(
        "--max-trades-per-day", 
        type=int, 
        default=5,
        help="Maximum number of trades per day (default: 5)"
    )
    
    # Other options
    parser.add_argument(
        "--config-file", 
        type=str,
        help="Path to config file (JSON format)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def load_config(config_file):
    """Load configuration from a JSON file.
    
    Parameters
    ----------
    config_file : str
        Path to the configuration file.
        
    Returns
    -------
    dict
        Configuration dictionary.
    """
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config file: {e}")
        return {}


def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load config file if provided
    config = {}
    if args.config_file:
        config = load_config(args.config_file)
    
    # Determine broker configuration
    broker_type = BrokerType(args.broker)
    broker_config = {}
    
    if broker_type == BrokerType.DHAN:
        broker_config = {
            'client_id': args.dhan_client_id or config.get('dhan_client_id') or os.environ.get('DHAN_CLIENT_ID'),
            'access_token': args.dhan_access_token or config.get('dhan_access_token') or os.environ.get('DHAN_ACCESS_TOKEN')
        }
        
        if not broker_config['client_id'] or not broker_config['access_token']:
            logger.error("Dhan broker requires client_id and access_token")
            return 1
    
    elif broker_type == BrokerType.ICICI:
        broker_config = {
            'api_key': args.icici_api_key or config.get('icici_api_key') or os.environ.get('ICICI_API_KEY'),
            'api_secret': args.icici_api_secret or config.get('icici_api_secret') or os.environ.get('ICICI_API_SECRET'),
            'session_token': args.icici_session_token or config.get('icici_session_token') or os.environ.get('ICICI_SESSION_TOKEN')
        }
        
        if not broker_config['api_key'] or not broker_config['api_secret'] or not broker_config['session_token']:
            logger.error("ICICI broker requires api_key, api_secret, and session_token")
            return 1
    
    # Initialize the trader
    try:
        trader = ManciniTrader(
            broker_type=broker_type,
            broker_config=broker_config,
            symbol=args.symbol,
            exchange=args.exchange,
            product_type=args.product_type,
            quantity=args.quantity,
            open_range_minutes=args.open_range_minutes,
            is_futures=args.is_futures,
            expiry_date=args.expiry_date,
            backtest_mode=args.backtest_mode
        )
        
        logger.info(f"Initialized ManciniTrader for {args.symbol} on {args.exchange}")
        
        # Run the trader
        if args.continuous:
            logger.info(f"Running in continuous mode with interval {args.interval}s")
            try:
                trader.run_continuous(
                    interval_seconds=args.interval,
                    start_time=args.start_time,
                    end_time=args.end_time,
                    max_trades_per_day=args.max_trades_per_day
                )
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, stopping")
            finally:
                trader.stop()
        else:
            logger.info("Running single iteration")
            result = trader.run_once()
            logger.info(f"Result: {result}")
        
        # Print trade history if in backtest mode
        if args.backtest_mode:
            history = trader.get_trade_history()
            if not history.empty:
                logger.info("\nTrade History:")
                print(history.to_string())
            else:
                logger.info("No trades executed")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error initializing or running trader: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())