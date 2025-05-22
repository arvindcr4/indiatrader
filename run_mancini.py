#!/usr/bin/env python3
"""Run script for the Mancini trading strategy."""

import os
import sys
import argparse
import logging
import json
from datetime import datetime

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from indiatrader.strategies.mancini_trader import ManciniTrader, BrokerType, SignalType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"mancini_trade_{datetime.now().strftime('%Y%m%d')}.log")
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Run Adam Mancini trading strategy")
    
    # Required arguments
    parser.add_argument(
        "--symbol", 
        type=str, 
        default="NIFTY",
        help="Trading symbol (e.g., NIFTY, BANKNIFTY)"
    )
    parser.add_argument(
        "--exchange", 
        type=str, 
        default="NFO",
        help="Exchange (e.g., NSE, BSE, NFO)"
    )
    
    # Broker arguments
    parser.add_argument(
        "--broker", 
        type=str, 
        choices=["dhan", "icici"],
        default="dhan",
        help="Broker to use (default: dhan)"
    )
    
    # Trading parameters
    parser.add_argument(
        "--quantity", 
        type=int, 
        default=1,
        help="Trading quantity (default: 1)"
    )
    parser.add_argument(
        "--product-type", 
        type=str, 
        default="intraday",
        choices=["intraday", "delivery", "margin", "bo", "co"],
        help="Product type (default: intraday)"
    )
    parser.add_argument(
        "--is-futures", 
        action="store_true",
        help="Specify if the symbol is a futures contract"
    )
    parser.add_argument(
        "--expiry-date", 
        type=str,
        help="Expiry date for futures/options in DD-MMM-YYYY format (e.g., 28-Jun-2023)"
    )
    parser.add_argument(
        "--open-range-minutes", 
        type=int, 
        default=15,
        help="Number of minutes for the opening range (default: 15)"
    )
    
    # Execution mode
    parser.add_argument(
        "--backtest-mode", 
        action="store_true",
        help="Run in backtest mode (simulated trades)"
    )
    parser.add_argument(
        "--continuous", 
        action="store_true",
        help="Run in continuous mode"
    )
    parser.add_argument(
        "--interval", 
        type=int, 
        default=60,
        help="Interval between iterations in seconds (default: 60)"
    )
    parser.add_argument(
        "--config-file", 
        type=str,
        default="config/credentials.json",
        help="Path to config file (JSON format)"
    )
    
    args = parser.parse_args()
    
    # Load credentials
    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config_file)
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Determine broker configuration
    broker_type = BrokerType(args.broker)
    broker_config = {}
    
    if broker_type == BrokerType.DHAN:
        broker_config = {
            'client_id': config['dhan'].get('client_id', os.environ.get('DHAN_CLIENT_ID')),
            'access_token': config['dhan'].get('access_token', os.environ.get('DHAN_ACCESS_TOKEN'))
        }
        
    elif broker_type == BrokerType.ICICI:
        broker_config = {
            'api_key': config['icici'].get('api_key', os.environ.get('ICICI_API_KEY')),
            'api_secret': config['icici'].get('api_secret', os.environ.get('ICICI_API_SECRET')),
            'session_token': config['icici'].get('session_token', os.environ.get('ICICI_SESSION_TOKEN'))
        }
    
    # Initialize the trader
    try:
        # Use dummy values for API credentials in backtest mode
        if args.backtest_mode:
            if broker_type == BrokerType.DHAN:
                broker_config = {
                    'client_id': 'BACKTEST',
                    'access_token': 'BACKTEST'
                }
            elif broker_type == BrokerType.ICICI:
                broker_config = {
                    'api_key': 'BACKTEST',
                    'api_secret': 'BACKTEST',
                    'session_token': 'BACKTEST'
                }
        
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
            trader.run_continuous(
                interval_seconds=args.interval,
                start_time="09:15",
                end_time="15:30",
                max_trades_per_day=5
            )
        else:
            logger.info("Running single iteration")
            result = trader.run_once()
            logger.info(f"Result: {result}")
        
        # Print trade history if in backtest mode
        if args.backtest_mode:
            history = trader.get_trade_history()
            if history:
                print("\nTrade History:")
                for trade in history:
                    print(f"{trade['timestamp']} | {trade['action']} | {trade['quantity']} | {trade['price']}")
            else:
                print("No trades executed")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())