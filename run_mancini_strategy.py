#!/usr/bin/env python3
"""Run the Adam Mancini trading strategy with optimal parameters."""

import os
import sys
import pandas as pd
import argparse
from datetime import datetime, timedelta
from standalone_mancini import AdamManciniNiftyStrategy, BacktestTrader

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Run Adam Mancini trading strategy with optimal parameters")
    
    parser.add_argument(
        "--symbol", 
        type=str, 
        default="NIFTY",
        help="Trading symbol (e.g., NIFTY, BANKNIFTY)"
    )
    parser.add_argument(
        "--exchange", 
        type=str, 
        default="NSE",
        help="Exchange (e.g., NSE, BSE, NFO)"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        help="Path to CSV file with historical data (for backtest mode)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["backtest", "paper", "live"],
        default="backtest",
        help="Trading mode (backtest, paper, live)"
    )
    parser.add_argument(
        "--broker",
        type=str,
        choices=["dhan", "icici"],
        help="Broker to use for paper/live trading"
    )
    
    args = parser.parse_args()
    
    # Optimal parameters based on testing
    open_range_minutes = 6  # 6-minute opening range
    
    # Initialize trader
    trader = BacktestTrader(
        symbol=args.symbol,
        exchange=args.exchange,
        open_range_minutes=open_range_minutes
    )
    
    if args.mode == "backtest":
        if not args.data_file:
            print("Error: data-file is required for backtest mode")
            sys.exit(1)
            
        # Load data
        print(f"Loading data from {args.data_file}...")
        data = pd.read_csv(args.data_file, index_col=0, parse_dates=True)
        
        # Handle Yahoo Finance column format: "('close', '^nsei')"
        if all(isinstance(col, str) and col.startswith("('") for col in data.columns):
            col_map = {}
            for col in data.columns:
                try:
                    # Extract the type (open, high, low, etc.)
                    col_type = col.split("'")[1].lower()
                    col_map[col] = col_type
                except:
                    pass
            
            # Rename columns
            data = data.rename(columns=col_map)
        
        # Convert column names to lowercase
        data.columns = [col.lower() for col in data.columns]
        
        # Run backtest
        print(f"\nRunning backtest for {args.symbol} on {args.exchange} with {open_range_minutes}-minute opening range...")
        trader.run_backtest(data)
    
    elif args.mode in ["paper", "live"]:
        if not args.broker:
            print("Error: broker is required for paper/live mode")
            sys.exit(1)
            
        print(f"\nRunning {args.mode} trading for {args.symbol} on {args.exchange} with {open_range_minutes}-minute opening range...")
        print(f"Using {args.broker} broker")
        
        # Connect to the appropriate broker API
        try:
            if args.broker == "icici":
                # Make the broker modules accessible
                import sys
                sys.path.append('/Users/arvindcr/indiatrader')
                
                # Try different import paths
                try:
                    from indiatrader.brokers.icici import ICICIBreezeClient
                except ImportError:
                    try:
                        from brokers.icici import ICICIBreezeClient
                    except ImportError:
                        try:
                            from indiatrader.brokers.icici import ICICIBreezeClient
                        except ImportError:
                            from icici import ICICIBreezeClient
                
                # Initialize the broker client
                # Use ICICI API key stored in the config
                api_key = "U6xG1V56S02aX5r95I933222!74f4#k2"  # This should come from a secure config in production
                
                # Different initialization based on mode
                if args.mode == 'paper':
                    icici_client = ICICIBreezeClient(api_key, paper_trade_mode=True)
                    print("Initialized ICICI Breeze client in paper trade mode")
                else:
                    # Live mode - need actual credentials
                    icici_client = ICICIBreezeClient(api_key)
                    print("Authenticating with ICICI Breeze for live trading...")
                
                # Setup market data connection is handled by place_mancini_trades method
                
                # Start the Adam Mancini strategy with live trading
                print("Starting Adam Mancini strategy with live trading...")
                
                # Configure trading parameters
                product_type = "MIS"  # Intraday
                quantity = 1  # NIFTY lot size (adjust as needed)
                
                # Start real-time monitoring and trading
                if args.mode == "live":
                    print("LIVE MODE: Will place actual trades")
                    icici_client.place_mancini_trades(
                        symbol=args.symbol,
                        exchange=args.exchange,
                        product_type=product_type,
                        quantity=quantity,
                        opening_range_minutes=open_range_minutes,
                        is_paper_trade=False
                    )
                else:  # paper mode
                    print("PAPER MODE: Simulating trades only")
                    icici_client.place_mancini_trades(
                        symbol=args.symbol,
                        exchange=args.exchange,
                        product_type=product_type,
                        quantity=quantity,
                        opening_range_minutes=open_range_minutes,
                        is_paper_trade=True
                    )
            
            elif args.broker == "dhan":
                # Import path is already set up above
                try:
                    from indiatrader.brokers.dhan import DhanClient
                except ImportError:
                    try:
                        from brokers.dhan import DhanClient
                    except ImportError:
                        try:
                            from indiatrader.brokers.dhan import DhanClient
                        except ImportError:
                            from dhan import DhanClient
                
                # Initialize the broker client
                client_id = "YOUR_CLIENT_ID"  # This should come from a secure config
                dhan_client = DhanClient(client_id)
                
                # Authenticate
                print("Authenticating with Dhan...")
                dhan_client.authenticate()
                
                # Configure trading parameters
                product_type = "INTRADAY"
                quantity = 1  # NIFTY lot size (adjust as needed)
                
                # Start real-time monitoring and trading
                if args.mode == "live":
                    print("LIVE MODE: Will place actual trades")
                    dhan_client.place_mancini_trades(
                        symbol=args.symbol,
                        exchange=args.exchange,
                        product_type=product_type,
                        quantity=quantity,
                        opening_range_minutes=open_range_minutes,
                        is_paper_trade=False
                    )
                else:  # paper mode
                    print("PAPER MODE: Simulating trades only")
                    dhan_client.place_mancini_trades(
                        symbol=args.symbol,
                        exchange=args.exchange,
                        product_type=product_type,
                        quantity=quantity,
                        opening_range_minutes=open_range_minutes,
                        is_paper_trade=True
                    )
            
            print(f"Successfully started {args.mode} trading with {args.broker}")
            
        except ImportError:
            print(f"Error: Could not import {args.broker} API client.")
            print("Please ensure the broker API client is properly installed.")
            print("  - For ICICI Breeze: pip install breeze-connect")
            print("  - For Dhan: pip install dhanhq")
        except Exception as e:
            print(f"Error connecting to {args.broker} API: {str(e)}")
            print("Please check your API credentials and internet connection.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())