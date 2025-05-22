#!/usr/bin/env python3
"""Download historical data from NSE for testing Adam Mancini strategies."""

import os
import sys
import pandas as pd
import numpy as np
import argparse
import yfinance as yf
from datetime import datetime, timedelta

def download_nse_data(symbol, start_date=None, end_date=None, interval='1d'):
    """Download historical data from Yahoo Finance for NSE symbols.
    
    Parameters
    ----------
    symbol : str
        Symbol to download (e.g., 'NIFTY 50', 'BANKNIFTY')
    start_date : str, optional
        Start date in YYYY-MM-DD format
    end_date : str, optional
        End date in YYYY-MM-DD format
    interval : str, optional
        Data interval ('1m', '5m', '15m', '30m', '60m', '1d')
        
    Returns
    -------
    pd.DataFrame
        Historical data
    """
    # Format symbol for Yahoo Finance
    if symbol == 'NIFTY' or symbol == 'NIFTY 50':
        yf_symbol = '^NSEI'
    elif symbol == 'BANKNIFTY' or symbol == 'NIFTY BANK':
        yf_symbol = '^NSEBANK'
    elif symbol.startswith('^'):  # Already a Yahoo Finance index symbol
        yf_symbol = symbol
    else:
        yf_symbol = f"{symbol}.NS"
    
    # Set default dates if not provided
    if start_date is None:
        start_date = datetime.now() - timedelta(days=30)
    else:
        start_date = pd.to_datetime(start_date)
    
    if end_date is None:
        end_date = datetime.now()
    else:
        end_date = pd.to_datetime(end_date)
    
    print(f"Downloading {symbol} data from {start_date} to {end_date} with {interval} interval")
    
    # Download data
    data = yf.download(
        yf_symbol,
        start=start_date,
        end=end_date,
        interval=interval,
        progress=False
    )
    
    # Clean up data
    data.index.name = 'datetime'
    
    # Filter to market hours (9:15 AM to 3:30 PM)
    if interval.endswith('m'):
        data = data.between_time('09:15', '15:30')
    
    # Lowercase column names
    data.columns = [str(col).lower() for col in data.columns]
    
    return data

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Download NSE historical data")
    
    parser.add_argument(
        "--symbol", 
        type=str, 
        default="NIFTY",
        help="Symbol to download (e.g., NIFTY, BANKNIFTY)"
    )
    parser.add_argument(
        "--start-date", 
        type=str, 
        help="Start date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--end-date", 
        type=str, 
        help="End date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--interval", 
        type=str, 
        default="1d",
        choices=['1m', '5m', '15m', '30m', '60m', '1d'],
        help="Data interval (1m, 5m, 15m, 30m, 60m, 1d)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        help="Output file path"
    )
    
    args = parser.parse_args()
    
    # Download data
    data = download_nse_data(
        args.symbol,
        args.start_date,
        args.end_date,
        args.interval
    )
    
    # Print info about downloaded data
    print(f"Downloaded {len(data)} bars of {args.symbol} data")
    
    if len(data) > 0:
        print(f"Date range: {data.index[0]} to {data.index[-1]}")
        
        # Save to file if output path is provided
        if args.output:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            
            # Save to CSV
            data.to_csv(args.output)
            print(f"Saved data to {args.output}")
        else:
            # Print first few rows
            print(f"\nFirst 5 rows:")
            print(data.head())
    else:
        print("No data available for the specified symbol and time range.")
        if args.interval in ['1m', '2m', '5m', '15m', '30m', '60m', '90m']:
            print("Note: Yahoo Finance has limitations on intraday historical data (typically 7-60 days).")
            print("Try a shorter time range or a daily interval for longer periods.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())