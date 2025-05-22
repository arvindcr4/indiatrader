#!/usr/bin/env python3
"""Optimize parameters for the Adam Mancini trading strategy."""

import os
import sys
import pandas as pd
import numpy as np
import argparse
from datetime import datetime, timedelta
from standalone_mancini import AdamManciniNiftyStrategy, BacktestTrader

def run_parameter_scan(data_file, parameter_ranges):
    """Run multiple backtests with different parameter combinations.
    
    Parameters
    ----------
    data_file : str
        Path to CSV file with historical data
    parameter_ranges : dict
        Dictionary of parameter ranges to test
        
    Returns
    -------
    pd.DataFrame
        Results of the parameter scan
    """
    # Load data
    print(f"Loading data from {data_file}...")
    data = pd.read_csv(data_file, index_col=0, parse_dates=True)
    
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
    
    # Ensure required columns exist
    required_cols = ['open', 'high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in data.columns]
    
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        print(f"Available columns: {list(data.columns)}")
        sys.exit(1)
    
    print(f"Loaded {len(data)} bars of data with columns: {list(data.columns)}")
    
    # Run parameter scan
    results = []
    
    for opening_range_minutes in parameter_ranges.get('opening_range_minutes', [15]):
        print(f"\nTesting with opening_range_minutes={opening_range_minutes}")
        
        # Initialize strategy and trader
        trader = BacktestTrader(
            symbol="NIFTY",
            exchange="NSE",
            open_range_minutes=opening_range_minutes
        )
        
        # Run backtest
        signals = trader.run_backtest(data)
        
        # Calculate metrics
        trades = trader.trades
        if trades:
            buy_count = len([t for t in trades if t['action'] == 'BUY'])
            sell_count = len([t for t in trades if t['action'] == 'SELL'])
            total_buy = sum(t['price'] for t in trades if t['action'] == 'BUY')
            total_sell = sum(t['price'] for t in trades if t['action'] == 'SELL')
            pnl = total_sell - total_buy
            
            # Count signals
            long_signals = (signals['long_signal'] == 1).sum()
            short_signals = (signals['short_signal'] == -1).sum()
            
            # Calculate win rate (simplified)
            win_count = 0
            for i in range(1, len(trades)):
                if trades[i]['action'] != trades[i-1]['action']:  # Exit trade
                    if (trades[i-1]['action'] == 'BUY' and trades[i]['price'] > trades[i-1]['price']) or \
                       (trades[i-1]['action'] == 'SELL' and trades[i]['price'] < trades[i-1]['price']):
                        win_count += 1
            
            win_rate = win_count / (len(trades) - 1) if len(trades) > 1 else 0
            
            results.append({
                'opening_range_minutes': opening_range_minutes,
                'total_trades': len(trades),
                'buy_count': buy_count,
                'sell_count': sell_count,
                'pnl': pnl,
                'win_rate': win_rate,
                'long_signals': long_signals,
                'short_signals': short_signals,
            })
        else:
            results.append({
                'opening_range_minutes': opening_range_minutes,
                'total_trades': 0,
                'buy_count': 0,
                'sell_count': 0,
                'pnl': 0,
                'win_rate': 0,
                'long_signals': 0,
                'short_signals': 0,
            })
    
    return pd.DataFrame(results)

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Optimize parameters for Adam Mancini trading strategy")
    
    parser.add_argument(
        "--data-file",
        type=str,
        required=True,
        help="Path to CSV file with historical data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="optimization_results.csv",
        help="Output file for optimization results"
    )
    
    args = parser.parse_args()
    
    # Define parameter ranges to test
    parameter_ranges = {
        'opening_range_minutes': [3, 6, 9, 12, 15, 18, 21, 24]
    }
    
    # Run parameter scan
    results = run_parameter_scan(args.data_file, parameter_ranges)
    
    # Save results
    results.to_csv(args.output)
    print(f"\nParameter scan complete. Results saved to {args.output}")
    
    # Print summary
    print("\nOptimization Results Summary:")
    print(results.sort_values('pnl', ascending=False))
    
    return 0

if __name__ == "__main__":
    sys.exit(main())