#!/usr/bin/env python3
"""
Backtest for Adam Mancini strategy on Nifty data over the last 30 days.
Generates daily P&L report to show how the strategy performed each day.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from indiatrader.strategies.adam_mancini import AdamManciniNiftyStrategy

# Constants
DATA_FILE = 'data/nifty_5min_30days.csv'
OPEN_RANGE_MINUTES = 6  # Optimized parameter
POSITION_SIZE = 100  # Number of Nifty units per trade

def load_data(file_path):
    """Load and preprocess the OHLC data."""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Convert datetime to proper format
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    
    # Rename columns to match the strategy requirements
    df.rename(columns={
        "('close', '^nsei')": "close",
        "('high', '^nsei')": "high",
        "('low', '^nsei')": "low",
        "('open', '^nsei')": "open",
        "('volume', '^nsei')": "volume"
    }, inplace=True)
    
    return df

def backtest_strategy(data, open_range_minutes=15):
    """Run the Adam Mancini strategy on the data."""
    print(f"Running backtest with {open_range_minutes} minute opening range...")
    
    # Initialize the strategy
    strategy = AdamManciniNiftyStrategy(open_range_minutes=open_range_minutes)
    
    # Generate signals
    signals_df = strategy.generate_signals(data)
    
    return signals_df

def calculate_daily_pnl(signals_df, position_size=100):
    """Calculate daily profit and loss from the signals."""
    print("Calculating daily P&L...")
    
    # Create position column (1 for long, -1 for short, 0 for no position)
    signals_df['position'] = signals_df['long_signal'] - signals_df['short_signal']
    
    # Calculate percentage returns
    signals_df['pct_change'] = signals_df['close'].pct_change()
    
    # Calculate PnL for each bar based on the previous bar's position
    signals_df['pnl'] = signals_df['position'].shift(1) * signals_df['pct_change'] * position_size
    
    # Group by date to get daily PnL
    daily_pnl = signals_df.groupby(signals_df.index.date)['pnl'].sum()
    
    # Convert to DataFrame and add formatting
    daily_pnl_df = pd.DataFrame({
        'Date': daily_pnl.index,
        'P&L (points)': daily_pnl.values
    })
    
    # Calculate running totals
    daily_pnl_df['Cumulative P&L'] = daily_pnl_df['P&L (points)'].cumsum()
    
    # Calculate statistics
    total_days = len(daily_pnl_df)
    profitable_days = sum(daily_pnl_df['P&L (points)'] > 0)
    losing_days = sum(daily_pnl_df['P&L (points)'] < 0)
    
    # Calculate winrate
    win_rate = profitable_days / total_days * 100 if total_days > 0 else 0
    
    stats = {
        'total_days': total_days,
        'profitable_days': profitable_days,
        'losing_days': losing_days,
        'win_rate': win_rate,
        'total_pnl': daily_pnl_df['P&L (points)'].sum()
    }
    
    return daily_pnl_df, stats

def generate_report(daily_pnl_df, stats):
    """Print a report of the backtest results."""
    print("\n===== ADAM MANCINI STRATEGY BACKTEST REPORT =====")
    print(f"Period: {daily_pnl_df['Date'].min()} to {daily_pnl_df['Date'].max()}")
    print(f"Opening Range: {OPEN_RANGE_MINUTES} minutes")
    print("---------------------------------------------------")
    print(f"Total Trading Days: {stats['total_days']}")
    print(f"Profitable Days: {stats['profitable_days']} ({stats['win_rate']:.1f}%)")
    print(f"Losing Days: {stats['losing_days']} ({100-stats['win_rate']:.1f}%)")
    print(f"Total P&L: {stats['total_pnl']:.2f} points")
    print("---------------------------------------------------")
    print("\nDAILY P&L BREAKDOWN:")
    
    # Format the table for display
    display_df = daily_pnl_df.copy()
    display_df['Date'] = display_df['Date'].astype(str)
    display_df['P&L (points)'] = display_df['P&L (points)'].round(2)
    display_df['Cumulative P&L'] = display_df['Cumulative P&L'].round(2)
    
    print(display_df.to_string(index=False))

def main():
    """Main function to run the backtest."""
    # Load data
    data = load_data(DATA_FILE)
    
    # Run strategy
    signals_df = backtest_strategy(data, open_range_minutes=OPEN_RANGE_MINUTES)
    
    # Calculate daily P&L
    daily_pnl_df, stats = calculate_daily_pnl(signals_df, position_size=POSITION_SIZE)
    
    # Generate report
    generate_report(daily_pnl_df, stats)
    
    # Save results to CSV
    daily_pnl_df.to_csv('mancini_daily_pnl_report.csv', index=False)
    print("\nDetailed results saved to 'mancini_daily_pnl_report.csv'")

if __name__ == "__main__":
    main()