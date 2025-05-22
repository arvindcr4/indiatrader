#!/usr/bin/env python3
"""Calculate daily profit/loss for the Adam Mancini strategy."""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def calculate_pivot_levels(daily_data):
    """Calculate pivot levels based on previous day's data."""
    pivot = (daily_data['high'] + daily_data['low'] + daily_data['close']) / 3
    r1 = 2 * pivot - daily_data['low']
    s1 = 2 * pivot - daily_data['high']
    r2 = pivot + (daily_data['high'] - daily_data['low'])
    s2 = pivot - (daily_data['high'] - daily_data['low'])
    r3 = r1 + (daily_data['high'] - daily_data['low'])
    s3 = s1 - (daily_data['high'] - daily_data['low'])
    
    return {
        'pivot': pivot,
        'r1': r1,
        'r2': r2,
        'r3': r3,
        's1': s1,
        's2': s2,
        's3': s3
    }

def backtest_daily_pnl(data_file, opening_range_minutes=6):
    """Run a backtest and calculate daily P&L."""
    print(f"Loading data from {data_file}...")
    
    # Load the data
    data = pd.read_csv(data_file, index_col=0, parse_dates=True)
    
    # Handle column names from Yahoo Finance
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
    
    print(f"Loaded {len(data)} bars of data with columns: {list(data.columns)}")
    
    # Process data by day
    daily_results = {}
    trade_list = []
    position = 0  # 0: no position, 1: long, -1: short
    
    # Group data by day
    data['date'] = data.index.date
    days = data['date'].unique()
    
    for i, day in enumerate(days):
        day_str = day.strftime("%Y-%m-%d")
        daily_results[day_str] = {'trades': [], 'pnl': 0}
        
        # Get data for this day
        day_data = data[data['date'] == day].copy()
        
        if len(day_data) == 0:
            continue
            
        # Calculate pivot levels using previous day's data (if available)
        if i > 0:
            prev_day = days[i-1]
            prev_day_data = data[data['date'] == prev_day]
            if len(prev_day_data) > 0:
                pivot_levels = calculate_pivot_levels({
                    'high': prev_day_data['high'].max(),
                    'low': prev_day_data['low'].min(),
                    'close': prev_day_data['close'].iloc[-1]
                })
                day_data['pivot'] = pivot_levels['pivot']
            else:
                day_data['pivot'] = day_data['open'].iloc[0]  # Use today's open as fallback
        else:
            # First day, use the day's open as pivot
            day_data['pivot'] = day_data['open'].iloc[0]
        
        # Calculate opening range
        if len(day_data) >= opening_range_minutes:
            opening_range_data = day_data.iloc[:opening_range_minutes]
            high_or = opening_range_data['high'].max()
            low_or = opening_range_data['low'].min()
            
            # Add opening range to dataframe
            day_data['high_or'] = high_or
            day_data['low_or'] = low_or
            
            # Skip opening range period for trading
            trading_data = day_data.iloc[opening_range_minutes:]
            
            # Generate signals
            for idx, row in trading_data.iterrows():
                # Generate signals based on price relative to opening range and pivot
                long_signal = (row['close'] > high_or) and (row['close'] > row['pivot'])
                short_signal = (row['close'] < low_or) and (row['close'] < row['pivot'])
                
                # Trading logic
                if long_signal and position <= 0:
                    # Long signal - buy
                    trade = {
                        'timestamp': idx,
                        'day': day_str,
                        'action': 'BUY',
                        'price': row['close'],
                        'pl': 0
                    }
                    
                    # Calculate P&L for the previous position if it was a short
                    if position == -1 and len(trade_list) > 0:
                        prev_trade = trade_list[-1]
                        trade['pl'] = prev_trade['price'] - row['close']
                        daily_results[day_str]['pnl'] += trade['pl']
                    
                    trade_list.append(trade)
                    daily_results[day_str]['trades'].append(trade)
                    position = 1
                    
                elif short_signal and position >= 0:
                    # Short signal - sell
                    trade = {
                        'timestamp': idx,
                        'day': day_str,
                        'action': 'SELL',
                        'price': row['close'],
                        'pl': 0
                    }
                    
                    # Calculate P&L for the previous position if it was a long
                    if position == 1 and len(trade_list) > 0:
                        prev_trade = trade_list[-1]
                        trade['pl'] = row['close'] - prev_trade['price']
                        daily_results[day_str]['pnl'] += trade['pl']
                    
                    trade_list.append(trade)
                    daily_results[day_str]['trades'].append(trade)
                    position = -1
    
    # Close any open position at the end using the last price
    if position != 0 and len(trade_list) > 0:
        last_day = days[-1].strftime("%Y-%m-%d")
        last_price = data.iloc[-1]['close']
        last_trade = trade_list[-1]
        
        if position == 1:
            # Close long position
            pl = last_price - last_trade['price']
        else:
            # Close short position
            pl = last_trade['price'] - last_price
            
        daily_results[last_day]['pnl'] += pl
    
    # Print daily P&L summary
    print("\nDaily P&L Summary:")
    print("=" * 40)
    print(f"{'Date':<12} {'# Trades':<10} {'P&L':<10}")
    print("-" * 40)
    
    total_pnl = 0
    total_trades = 0
    profitable_days = 0
    losing_days = 0
    
    for day, results in daily_results.items():
        pnl = results['pnl']
        trades = len(results['trades'])
        total_pnl += pnl
        total_trades += trades
        
        if pnl > 0:
            profitable_days += 1
        elif pnl < 0:
            losing_days += 1
            
        print(f"{day:<12} {trades:<10} {pnl:>10.2f}")
    
    print("-" * 40)
    print(f"Total      {total_trades:<10} {total_pnl:>10.2f}")
    
    # Print summary statistics
    days_with_trades = sum(1 for day, results in daily_results.items() if len(results['trades']) > 0)
    print("\nSummary Statistics:")
    print(f"Total days analyzed: {len(daily_results)}")
    print(f"Days with trades: {days_with_trades}")
    print(f"Profitable days: {profitable_days}")
    print(f"Losing days: {losing_days}")
    print(f"Win rate: {profitable_days / max(days_with_trades, 1) * 100:.2f}%")
    print(f"Total P&L: {total_pnl:.2f}")
    print(f"Average daily P&L: {total_pnl / max(days_with_trades, 1):.2f}")
    
    # Plot daily P&L
    fig, ax = plt.subplots(figsize=(10, 6))
    
    days = []
    pnls = []
    
    for day, results in daily_results.items():
        if len(results['trades']) > 0:
            days.append(day)
            pnls.append(results['pnl'])
    
    ax.bar(days, pnls, color=['green' if pnl > 0 else 'red' for pnl in pnls])
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    ax.set_title('Adam Mancini Strategy - Daily P&L')
    ax.set_xlabel('Date')
    ax.set_ylabel('P&L (points)')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('daily_pnl.png')
    print("P&L chart saved to daily_pnl.png")
    
    # Calculate cumulative P&L
    cum_pnl = []
    running_total = 0
    
    for pnl in pnls:
        running_total += pnl
        cum_pnl.append(running_total)
    
    # Plot cumulative P&L
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(days, cum_pnl, marker='o', linewidth=2)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    ax.set_title('Adam Mancini Strategy - Cumulative P&L')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative P&L (points)')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('cumulative_pnl.png')
    print("Cumulative P&L chart saved to cumulative_pnl.png")
    
    return daily_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate daily P&L for Adam Mancini strategy")
    
    parser.add_argument(
        "--data-file", 
        type=str, 
        default="data/nifty_5min_30days.csv",
        help="Path to CSV file with historical data"
    )
    parser.add_argument(
        "--open-range-minutes", 
        type=int, 
        default=6,
        help="Number of minutes for the opening range (default: 6)"
    )
    
    args = parser.parse_args()
    
    backtest_daily_pnl(args.data_file, args.open_range_minutes)