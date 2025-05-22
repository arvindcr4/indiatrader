#!/usr/bin/env python3
"""Run script for the Mancini trading strategy (simplified version)."""

import os
import sys
import argparse
import logging
import json
from datetime import datetime

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import only what we need
from indiatrader.strategies.adam_mancini import AdamManciniNiftyStrategy
from indiatrader.strategies.mancini_trader import BrokerType

class BacktestTrader:
    """Simple backtester for the Adam Mancini strategy."""
    
    def __init__(self, symbol, exchange, open_range_minutes=15):
        """Initialize the backtester."""
        self.symbol = symbol
        self.exchange = exchange
        self.strategy = AdamManciniNiftyStrategy(open_range_minutes=open_range_minutes)
        self.position = 0
        self.trades = []
    
    def run_backtest(self, data=None):
        """Run a backtest using sample data."""
        if data is None:
            # Generate sample data if none provided
            data = self._generate_sample_data()
        
        # Generate signals
        signals = self.strategy.generate_signals(data)
        
        # Simulate trades
        for idx, row in signals.iterrows():
            if row['long_signal'] == 1 and self.position <= 0:
                # Long signal - buy
                self.position += 1
                self.trades.append({
                    'timestamp': idx,
                    'action': 'BUY',
                    'price': row['close'],
                    'quantity': 1,
                    'position': self.position
                })
                print(f"{idx} | BUY | Price: {row['close']:.2f} | Position: {self.position}")
            
            elif row['short_signal'] == -1 and self.position >= 0:
                # Short signal - sell
                self.position -= 1
                self.trades.append({
                    'timestamp': idx,
                    'action': 'SELL',
                    'price': row['close'],
                    'quantity': 1,
                    'position': self.position
                })
                print(f"{idx} | SELL | Price: {row['close']:.2f} | Position: {self.position}")
        
        # Calculate P&L
        if self.trades:
            total_buy = sum(t['price'] for t in self.trades if t['action'] == 'BUY')
            total_sell = sum(t['price'] for t in self.trades if t['action'] == 'SELL')
            buy_count = len([t for t in self.trades if t['action'] == 'BUY'])
            sell_count = len([t for t in self.trades if t['action'] == 'SELL'])
            
            pnl = total_sell - total_buy
            print(f"\nTotal trades: {len(self.trades)}")
            print(f"Buy trades: {buy_count} | Sell trades: {sell_count}")
            print(f"P&L: {pnl:.2f}")
        
        return signals
    
    def _generate_sample_data(self):
        """Generate sample data for backtest."""
        import pandas as pd
        import numpy as np
        
        # Generate sample date range
        dates = pd.date_range(start='2023-01-01 09:15:00', periods=100, freq='1min')
        
        # Generate price data
        base_price = 18000
        volatility = 50
        trend = 0.5  # Small upward trend
        
        closes = [base_price]
        for i in range(1, len(dates)):
            # Random walk with drift
            closes.append(closes[-1] + np.random.normal(trend, volatility))
        
        # Generate sample OHLC data
        highs = [c + np.random.uniform(10, 30) for c in closes]
        lows = [c - np.random.uniform(10, 30) for c in closes]
        opens = [l + np.random.uniform(0, 1) * (h - l) for h, l in zip(highs, lows)]
        
        # Create DataFrame
        df = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes
        }, index=dates)
        
        return df

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Run Adam Mancini trading strategy (simple backtest)")
    
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
    parser.add_argument(
        "--open-range-minutes", 
        type=int, 
        default=15,
        help="Number of minutes for the opening range (default: 15)"
    )
    
    args = parser.parse_args()
    
    # Run backtest
    print(f"Running backtest for {args.symbol} on {args.exchange}...")
    trader = BacktestTrader(
        symbol=args.symbol,
        exchange=args.exchange,
        open_range_minutes=args.open_range_minutes
    )
    
    trader.run_backtest()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())