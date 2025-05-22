#!/usr/bin/env python3
"""Standalone script for the Adam Mancini trading strategy."""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict
from datetime import datetime, timedelta
import sys

@dataclass
class Levels:
    """Container for daily pivot levels."""
    pivot: float
    r1: float
    r2: float
    r3: float
    s1: float
    s2: float
    s3: float

class AdamManciniNiftyStrategy:
    """Generate trading signals for Nifty based on Adam Mancini style levels."""

    def __init__(self, open_range_minutes: int = 15):
        self.open_range_minutes = open_range_minutes

    def compute_levels(self, daily_ohlc: pd.DataFrame) -> pd.DataFrame:
        """Compute pivot, support and resistance levels."""
        pivot = (daily_ohlc["high"] + daily_ohlc["low"] + daily_ohlc["close"]) / 3.0
        r1 = 2 * pivot - daily_ohlc["low"]
        s1 = 2 * pivot - daily_ohlc["high"]
        r2 = pivot + (daily_ohlc["high"] - daily_ohlc["low"])
        s2 = pivot - (daily_ohlc["high"] - daily_ohlc["low"])
        r3 = r1 + (daily_ohlc["high"] - daily_ohlc["low"])
        s3 = s1 - (daily_ohlc["high"] - daily_ohlc["low"])

        levels = pd.DataFrame(
            {
                "pivot": pivot,
                "r1": r1,
                "r2": r2,
                "r3": r3,
                "s1": s1,
                "s2": s2,
                "s3": s3,
            }
        )
        return levels

    def _open_range(self, data):
        """Calculate opening range high and low."""
        # Return high and low for the entire day
        if 'high' in data.columns and 'low' in data.columns:
            return pd.Series({
                'high_or': data['high'].max(),
                'low_or': data['low'].min()
            })
        else:
            # Fallback
            print("Warning: Missing high/low columns in data")
            return pd.Series({
                'high_or': data.iloc[0] if not data.empty else 0, 
                'low_or': data.iloc[0] if not data.empty else 0
            })

    def generate_signals(self, intraday: pd.DataFrame) -> pd.DataFrame:
        """Generate intraday trading signals."""
        if not {"high", "low", "close"}.issubset(intraday.columns):
            raise ValueError("DataFrame must contain high, low and close columns")

        print(f"Generating signals for {len(intraday)} bars of data")
        
        # Resample to daily data
        daily = intraday.resample("1D").agg({"high": "max", "low": "min", "close": "last"})
        print(f"Resampled to {len(daily)} daily bars")
        
        # Compute pivot levels
        levels = self.compute_levels(daily)
        print(f"Computed pivot levels: {levels.iloc[0].to_dict()}")
        
        # For each day, get the opening range (first 15 minutes)
        open_ranges = []
        for day in daily.index:
            day_start = day.strftime('%Y-%m-%d')
            day_end = (day + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            
            day_data = intraday.loc[day_start:day_end]
            
            # Get first N minutes
            open_range_data = day_data.iloc[:self.open_range_minutes]
            
            if not open_range_data.empty:
                open_ranges.append({
                    'date': day,
                    'high_or': open_range_data['high'].max(),
                    'low_or': open_range_data['low'].min()
                })
            else:
                open_ranges.append({
                    'date': day,
                    'high_or': np.nan,
                    'low_or': np.nan
                })
        
        # Create dataframe with opening ranges
        open_range_df = pd.DataFrame(open_ranges).set_index('date')
        print(f"Calculated opening ranges:\n{open_range_df}")
        
        # Join data
        data = intraday.copy()
        
        # Add pivot levels to each intraday bar
        for col in levels.columns:
            data[col] = np.nan
            for day in levels.index:
                day_mask = (data.index.date == day.date())
                data.loc[day_mask, col] = levels.loc[day, col]
        
        # Add opening range to each intraday bar
        for col in open_range_df.columns:
            data[col] = np.nan
            for day in open_range_df.index:
                day_mask = (data.index.date == day.date())
                data.loc[day_mask, col] = open_range_df.loc[day, col]
        
        # Fill forward
        data = data.ffill()
        
        # Generate signals
        data["long_signal"] = np.where(
            (data["close"] > data["high_or"]) & (data["close"] > data["pivot"]), 1, 0
        )
        data["short_signal"] = np.where(
            (data["close"] < data["low_or"]) & (data["close"] < data["pivot"]), -1, 0
        )
        
        # Count signals
        long_count = data["long_signal"].sum()
        short_count = (data["short_signal"] == -1).sum()
        print(f"Generated {long_count} long signals and {short_count} short signals")
        
        return data

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
        
        print(f"Sample data shape: {data.shape}")
        print(f"First 5 rows of data:")
        print(data.head())
        
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
            print(f"\nBacktest Summary:")
            print(f"Total trades: {len(self.trades)}")
            print(f"Buy trades: {buy_count} | Sell trades: {sell_count}")
            print(f"P&L: {pnl:.2f}")
        else:
            print("\nNo trades executed during backtest")
        
        return signals
    
    def _generate_sample_data(self):
        """Generate sample data for backtest."""
        # Generate sample date range
        start_date = pd.Timestamp('2023-01-01 09:15:00')
        dates = []
        
        # Create a more realistic trading calendar
        for i in range(5):  # 5 trading days
            day_start = start_date + pd.Timedelta(days=i)
            # Generate intraday times from 9:15 to 15:30
            for j in range(375):  # 375 minutes in a trading day (6h15m)
                dates.append(day_start + pd.Timedelta(minutes=j))
        
        # Generate price data (more realistic)
        base_price = 18000
        prices = []
        
        # Day 1: Slightly up
        day1_close = base_price + np.random.normal(50, 20)
        day1_prices = np.linspace(base_price, day1_close, 375) + np.random.normal(0, 15, 375)
        prices.extend(day1_prices)
        
        # Day 2: Down
        day2_close = day1_close - np.random.normal(100, 30)
        day2_prices = np.linspace(day1_close, day2_close, 375) + np.random.normal(0, 20, 375)
        prices.extend(day2_prices)
        
        # Day 3: Big up
        day3_close = day2_close + np.random.normal(200, 50)
        day3_prices = np.linspace(day2_close, day3_close, 375) + np.random.normal(0, 25, 375)
        prices.extend(day3_prices)
        
        # Day 4: Sideways
        day4_close = day3_close + np.random.normal(0, 20)
        day4_prices = np.linspace(day3_close, day4_close, 375) + np.random.normal(0, 30, 375)
        prices.extend(day4_prices)
        
        # Day 5: Down
        day5_close = day4_close - np.random.normal(150, 40)
        day5_prices = np.linspace(day4_close, day5_close, 375) + np.random.normal(0, 35, 375)
        prices.extend(day5_prices)
        
        # Create OHLC data
        closes = prices
        opens = [prices[i-1] if i > 0 else prices[0] for i in range(len(prices))]
        
        # Add some randomness to high/low
        highs = [max(o, c) + np.random.uniform(5, 25) for o, c in zip(opens, closes)]
        lows = [min(o, c) - np.random.uniform(5, 25) for o, c in zip(opens, closes)]
        
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
    # Get command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run Adam Mancini trading strategy (standalone backtest)")
    
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
    parser.add_argument(
        "--data-file",
        type=str,
        help="Path to CSV file with historical data"
    )
    
    args = parser.parse_args()
    
    # Load data if provided
    data = None
    if args.data_file:
        print(f"Loading data from {args.data_file}...")
        data = pd.read_csv(args.data_file, index_col=0, parse_dates=True)
        
        # Print raw column names for debugging
        print(f"Original columns: {list(data.columns)}")
        
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
            # Try to map common column names
            common_map = {
                'adj close': 'close',
                'volume': 'volume'
            }
            
            for old_col, new_col in common_map.items():
                if old_col in data.columns and new_col not in data.columns:
                    data[new_col] = data[old_col]
            
            # Check again for missing columns
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                print(f"Error: Missing required columns: {missing_cols}")
                print(f"Available columns: {list(data.columns)}")
                sys.exit(1)
        
        print(f"Loaded {len(data)} bars of data with columns: {list(data.columns)}")
    
    # Run backtest
    print(f"Running backtest for {args.symbol} on {args.exchange}...")
    trader = BacktestTrader(
        symbol=args.symbol,
        exchange=args.exchange,
        open_range_minutes=args.open_range_minutes
    )
    
    trader.run_backtest(data)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())