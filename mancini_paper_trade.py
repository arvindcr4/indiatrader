#!/usr/bin/env python3
"""Simplified Adam Mancini strategy paper trading simulation."""

import sys
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add indiatrader to the import path
sys.path.append('/Users/arvindcr/indiatrader')

try:
    from indiatrader.strategies.mancini_trader import AdamManciniNiftyStrategy
except ImportError:
    print("Failed to import AdamManciniNiftyStrategy, creating simplified version")
    
    class AdamManciniNiftyStrategy:
        """Simplified version of Adam Mancini strategy."""
        
        def __init__(self, open_range_minutes=6):
            self.open_range_minutes = open_range_minutes
            
        def compute_levels(self, daily_ohlc):
            """Compute pivot, support and resistance levels."""
            pivot = (daily_ohlc["high"] + daily_ohlc["low"] + daily_ohlc["close"]) / 3.0
            r1 = 2 * pivot - daily_ohlc["low"]
            s1 = 2 * pivot - daily_ohlc["high"]
            r2 = pivot + (daily_ohlc["high"] - daily_ohlc["low"])
            s2 = pivot - (daily_ohlc["high"] - daily_ohlc["low"])
            r3 = r1 + (daily_ohlc["high"] - daily_ohlc["low"])
            s3 = s1 - (daily_ohlc["high"] - daily_ohlc["low"])
            
            levels = pd.DataFrame({
                "pivot": pivot,
                "r1": r1,
                "r2": r2,
                "r3": r3,
                "s1": s1,
                "s2": s2,
                "s3": s3,
            })
            return levels

def simulate_ticks(symbol, callback):
    """Simulate market tick data for paper trading."""
    # Start with current price around NIFTY level
    price = 24500.0
    
    # Initial price at the start of the day (for opening range)
    open_price = price
    
    while True:
        # Simulate price movement
        price_change = np.random.normal(0, 5)  # Random change with mean 0, std 5
        price += price_change
        
        # Create simulated tick
        current_time = datetime.now()
        
        # Reset price at the start of each day
        if current_time.hour == 9 and current_time.minute == 15:
            open_price = price
            
        tick = {
            symbol: {
                'last_price': price,
                'open': open_price,
                'high': max(price, open_price),
                'low': min(price, open_price),
                'close': price,
                'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        # Call callback with tick data
        callback(tick)
            
        time.sleep(1)  # Simulate 1-second updates

def run_paper_trading(symbol="NIFTY", opening_range_minutes=6):
    """Run Adam Mancini strategy in paper trading mode."""
    logger.info(f"Starting Adam Mancini strategy paper trading for {symbol}")
    logger.info(f"Opening range minutes: {opening_range_minutes}")
    
    # Initialize strategy
    strategy = AdamManciniNiftyStrategy(open_range_minutes=opening_range_minutes)
    
    # Current state
    position = 0  # 0: no position, 1: long, -1: short
    entry_price = 0
    daily_high_low = {}
    opening_range = {}
    
    # Trades list
    trades = []
    
    # Real-time data processing
    def on_ticks(ticks):
        nonlocal position, entry_price, daily_high_low, opening_range
        
        if not ticks or symbol not in ticks:
            return
            
        tick = ticks[symbol]
        current_time = datetime.now()
        market_open_time = current_time.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close_time = current_time.replace(hour=15, minute=30, second=0, microsecond=0)
        
        # Only trade during market hours
        if current_time.hour < 9 or current_time.hour > 15:
            return
            
        # Get current date
        current_date = current_time.strftime('%Y-%m-%d')
        
        # Print current price every 5 seconds
        if current_time.second % 5 == 0:
            logger.info(f"Current price: {tick['last_price']:.2f}, Position: {position}")
        
        # Initialize daily high/low if needed
        if current_date not in daily_high_low:
            daily_high_low[current_date] = {
                'high': float('-inf'),
                'low': float('inf'),
                'close': None
            }
            
        # Update daily high/low
        price = tick.get('last_price', tick.get('ltp', 0))
        daily_high_low[current_date]['high'] = max(daily_high_low[current_date]['high'], price)
        daily_high_low[current_date]['low'] = min(daily_high_low[current_date]['low'], price)
        daily_high_low[current_date]['close'] = price
        
        # Calculate opening range (first N minutes of trading)
        opening_range_end_time = market_open_time + timedelta(minutes=opening_range_minutes)
        
        if current_date not in opening_range and current_time.hour == 9:
            opening_range[current_date] = {
                'high': float('-inf'),
                'low': float('inf')
            }
            
        if current_date in opening_range and current_time.hour == 9 and current_time.minute < 9 + opening_range_minutes:
            opening_range[current_date]['high'] = max(opening_range[current_date]['high'], price)
            opening_range[current_date]['low'] = min(opening_range[current_date]['low'], price)
            
            if current_time.minute == 9 + opening_range_minutes - 1 and current_time.second >= 55:
                logger.info(f"Opening range established: High {opening_range[current_date]['high']:.2f}, Low {opening_range[current_date]['low']:.2f}")
            
        # Trading logic - only after opening range is established
        if (current_date in opening_range and 
            ((current_time.hour == 9 and current_time.minute >= 9 + opening_range_minutes) or current_time.hour > 9) and
            current_date in daily_high_low):
            
            # Simple pivot calculation
            pivot = (daily_high_low[current_date]['high'] + daily_high_low[current_date]['low'] + daily_high_low[current_date]['close']) / 3.0
            
            # Generate signals
            long_signal = (price > opening_range[current_date]['high']) and (price > pivot)
            short_signal = (price < opening_range[current_date]['low']) and (price < pivot)
            
            # Trading logic
            if long_signal and position <= 0:
                # Long signal - buy
                logger.info(f"PAPER TRADE: BUY {symbol} at {price:.2f}")
                trades.append({
                    'timestamp': current_time,
                    'action': 'BUY',
                    'price': price,
                    'quantity': 1
                })
                position = 1
                entry_price = price
                    
            elif short_signal and position >= 0:
                # Short signal - sell
                logger.info(f"PAPER TRADE: SELL {symbol} at {price:.2f}")
                trades.append({
                    'timestamp': current_time,
                    'action': 'SELL',
                    'price': price,
                    'quantity': 1
                })
                position = -1
                entry_price = price
    
    # Start simulation thread
    simulation_thread = threading.Thread(target=simulate_ticks, args=(symbol, on_ticks), daemon=True)
    simulation_thread.start()
    
    logger.info("Paper trading simulation started. Press Ctrl+C to stop.")
    
    try:
        # Keep the main thread running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Paper trading simulation stopped by user.")
        
        # Print trade summary
        if trades:
            logger.info("\nTrade Summary:")
            for i, trade in enumerate(trades):
                logger.info(f"{i+1}. {trade['timestamp']} | {trade['action']} at {trade['price']:.2f}")
            
            # Calculate P&L
            buy_trades = [t for t in trades if t['action'] == 'BUY']
            sell_trades = [t for t in trades if t['action'] == 'SELL']
            
            total_buy = sum(t['price'] for t in buy_trades)
            total_sell = sum(t['price'] for t in sell_trades)
            
            pnl = total_sell - total_buy
            
            logger.info(f"\nTotal trades: {len(trades)}")
            logger.info(f"Buy trades: {len(buy_trades)}")
            logger.info(f"Sell trades: {len(sell_trades)}")
            logger.info(f"P&L: {pnl:.2f}")
        else:
            logger.info("No trades executed during the simulation.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Adam Mancini strategy in paper trading mode")
    
    parser.add_argument(
        "--symbol", 
        type=str, 
        default="NIFTY",
        help="Trading symbol"
    )
    parser.add_argument(
        "--open-range-minutes", 
        type=int, 
        default=6,
        help="Number of minutes for the opening range (default: 6)"
    )
    
    args = parser.parse_args()
    
    run_paper_trading(symbol=args.symbol, opening_range_minutes=args.open_range_minutes)