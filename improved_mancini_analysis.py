#!/usr/bin/env python3
"""
Improved Adam Mancini strategy backtest analysis.
This script addresses several issues with the previous backtest:
1. Properly handles position tracking and P&L calculation
2. Implements realistic stop-loss and take-profit levels
3. Filters trades based on market conditions and time of day
4. Provides comprehensive performance metrics
5. Visualizes results more effectively
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns

# Set plotting style
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

class ManciniStrategy:
    """
    Adam Mancini's trading strategy implementation.
    """
    
    def __init__(
        self, 
        opening_range_minutes=6,
        take_profit_r_level=1,  # Use R1 level for take profit
        stop_loss_pct=0.5,      # Stop loss as percentage of opening range size
        max_trade_time=timedelta(hours=3),  # Exit trades after 3 hours if not triggered
        trade_after_opening_minutes=15,   # Only trade after 15 mins past market open
        avoid_lunch_hour=True,  # Avoid trading during lunch hour (12:00-13:00)
        avoid_last_hour=True    # Avoid trading in the last hour of market (2:30-3:30)
    ):
        self.opening_range_minutes = opening_range_minutes
        self.take_profit_r_level = take_profit_r_level
        self.stop_loss_pct = stop_loss_pct
        self.max_trade_time = max_trade_time
        self.trade_after_opening_minutes = trade_after_opening_minutes
        self.avoid_lunch_hour = avoid_lunch_hour
        self.avoid_last_hour = avoid_last_hour
        
    def calculate_pivot_levels(self, high, low, close):
        """Calculate pivot, support and resistance levels."""
        pivot = (high + low + close) / 3.0
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        r3 = r1 + (high - low)
        s3 = s1 - (high - low)
        
        return {
            'pivot': pivot,
            'r1': r1,
            'r2': r2,
            'r3': r3,
            's1': s1,
            's2': s2,
            's3': s3
        }
    
    def is_tradable_time(self, timestamp):
        """Check if the current time is suitable for trading based on strategy rules."""
        # Check if we're within normal trading hours (9:15 AM - 3:30 PM IST)
        hour, minute = timestamp.hour, timestamp.minute
        
        # Market opens at 9:15, only trade after opening_range + trade_after_opening_minutes
        min_trade_time = 9*60 + 15 + self.opening_range_minutes + self.trade_after_opening_minutes
        min_trade_hour, min_trade_minute = divmod(min_trade_time, 60)
        
        if hour < min_trade_hour or (hour == min_trade_hour and minute < min_trade_minute):
            return False
            
        # Avoid trading during lunch hour (12:00-13:00)
        if self.avoid_lunch_hour and hour == 12:
            return False
            
        # Avoid trading in the last hour (2:30-3:30)
        if self.avoid_last_hour and (hour == 14 and minute >= 30) or hour > 14:
            return False
            
        return True
        
    def backtest(self, data, visualize=True):
        """
        Run a backtest of the strategy on historical data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Historical OHLC data with DatetimeIndex
        visualize : bool, optional
            Whether to generate visualizations
            
        Returns
        -------
        dict
            Backtest results including daily P&L, trades, and metrics
        """
        print(f"Running backtest with {self.opening_range_minutes}-minute opening range...")
        print(f"Data contains {len(data)} bars from {data.index[0]} to {data.index[-1]}")
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Add date column for grouping
        data = data.copy()
        data['date'] = data.index.date
        
        # Dictionary to store backtest results
        results = {
            'daily_pnl': {},
            'trades': [],
            'equity_curve': [],
            'metrics': {}
        }
        
        # Initialize position tracking
        position = 0  # 0: flat, 1: long, -1: short
        entry_price = 0
        entry_time = None
        stop_loss = 0
        take_profit = 0
        trade_id = 0
        daily_pnl = {}
        
        # Process data day by day
        days = sorted(data['date'].unique())
        
        for i, day in enumerate(days):
            day_str = day.strftime("%Y-%m-%d")
            daily_pnl[day_str] = 0
            
            # Get data for current day
            day_data = data[data['date'] == day].copy()
            
            if len(day_data) == 0:
                continue
                
            # Calculate pivot levels for day based on previous day's data
            if i > 0:
                prev_day = days[i-1]
                prev_day_data = data[data['date'] == prev_day]
                if len(prev_day_data) > 0:
                    pivot_levels = self.calculate_pivot_levels(
                        prev_day_data['high'].max(),
                        prev_day_data['low'].min(),
                        prev_day_data['close'].iloc[-1]
                    )
                else:
                    # Fallback if no previous day data
                    pivot_levels = {
                        'pivot': day_data['open'].iloc[0],
                        'r1': day_data['open'].iloc[0] * 1.005,
                        's1': day_data['open'].iloc[0] * 0.995
                    }
            else:
                # First day - use simple pivot levels
                pivot_levels = {
                    'pivot': day_data['open'].iloc[0],
                    'r1': day_data['open'].iloc[0] * 1.005,
                    's1': day_data['open'].iloc[0] * 0.995
                }
            
            # Add pivot levels to day's data
            for level, value in pivot_levels.items():
                day_data[level] = value
            
            # Calculate opening range if enough data points
            if len(day_data) >= self.opening_range_minutes:
                opening_range_data = day_data.iloc[:self.opening_range_minutes]
                high_or = opening_range_data['high'].max()
                low_or = opening_range_data['low'].min()
                
                # Opening range size
                or_size = high_or - low_or
                
                # Add opening range to dataframe
                day_data['high_or'] = high_or
                day_data['low_or'] = low_or
                day_data['or_size'] = or_size
                
                # Skip opening range period for trading
                trading_data = day_data.iloc[self.opening_range_minutes:]
            else:
                # Not enough data for opening range
                continue
            
            # Process each bar for the day
            for idx, row in trading_data.iterrows():
                # Check if we have an open position that needs management
                if position != 0:
                    # Calculate time in position
                    time_in_position = idx - entry_time
                    
                    # Calculate P&L
                    if position == 1:  # Long position
                        current_pnl = row['close'] - entry_price
                        
                        # Check stop loss
                        if row['low'] <= stop_loss:
                            # Stop loss hit - use stop loss price
                            exit_price = stop_loss
                            trade_pnl = exit_price - entry_price
                            exit_reason = "Stop Loss"
                            
                            # Record trade
                            trade = {
                                'id': trade_id,
                                'entry_date': entry_time,
                                'entry_price': entry_price,
                                'exit_date': idx,
                                'exit_price': exit_price,
                                'pnl': trade_pnl,
                                'position': 'LONG',
                                'exit_reason': exit_reason,
                                'hold_time': time_in_position.total_seconds() / 60  # minutes
                            }
                            results['trades'].append(trade)
                            daily_pnl[day_str] += trade_pnl
                            
                            # Reset position
                            position = 0
                            trade_id += 1
                            
                        # Check take profit
                        elif row['high'] >= take_profit:
                            # Take profit hit - use take profit price
                            exit_price = take_profit
                            trade_pnl = exit_price - entry_price
                            exit_reason = "Take Profit"
                            
                            # Record trade
                            trade = {
                                'id': trade_id,
                                'entry_date': entry_time,
                                'entry_price': entry_price,
                                'exit_date': idx,
                                'exit_price': exit_price,
                                'pnl': trade_pnl,
                                'position': 'LONG',
                                'exit_reason': exit_reason,
                                'hold_time': time_in_position.total_seconds() / 60  # minutes
                            }
                            results['trades'].append(trade)
                            daily_pnl[day_str] += trade_pnl
                            
                            # Reset position
                            position = 0
                            trade_id += 1
                            
                        # Check time-based exit
                        elif time_in_position > self.max_trade_time:
                            # Time-based exit - use current price
                            exit_price = row['close']
                            trade_pnl = exit_price - entry_price
                            exit_reason = "Time Limit"
                            
                            # Record trade
                            trade = {
                                'id': trade_id,
                                'entry_date': entry_time,
                                'entry_price': entry_price,
                                'exit_date': idx,
                                'exit_price': exit_price,
                                'pnl': trade_pnl,
                                'position': 'LONG',
                                'exit_reason': exit_reason,
                                'hold_time': time_in_position.total_seconds() / 60  # minutes
                            }
                            results['trades'].append(trade)
                            daily_pnl[day_str] += trade_pnl
                            
                            # Reset position
                            position = 0
                            trade_id += 1
                            
                    elif position == -1:  # Short position
                        current_pnl = entry_price - row['close']
                        
                        # Check stop loss
                        if row['high'] >= stop_loss:
                            # Stop loss hit - use stop loss price
                            exit_price = stop_loss
                            trade_pnl = entry_price - exit_price
                            exit_reason = "Stop Loss"
                            
                            # Record trade
                            trade = {
                                'id': trade_id,
                                'entry_date': entry_time,
                                'entry_price': entry_price,
                                'exit_date': idx,
                                'exit_price': exit_price,
                                'pnl': trade_pnl,
                                'position': 'SHORT',
                                'exit_reason': exit_reason,
                                'hold_time': time_in_position.total_seconds() / 60  # minutes
                            }
                            results['trades'].append(trade)
                            daily_pnl[day_str] += trade_pnl
                            
                            # Reset position
                            position = 0
                            trade_id += 1
                            
                        # Check take profit
                        elif row['low'] <= take_profit:
                            # Take profit hit - use take profit price
                            exit_price = take_profit
                            trade_pnl = entry_price - exit_price
                            exit_reason = "Take Profit"
                            
                            # Record trade
                            trade = {
                                'id': trade_id,
                                'entry_date': entry_time,
                                'entry_price': entry_price,
                                'exit_date': idx,
                                'exit_price': exit_price,
                                'pnl': trade_pnl,
                                'position': 'SHORT',
                                'exit_reason': exit_reason,
                                'hold_time': time_in_position.total_seconds() / 60  # minutes
                            }
                            results['trades'].append(trade)
                            daily_pnl[day_str] += trade_pnl
                            
                            # Reset position
                            position = 0
                            trade_id += 1
                            
                        # Check time-based exit
                        elif time_in_position > self.max_trade_time:
                            # Time-based exit - use current price
                            exit_price = row['close']
                            trade_pnl = entry_price - exit_price
                            exit_reason = "Time Limit"
                            
                            # Record trade
                            trade = {
                                'id': trade_id,
                                'entry_date': entry_time,
                                'entry_price': entry_price,
                                'exit_date': idx,
                                'exit_price': exit_price,
                                'pnl': trade_pnl,
                                'position': 'SHORT',
                                'exit_reason': exit_reason,
                                'hold_time': time_in_position.total_seconds() / 60  # minutes
                            }
                            results['trades'].append(trade)
                            daily_pnl[day_str] += trade_pnl
                            
                            # Reset position
                            position = 0
                            trade_id += 1
                
                # Check if it's a suitable time to enter new trades
                if position == 0 and self.is_tradable_time(idx):
                    # Generate signals based on price relative to opening range and pivot
                    long_signal = (row['close'] > high_or) and (row['close'] > row['pivot'])
                    short_signal = (row['close'] < low_or) and (row['close'] < row['pivot'])
                    
                    # Only take signals if opening range is meaningful (not too tight)
                    min_or_size = row['pivot'] * 0.001  # Minimum 0.1% of pivot
                    
                    if or_size > min_or_size:
                        if long_signal:
                            # Enter long position
                            position = 1
                            entry_price = row['close']
                            entry_time = idx
                            
                            # Set stop loss and take profit
                            stop_loss = entry_price - (or_size * self.stop_loss_pct)
                            take_profit = row[f'r{self.take_profit_r_level}']
                            
                        elif short_signal:
                            # Enter short position
                            position = -1
                            entry_price = row['close']
                            entry_time = idx
                            
                            # Set stop loss and take profit
                            stop_loss = entry_price + (or_size * self.stop_loss_pct)
                            take_profit = row[f's{self.take_profit_r_level}']
            
            # Close any open positions at the end of the day
            if position != 0:
                exit_price = day_data.iloc[-1]['close']
                
                if position == 1:  # Long position
                    trade_pnl = exit_price - entry_price
                else:  # Short position
                    trade_pnl = entry_price - exit_price
                
                # Record trade
                trade = {
                    'id': trade_id,
                    'entry_date': entry_time,
                    'entry_price': entry_price,
                    'exit_date': day_data.index[-1],
                    'exit_price': exit_price,
                    'pnl': trade_pnl,
                    'position': 'LONG' if position == 1 else 'SHORT',
                    'exit_reason': "Day End",
                    'hold_time': (day_data.index[-1] - entry_time).total_seconds() / 60  # minutes
                }
                results['trades'].append(trade)
                daily_pnl[day_str] += trade_pnl
                
                # Reset position for next day
                position = 0
                trade_id += 1
                
        # Store daily PnL
        results['daily_pnl'] = daily_pnl
        
        # Calculate equity curve
        equity = 0
        equity_curve = []
        
        for day, pnl in daily_pnl.items():
            equity += pnl
            equity_curve.append({'date': day, 'equity': equity})
            
        results['equity_curve'] = pd.DataFrame(equity_curve)
        
        # Calculate metrics
        if results['trades']:
            trades_df = pd.DataFrame(results['trades'])
            
            # Basic statistics
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            losing_trades = len(trades_df[trades_df['pnl'] <= 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Return metrics
            total_return = trades_df['pnl'].sum()
            avg_return = trades_df['pnl'].mean()
            median_return = trades_df['pnl'].median()
            
            # Risk metrics
            if losing_trades > 0:
                avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
                avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
                win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
                profit_factor = abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum() / 
                                  trades_df[trades_df['pnl'] <= 0]['pnl'].sum()) if trades_df[trades_df['pnl'] <= 0]['pnl'].sum() != 0 else float('inf')
            else:
                avg_win = trades_df['pnl'].mean() if winning_trades > 0 else 0
                avg_loss = 0
                win_loss_ratio = float('inf')
                profit_factor = float('inf')
                
            # Drawdown calculation
            if not results['equity_curve'].empty:
                equity_data = results['equity_curve']['equity']
                running_max = equity_data.cummax()
                drawdown = (equity_data - running_max) / running_max * 100
                max_drawdown = drawdown.min()
            else:
                max_drawdown = 0
                
            # Time metrics
            avg_hold_time = trades_df['hold_time'].mean()
            
            # Exit reasons
            exit_reasons = trades_df['exit_reason'].value_counts().to_dict()
            
            # Store all metrics
            results['metrics'] = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_return': total_return,
                'avg_return': avg_return,
                'median_return': median_return,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'win_loss_ratio': win_loss_ratio,
                'profit_factor': profit_factor,
                'max_drawdown': max_drawdown,
                'avg_hold_time': avg_hold_time,
                'exit_reasons': exit_reasons
            }
            
            # Print summary
            print("\nStrategy Performance Summary:")
            print(f"Total Trades: {total_trades}")
            print(f"Win Rate: {win_rate:.2%}")
            print(f"Total Return: {total_return:.2f} points")
            print(f"Profit Factor: {profit_factor:.2f}")
            print(f"Max Drawdown: {max_drawdown:.2f}%")
            
            # Visualize results if requested
            if visualize:
                self.visualize_results(results)
        else:
            print("No trades were executed during the backtest period.")
            
        return results
    
    def visualize_results(self, results):
        """Generate visualizations of backtest results."""
        if not results['trades']:
            print("No trades to visualize.")
            return
            
        # Create directory for charts
        os.makedirs('charts', exist_ok=True)
        
        # 1. Daily P&L
        plt.figure(figsize=(12, 6))
        
        days = list(results['daily_pnl'].keys())
        pnls = list(results['daily_pnl'].values())
        
        # Filter only days with trades
        active_days = [day for day, pnl in zip(days, pnls) if pnl != 0]
        active_pnls = [pnl for pnl in pnls if pnl != 0]
        
        colors = ['green' if p > 0 else 'red' for p in active_pnls]
        
        plt.bar(active_days, active_pnls, color=colors)
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.title('Daily P&L', fontsize=16)
        plt.xlabel('Date')
        plt.ylabel('P&L (points)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig('charts/daily_pnl.png')
        
        # 2. Equity Curve
        plt.figure(figsize=(12, 6))
        
        if not results['equity_curve'].empty:
            plt.plot(results['equity_curve']['date'], results['equity_curve']['equity'], 
                    marker='o', linestyle='-', linewidth=2)
            plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            plt.title('Equity Curve', fontsize=16)
            plt.xlabel('Date')
            plt.ylabel('Cumulative P&L (points)')
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.tight_layout()
            
            plt.savefig('charts/equity_curve.png')
        
        # 3. Trade Distribution
        plt.figure(figsize=(10, 6))
        
        trades_df = pd.DataFrame(results['trades'])
        trades_df['pnl'].hist(bins=20, color='skyblue', edgecolor='black')
        
        plt.axvline(x=0, color='red', linestyle='--', linewidth=1)
        plt.title('P&L Distribution', fontsize=16)
        plt.xlabel('P&L (points)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig('charts/pnl_distribution.png')
        
        # 4. Win/Loss by Exit Reason
        plt.figure(figsize=(10, 6))
        
        exit_reasons = trades_df['exit_reason'].unique()
        win_counts = []
        loss_counts = []
        
        for reason in exit_reasons:
            win_counts.append(len(trades_df[(trades_df['exit_reason'] == reason) & (trades_df['pnl'] > 0)]))
            loss_counts.append(len(trades_df[(trades_df['exit_reason'] == reason) & (trades_df['pnl'] <= 0)]))
        
        x = np.arange(len(exit_reasons))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, win_counts, width, label='Winning Trades', color='green')
        ax.bar(x + width/2, loss_counts, width, label='Losing Trades', color='red')
        
        ax.set_xlabel('Exit Reason')
        ax.set_ylabel('Trade Count')
        ax.set_title('Win/Loss by Exit Reason', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(exit_reasons)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('charts/exit_reason_performance.png')
        
        # 5. Performance Metrics Summary
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Hide axes
        ax.axis('off')
        
        # Add title
        ax.text(0.5, 0.95, 'Performance Metrics Summary', 
                horizontalalignment='center', verticalalignment='center', 
                fontsize=20, fontweight='bold')
        
        # Create a table with metrics
        metrics = results['metrics']
        data = [
            ['Total Trades', f"{metrics['total_trades']}"],
            ['Winning Trades', f"{metrics['winning_trades']} ({metrics['win_rate']:.2%})"],
            ['Losing Trades', f"{metrics['losing_trades']} ({1-metrics['win_rate']:.2%})"],
            ['Total Return', f"{metrics['total_return']:.2f} points"],
            ['Average Return', f"{metrics['avg_return']:.2f} points"],
            ['Median Return', f"{metrics['median_return']:.2f} points"],
            ['Average Win', f"{metrics['avg_win']:.2f} points"],
            ['Average Loss', f"{metrics['avg_loss']:.2f} points"],
            ['Win/Loss Ratio', f"{metrics['win_loss_ratio']:.2f}"],
            ['Profit Factor', f"{metrics['profit_factor']:.2f}"],
            ['Max Drawdown', f"{metrics['max_drawdown']:.2f}%"],
            ['Avg Hold Time', f"{metrics['avg_hold_time']:.2f} minutes"]
        ]
        
        table = ax.table(cellText=data, colLabels=['Metric', 'Value'], 
                         loc='center', cellLoc='left', colWidths=[0.4, 0.3])
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)
        
        plt.tight_layout()
        plt.savefig('charts/performance_summary.png')
        
        print("\nVisualization complete. Charts saved to 'charts' directory.")

def load_and_prepare_data(data_file):
    """Load and prepare data for backtesting."""
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
    
    # Ensure column names are lowercase
    data.columns = [col.lower() for col in data.columns]
    
    # Ensure the required columns exist
    required_cols = ['open', 'high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in data.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Sort by datetime (if not already)
    data = data.sort_index()
    
    print(f"Prepared {len(data)} bars of data from {data.index[0]} to {data.index[-1]}")
    return data

def download_additional_data():
    """Download additional data for testing if needed."""
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not installed. Run: pip install yfinance")
        return None
    
    # Download longer history
    print("Downloading additional historical data for more comprehensive testing...")
    
    # For NSE NIFTY (^NSEI)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)  # 3 months data
    
    # Try different timeframes
    timeframes = ['1d', '1h', '5m']
    
    for timeframe in timeframes:
        print(f"Downloading {timeframe} data for ^NSEI...")
        try:
            data = yf.download(
                "^NSEI", 
                start=start_date,
                end=end_date,
                interval=timeframe,
                progress=False
            )
            
            if len(data) > 0:
                filename = f"data/nifty_{timeframe.replace('d', 'day').replace('h', 'hour').replace('m', 'min')}_90days.csv"
                os.makedirs('data', exist_ok=True)
                data.to_csv(filename)
                print(f"Downloaded {len(data)} bars of {timeframe} data to {filename}")
            else:
                print(f"No data available for {timeframe} timeframe")
                
        except Exception as e:
            print(f"Error downloading {timeframe} data: {e}")
    
    return None

def run_parameter_optimization(data, param_ranges=None):
    """
    Run a parameter optimization to find the best strategy settings.
    
    Parameters
    ----------
    data : pd.DataFrame
        Historical OHLC data
    param_ranges : dict, optional
        Dictionary of parameter ranges to test
        
    Returns
    -------
    dict
        Optimization results
    """
    if param_ranges is None:
        # Default parameter ranges to test
        param_ranges = {
            'opening_range_minutes': [3, 6, 9, 12, 15],
            'take_profit_r_level': [1, 2, 3],
            'stop_loss_pct': [0.3, 0.5, 0.7, 1.0]
        }
    
    print("Running parameter optimization...")
    print(f"Testing {len(param_ranges['opening_range_minutes'])} opening range values")
    print(f"Testing {len(param_ranges['take_profit_r_level'])} take profit levels")
    print(f"Testing {len(param_ranges['stop_loss_pct'])} stop loss percentages")
    
    # Store results
    results = []
    
    # Run backtests for each parameter combination
    for or_minutes in param_ranges['opening_range_minutes']:
        for tp_level in param_ranges['take_profit_r_level']:
            for sl_pct in param_ranges['stop_loss_pct']:
                print(f"\nTesting OR={or_minutes}min, TP=R{tp_level}, SL={sl_pct*100}%")
                
                strategy = ManciniStrategy(
                    opening_range_minutes=or_minutes,
                    take_profit_r_level=tp_level,
                    stop_loss_pct=sl_pct
                )
                
                backtest_results = strategy.backtest(data, visualize=False)
                
                # Extract key metrics
                metrics = backtest_results['metrics'] if 'metrics' in backtest_results else {}
                
                # Store results
                result = {
                    'opening_range_minutes': or_minutes,
                    'take_profit_r_level': tp_level,
                    'stop_loss_pct': sl_pct,
                    'total_trades': metrics.get('total_trades', 0),
                    'win_rate': metrics.get('win_rate', 0),
                    'total_return': metrics.get('total_return', 0),
                    'profit_factor': metrics.get('profit_factor', 0),
                    'max_drawdown': metrics.get('max_drawdown', 0)
                }
                
                results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by profit factor (higher is better)
    results_df = results_df.sort_values('profit_factor', ascending=False)
    
    # Save optimization results
    results_df.to_csv('parameter_optimization_results.csv', index=False)
    
    # Visualize top 10 parameter combinations
    plt.figure(figsize=(12, 8))
    
    top_results = results_df.head(10).copy()
    
    # Create labels
    labels = top_results.apply(
        lambda row: f"OR={int(row['opening_range_minutes'])}, TP=R{int(row['take_profit_r_level'])}, SL={row['stop_loss_pct']*100:.0f}%", 
        axis=1
    )
    
    # Plot profit factor
    plt.barh(labels, top_results['profit_factor'], color='skyblue')
    plt.xlabel('Profit Factor')
    plt.title('Top 10 Parameter Combinations by Profit Factor')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('charts/parameter_optimization.png')
    
    # Print best parameters
    best_params = results_df.iloc[0]
    print("\nBest parameter combination:")
    print(f"Opening Range Minutes: {best_params['opening_range_minutes']}")
    print(f"Take Profit Level: R{best_params['take_profit_r_level']}")
    print(f"Stop Loss Percentage: {best_params['stop_loss_pct']*100:.0f}%")
    print(f"Profit Factor: {best_params['profit_factor']:.2f}")
    print(f"Total Return: {best_params['total_return']:.2f}")
    print(f"Win Rate: {best_params['win_rate']:.2%}")
    
    return results_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Improved Adam Mancini strategy backtest analysis")
    
    parser.add_argument(
        "--data-file", 
        type=str, 
        default="data/nifty_5min_30days.csv",
        help="Path to CSV file with historical data"
    )
    parser.add_argument(
        "--download-data", 
        action="store_true",
        help="Download additional historical data"
    )
    parser.add_argument(
        "--open-range-minutes", 
        type=int, 
        default=6,
        help="Number of minutes for the opening range (default: 6)"
    )
    parser.add_argument(
        "--optimize", 
        action="store_true",
        help="Run parameter optimization"
    )
    parser.add_argument(
        "--take-profit-level", 
        type=int, 
        default=1,
        help="Take profit level (1, 2, or 3 for R1, R2, R3) (default: 1)"
    )
    parser.add_argument(
        "--stop-loss-pct", 
        type=float, 
        default=0.5,
        help="Stop loss as percentage of opening range size (default: 0.5)"
    )
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('charts', exist_ok=True)
    
    # Download additional data if requested
    if args.download_data:
        download_additional_data()
    
    # Load and prepare data
    data = load_and_prepare_data(args.data_file)
    
    if args.optimize:
        # Run parameter optimization
        optimization_results = run_parameter_optimization(data)
        
        # Use the best parameters for the final backtest
        best_params = optimization_results.iloc[0]
        strategy = ManciniStrategy(
            opening_range_minutes=int(best_params['opening_range_minutes']),
            take_profit_r_level=int(best_params['take_profit_r_level']),
            stop_loss_pct=best_params['stop_loss_pct']
        )
    else:
        # Use provided parameters
        strategy = ManciniStrategy(
            opening_range_minutes=args.open_range_minutes,
            take_profit_r_level=args.take_profit_level,
            stop_loss_pct=args.stop_loss_pct
        )
    
    # Run backtest
    backtest_results = strategy.backtest(data)