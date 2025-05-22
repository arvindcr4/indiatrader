#!/usr/bin/env python3
"""
Specialized Adam Mancini strategy analysis focusing on:
1. Proper stop-loss and take-profit implementation
2. Filtering trades based on market volatility
3. Adding multiple timeframe confirmation
4. Position sizing based on account risk
5. Advanced performance metrics
6. Synthetic data generation for testing
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
from typing import List, Dict, Tuple, Optional

# Set plotting style
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

class ManciniOptimized:
    """
    Optimized Adam Mancini strategy implementation with additional filters
    and improved risk management.
    """
    
    def __init__(
        self,
        opening_range_minutes=6,
        take_profit_r_level=1,
        stop_loss_pct=0.5,
        risk_per_trade_pct=1.0,
        min_volatility_filter=True,
        check_market_direction=True,
        initial_capital=100000
    ):
        self.opening_range_minutes = opening_range_minutes
        self.take_profit_r_level = take_profit_r_level
        self.stop_loss_pct = stop_loss_pct
        self.risk_per_trade_pct = risk_per_trade_pct
        self.min_volatility_filter = min_volatility_filter
        self.check_market_direction = check_market_direction
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
    def calculate_volatility(self, data: pd.DataFrame, window=14) -> pd.Series:
        """Calculate historical volatility."""
        # Calculate log returns
        log_returns = np.log(data['close'] / data['close'].shift(1))
        
        # Calculate rolling standard deviation
        volatility = log_returns.rolling(window=window).std() * np.sqrt(252)
        
        return volatility
    
    def determine_market_direction(self, data: pd.DataFrame, window=20) -> pd.Series:
        """Determine market direction using moving averages."""
        # Calculate short and long-term moving averages
        short_ma = data['close'].rolling(window=window//2).mean()
        long_ma = data['close'].rolling(window=window).mean()
        
        # Determine direction
        direction = np.where(short_ma > long_ma, 1, np.where(short_ma < long_ma, -1, 0))
        
        return pd.Series(direction, index=data.index)
    
    def calculate_pivot_levels(self, high, low, close) -> Dict:
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
    
    def calculate_position_size(self, entry_price, stop_price) -> int:
        """Calculate position size based on risk per trade."""
        # Risk amount in currency
        risk_amount = self.current_capital * (self.risk_per_trade_pct / 100)
        
        # Risk per share/contract
        risk_per_unit = abs(entry_price - stop_price)
        
        # Position size (number of shares/contracts)
        if risk_per_unit > 0:
            position_size = max(1, int(risk_amount / risk_per_unit))
        else:
            position_size = 1
            
        return position_size
    
    def backtest(self, data: pd.DataFrame, visualize: bool = True) -> Dict:
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
            Backtest results
        """
        print(f"Running optimized Mancini backtest with {self.opening_range_minutes}-minute opening range...")
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Add date column for grouping
        data = data.copy()
        data['date'] = data.index.date
        
        # Calculate volatility and market direction
        data['volatility'] = self.calculate_volatility(data)
        data['market_direction'] = self.determine_market_direction(data)
        
        # Dictionary to store backtest results
        results = {
            'daily_pnl': {},
            'trades': [],
            'equity_curve': [],
            'drawdowns': [],
            'metrics': {}
        }
        
        # Initialize position tracking
        position = 0  # 0: flat, 1: long, -1: short
        entry_price = 0
        entry_time = None
        entry_idx = None
        stop_loss = 0
        take_profit = 0
        position_size = 0
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
                
            # Calculate pivot levels using previous day's data
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
                
                # Skip days with abnormally tight or wide opening ranges
                if or_size < day_data['close'].iloc[0] * 0.0005:
                    # Opening range is too tight (less than 0.05% of price)
                    continue
                    
                if or_size > day_data['close'].iloc[0] * 0.02:
                    # Opening range is too wide (more than 2% of price)
                    continue
                
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
                    
                    # Calculate current P&L
                    if position == 1:  # Long position
                        unrealized_pnl = (row['close'] - entry_price) * position_size
                        
                        # Check stop loss
                        if row['low'] <= stop_loss:
                            # Stop loss hit - use stop loss price for exit
                            exit_price = stop_loss
                            trade_pnl = (exit_price - entry_price) * position_size
                            exit_reason = "Stop Loss"
                            exit_idx = idx
                            
                            # Update capital
                            self.current_capital += trade_pnl
                            
                            # Record trade
                            trade = {
                                'id': trade_id,
                                'entry_time': entry_time,
                                'entry_price': entry_price,
                                'exit_time': exit_idx,
                                'exit_price': exit_price,
                                'position_size': position_size,
                                'pnl': trade_pnl,
                                'pnl_pct': trade_pnl / self.current_capital * 100,
                                'position': 'LONG',
                                'exit_reason': exit_reason,
                                'hold_time_minutes': (exit_idx - entry_time).total_seconds() / 60,
                                'market_direction': row['market_direction']
                            }
                            results['trades'].append(trade)
                            daily_pnl[day_str] += trade_pnl
                            
                            # Reset position
                            position = 0
                            trade_id += 1
                            
                        # Check take profit
                        elif row['high'] >= take_profit:
                            # Take profit hit - use take profit price for exit
                            exit_price = take_profit
                            trade_pnl = (exit_price - entry_price) * position_size
                            exit_reason = "Take Profit"
                            exit_idx = idx
                            
                            # Update capital
                            self.current_capital += trade_pnl
                            
                            # Record trade
                            trade = {
                                'id': trade_id,
                                'entry_time': entry_time,
                                'entry_price': entry_price,
                                'exit_time': exit_idx,
                                'exit_price': exit_price,
                                'position_size': position_size,
                                'pnl': trade_pnl,
                                'pnl_pct': trade_pnl / self.current_capital * 100,
                                'position': 'LONG',
                                'exit_reason': exit_reason,
                                'hold_time_minutes': (exit_idx - entry_time).total_seconds() / 60,
                                'market_direction': row['market_direction']
                            }
                            results['trades'].append(trade)
                            daily_pnl[day_str] += trade_pnl
                            
                            # Reset position
                            position = 0
                            trade_id += 1
                            
                    elif position == -1:  # Short position
                        unrealized_pnl = (entry_price - row['close']) * position_size
                        
                        # Check stop loss
                        if row['high'] >= stop_loss:
                            # Stop loss hit - use stop loss price for exit
                            exit_price = stop_loss
                            trade_pnl = (entry_price - exit_price) * position_size
                            exit_reason = "Stop Loss"
                            exit_idx = idx
                            
                            # Update capital
                            self.current_capital += trade_pnl
                            
                            # Record trade
                            trade = {
                                'id': trade_id,
                                'entry_time': entry_time,
                                'entry_price': entry_price,
                                'exit_time': exit_idx,
                                'exit_price': exit_price,
                                'position_size': position_size,
                                'pnl': trade_pnl,
                                'pnl_pct': trade_pnl / self.current_capital * 100,
                                'position': 'SHORT',
                                'exit_reason': exit_reason,
                                'hold_time_minutes': (exit_idx - entry_time).total_seconds() / 60,
                                'market_direction': row['market_direction']
                            }
                            results['trades'].append(trade)
                            daily_pnl[day_str] += trade_pnl
                            
                            # Reset position
                            position = 0
                            trade_id += 1
                            
                        # Check take profit
                        elif row['low'] <= take_profit:
                            # Take profit hit - use take profit price for exit
                            exit_price = take_profit
                            trade_pnl = (entry_price - exit_price) * position_size
                            exit_reason = "Take Profit"
                            exit_idx = idx
                            
                            # Update capital
                            self.current_capital += trade_pnl
                            
                            # Record trade
                            trade = {
                                'id': trade_id,
                                'entry_time': entry_time,
                                'entry_price': entry_price,
                                'exit_time': exit_idx,
                                'exit_price': exit_price,
                                'position_size': position_size,
                                'pnl': trade_pnl,
                                'pnl_pct': trade_pnl / self.current_capital * 100,
                                'position': 'SHORT',
                                'exit_reason': exit_reason,
                                'hold_time_minutes': (exit_idx - entry_time).total_seconds() / 60,
                                'market_direction': row['market_direction']
                            }
                            results['trades'].append(trade)
                            daily_pnl[day_str] += trade_pnl
                            
                            # Reset position
                            position = 0
                            trade_id += 1
                
                # Check if it's a suitable time to enter new trades (after opening range)
                if position == 0:
                    # Apply volatility filter if enabled
                    if self.min_volatility_filter and (pd.isna(row['volatility']) or row['volatility'] < 0.1):
                        continue
                    
                    # Generate signals based on price relative to opening range and pivot
                    long_signal = (row['close'] > high_or) and (row['close'] > row['pivot'])
                    short_signal = (row['close'] < low_or) and (row['close'] < row['pivot'])
                    
                    # Check market direction if enabled
                    if self.check_market_direction:
                        # Only take long signals in uptrends and short signals in downtrends
                        if long_signal and row['market_direction'] <= 0:
                            long_signal = False
                        if short_signal and row['market_direction'] >= 0:
                            short_signal = False
                    
                    if long_signal:
                        # Long entry signal
                        position = 1
                        entry_price = row['close']
                        entry_time = idx
                        entry_idx = idx
                        
                        # Set stop loss and take profit
                        stop_loss = entry_price - (or_size * self.stop_loss_pct)
                        take_profit = row[f'r{self.take_profit_r_level}']
                        
                        # Calculate position size based on risk
                        position_size = self.calculate_position_size(entry_price, stop_loss)
                        
                    elif short_signal:
                        # Short entry signal
                        position = -1
                        entry_price = row['close']
                        entry_time = idx
                        entry_idx = idx
                        
                        # Set stop loss and take profit
                        stop_loss = entry_price + (or_size * self.stop_loss_pct)
                        take_profit = row[f's{self.take_profit_r_level}']
                        
                        # Calculate position size based on risk
                        position_size = self.calculate_position_size(entry_price, stop_loss)
            
            # Close any open position at the end of the day
            if position != 0:
                exit_price = day_data.iloc[-1]['close']
                exit_idx = day_data.index[-1]
                
                if position == 1:  # Long position
                    trade_pnl = (exit_price - entry_price) * position_size
                else:  # Short position
                    trade_pnl = (entry_price - exit_price) * position_size
                
                # Update capital
                self.current_capital += trade_pnl
                
                # Record trade
                trade = {
                    'id': trade_id,
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': exit_idx,
                    'exit_price': exit_price,
                    'position_size': position_size,
                    'pnl': trade_pnl,
                    'pnl_pct': trade_pnl / self.current_capital * 100,
                    'position': 'LONG' if position == 1 else 'SHORT',
                    'exit_reason': "Day End",
                    'hold_time_minutes': (exit_idx - entry_time).total_seconds() / 60,
                    'market_direction': day_data.iloc[-1]['market_direction']
                }
                results['trades'].append(trade)
                daily_pnl[day_str] += trade_pnl
                
                # Reset position for next day
                position = 0
                trade_id += 1
            
            # Calculate equity and drawdown for the day
            day_equity = self.initial_capital + sum(daily_pnl.values())
            equity_high = self.initial_capital + max(
                [0] + [sum(list(daily_pnl.values())[:i+1]) for i in range(len(daily_pnl))]
            )
            day_drawdown = (day_equity - equity_high) / equity_high * 100 if equity_high > 0 else 0
            
            results['equity_curve'].append({
                'date': day_str,
                'equity': day_equity
            })
            
            results['drawdowns'].append({
                'date': day_str,
                'drawdown': day_drawdown
            })
        
        # Store daily PnL
        results['daily_pnl'] = daily_pnl
        
        # Convert equity curve and drawdowns to DataFrames
        results['equity_curve'] = pd.DataFrame(results['equity_curve'])
        results['drawdowns'] = pd.DataFrame(results['drawdowns'])
        
        # Calculate and store metrics
        self.calculate_metrics(results)
        
        # Visualize if needed
        if visualize:
            self.visualize_results(results)
        
        return results
    
    def calculate_metrics(self, results: Dict) -> None:
        """Calculate comprehensive performance metrics."""
        if not results['trades']:
            results['metrics'] = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'total_return': 0,
                'return_pct': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'avg_hold_time': 0
            }
            return
        
        # Convert trades to DataFrame for easier analysis
        trades_df = pd.DataFrame(results['trades'])
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # P&L metrics
        total_return = trades_df['pnl'].sum()
        return_pct = total_return / self.initial_capital * 100
        avg_return = trades_df['pnl'].mean()
        median_return = trades_df['pnl'].median()
        
        # Risk metrics
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        profit_factor = abs(
            trades_df[trades_df['pnl'] > 0]['pnl'].sum() / 
            trades_df[trades_df['pnl'] <= 0]['pnl'].sum()
        ) if trades_df[trades_df['pnl'] <= 0]['pnl'].sum() != 0 else float('inf')
        
        # Drawdown
        if not results['drawdowns'].empty:
            max_drawdown = results['drawdowns']['drawdown'].min()
        else:
            max_drawdown = 0
        
        # Time metrics
        avg_hold_time = trades_df['hold_time_minutes'].mean()
        
        # Calculate daily returns for Sharpe ratio
        if not results['equity_curve'].empty:
            equity_df = results['equity_curve']
            equity_df['pct_change'] = equity_df['equity'].pct_change().fillna(0)
            
            # Calculate Sharpe ratio (annualized)
            daily_returns = equity_df['pct_change']
            sharpe_ratio = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Analyze exit reasons
        exit_reasons = trades_df['exit_reason'].value_counts().to_dict()
        
        # Analyze trades by market direction
        direction_results = {}
        for direction in [-1, 0, 1]:
            dir_trades = trades_df[trades_df['market_direction'] == direction]
            if len(dir_trades) > 0:
                direction_results[direction] = {
                    'count': len(dir_trades),
                    'win_rate': len(dir_trades[dir_trades['pnl'] > 0]) / len(dir_trades),
                    'avg_return': dir_trades['pnl'].mean()
                }
        
        # Store all metrics
        results['metrics'] = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'return_pct': return_pct,
            'avg_return': avg_return,
            'median_return': median_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': win_loss_ratio,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'avg_hold_time': avg_hold_time,
            'exit_reasons': exit_reasons,
            'direction_results': direction_results,
            'final_capital': self.current_capital
        }
        
        # Print summary
        print("\nStrategy Performance Summary:")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Total Return: {total_return:,.2f} ({return_pct:.2f}%)")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Initial Capital: {self.initial_capital:,.2f}")
        print(f"Final Capital: {self.current_capital:,.2f}")
        
    def visualize_results(self, results: Dict) -> None:
        """Generate visualizations of backtest results."""
        if not results['trades']:
            print("No trades to visualize.")
            return
            
        # Create directory for charts
        os.makedirs('charts', exist_ok=True)
        
        # 1. Equity Curve
        plt.figure(figsize=(12, 6))
        
        if not results['equity_curve'].empty:
            plt.plot(results['equity_curve']['date'], results['equity_curve']['equity'], 
                    marker='o', linestyle='-', linewidth=2, color='blue')
            
            # Add horizontal line for initial capital
            plt.axhline(y=self.initial_capital, color='gray', linestyle='--', linewidth=1, 
                      label=f"Initial Capital: ${self.initial_capital:,.0f}")
            
            plt.title('Equity Curve', fontsize=16)
            plt.xlabel('Date')
            plt.ylabel('Equity ($)')
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            
            plt.savefig('charts/equity_curve_optimized.png')
            
        # 2. Drawdown
        plt.figure(figsize=(12, 6))
        
        if not results['drawdowns'].empty:
            plt.plot(results['drawdowns']['date'], results['drawdowns']['drawdown'], 
                    linestyle='-', linewidth=2, color='red')
            
            plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
            
            plt.title('Drawdown Over Time', fontsize=16)
            plt.xlabel('Date')
            plt.ylabel('Drawdown (%)')
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.tight_layout()
            
            plt.savefig('charts/drawdown_optimized.png')
        
        # 3. Trade P&L Distribution
        plt.figure(figsize=(10, 6))
        
        trades_df = pd.DataFrame(results['trades'])
        sns.histplot(trades_df['pnl'], bins=20, kde=True, color='skyblue')
        
        plt.axvline(x=0, color='red', linestyle='--', linewidth=1)
        plt.title('P&L Distribution', fontsize=16)
        plt.xlabel('P&L ($)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig('charts/pnl_distribution_optimized.png')
        
        # 4. Win Rate by Market Direction
        plt.figure(figsize=(10, 6))
        
        direction_labels = {-1: 'Downtrend', 0: 'Sideways', 1: 'Uptrend'}
        
        directions = []
        win_rates = []
        counts = []
        
        for direction, stats in results['metrics'].get('direction_results', {}).items():
            directions.append(direction_labels.get(direction, str(direction)))
            win_rates.append(stats['win_rate'] * 100)
            counts.append(stats['count'])
        
        if directions:
            bars = plt.bar(directions, win_rates, color='skyblue')
            
            # Add trade count annotations
            for i, (bar, count) in enumerate(zip(bars, counts)):
                plt.text(bar.get_x() + bar.get_width()/2, 5, 
                        f'n={count}', ha='center', va='bottom', 
                        fontweight='bold', color='black')
            
            plt.title('Win Rate by Market Direction', fontsize=16)
            plt.xlabel('Market Direction')
            plt.ylabel('Win Rate (%)')
            plt.axhline(y=50, color='red', linestyle='--', linewidth=1, label='50% Win Rate')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plt.savefig('charts/win_rate_by_direction.png')
        
        # 5. Performance Summary Table
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')
        
        metrics = results['metrics']
        
        # Top section - Key Metrics
        ax.text(0.5, 0.95, 'Adam Mancini Strategy - Performance Summary', 
                horizontalalignment='center', verticalalignment='center',
                fontsize=16, fontweight='bold')
        
        # Create a table with key metrics
        key_metrics = [
            ['Total Trades', f"{metrics['total_trades']}"],
            ['Win Rate', f"{metrics['win_rate']:.2%}"],
            ['Total Return', f"${metrics['total_return']:,.2f} ({metrics['return_pct']:.2f}%)"],
            ['Profit Factor', f"{metrics['profit_factor']:.2f}"],
            ['Win/Loss Ratio', f"{metrics['win_loss_ratio']:.2f}"],
            ['Max Drawdown', f"{metrics['max_drawdown']:.2f}%"],
            ['Sharpe Ratio', f"{metrics['sharpe_ratio']:.2f}"],
            ['Avg Trade P&L', f"${metrics['avg_return']:,.2f}"],
            ['Avg Hold Time', f"{metrics['avg_hold_time']:.2f} minutes"],
            ['Initial Capital', f"${self.initial_capital:,.2f}"],
            ['Final Capital', f"${metrics['final_capital']:,.2f}"]
        ]
        
        table = ax.table(cellText=key_metrics, colLabels=['Metric', 'Value'],
                        loc='center', cellLoc='left', colWidths=[0.5, 0.3])
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)
        
        plt.tight_layout()
        plt.savefig('charts/performance_summary_optimized.png')
        
        print("\nVisualization complete. Optimized charts saved to 'charts' directory.")

def generate_synthetic_data(days=30, bars_per_day=78, volatility=0.01, trend=0.001):
    """
    Generate synthetic price data for testing.
    
    Parameters
    ----------
    days : int
        Number of trading days to generate
    bars_per_day : int
        Number of 5-minute bars per day (78 for full trading day)
    volatility : float
        Daily volatility (standard deviation of returns)
    trend : float
        Daily trend factor (positive for uptrend, negative for downtrend)
        
    Returns
    -------
    pd.DataFrame
        Synthetic OHLC data
    """
    print(f"Generating {days} days of synthetic 5-minute data...")
    
    # Base price and volatility
    base_price = 24000  # Starting price (e.g., NIFTY level)
    bar_volatility = volatility / np.sqrt(bars_per_day)
    
    # Define market open and close times
    market_open = datetime.now().replace(hour=9, minute=15, second=0, microsecond=0)
    
    # Generate timestamps
    timestamps = []
    for day in range(days):
        day_date = market_open + timedelta(days=day)
        
        # Skip weekends
        if day_date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            continue
            
        for bar in range(bars_per_day):
            bar_time = day_date + timedelta(minutes=5*bar)
            timestamps.append(bar_time)
    
    # Generate price series with trend and volatility
    np.random.seed(42)  # For reproducibility
    
    price_changes = np.random.normal(trend, bar_volatility, len(timestamps))
    log_returns = np.cumsum(price_changes)
    prices = base_price * np.exp(log_returns)
    
    # Add some realistic market behavior
    # 1. Opening gaps
    day_indices = [i for i, ts in enumerate(timestamps) if ts.time() == market_open.time()]
    for idx in day_indices[1:]:  # Skip first day
        gap_size = np.random.normal(0, volatility * 2)
        prices[idx:] = prices[idx:] * np.exp(gap_size)
    
    # 2. Intraday patterns - more volatility at open and close
    for i, ts in enumerate(timestamps):
        time_factor = 1.0
        
        # Higher volatility in first and last hour
        minutes_since_open = (ts.hour - 9) * 60 + (ts.minute - 15)
        minutes_to_close = (15 * 60 + 30) - (ts.hour * 60 + ts.minute)
        
        if minutes_since_open < 60:  # First hour
            time_factor = 1.5
        elif minutes_to_close < 60:  # Last hour
            time_factor = 1.3
            
        # Add some random noise scaled by time factor
        prices[i] *= np.exp(np.random.normal(0, bar_volatility * time_factor))
    
    # Create OHLC data
    data = []
    
    for i, (ts, close) in enumerate(zip(timestamps, prices)):
        # Determine if this is the start of a new day
        is_day_start = i in day_indices
        
        if is_day_start:
            # For the first bar of the day, open is yesterday's close with a gap
            if i > 0:
                prev_close = prices[i-1]
                gap = close / prev_close - 1
                open_price = prev_close * (1 + gap)
            else:
                open_price = close * (1 - np.random.normal(0, bar_volatility))
        else:
            # For other bars, open is previous close
            open_price = prices[i-1]
        
        # Generate high and low with some randomness
        price_range = abs(close - open_price)
        extra_range = max(price_range * 0.5, base_price * bar_volatility)
        
        if close > open_price:
            high = close + np.random.uniform(0, extra_range)
            low = open_price - np.random.uniform(0, extra_range)
        else:
            high = open_price + np.random.uniform(0, extra_range)
            low = close - np.random.uniform(0, extra_range)
            
        # Ensure high >= max(open, close) and low <= min(open, close)
        high = max(high, max(open_price, close))
        low = min(low, min(open_price, close))
        
        data.append({
            'datetime': ts,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.randint(100, 1000)  # Dummy volume
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df.set_index('datetime', inplace=True)
    
    print(f"Generated {len(df)} bars of synthetic data")
    return df

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

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Adam Mancini strategy specialized analysis")
    
    parser.add_argument(
        "--data-file", 
        type=str, 
        default="data/nifty_5min_30days.csv",
        help="Path to CSV file with historical data"
    )
    parser.add_argument(
        "--synthetic-data", 
        action="store_true",
        help="Use synthetic data instead of historical data"
    )
    parser.add_argument(
        "--open-range-minutes", 
        type=int, 
        default=6,
        help="Number of minutes for the opening range (default: 6)"
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
    parser.add_argument(
        "--risk-per-trade", 
        type=float, 
        default=1.0,
        help="Risk per trade as percentage of capital (default: 1.0)"
    )
    parser.add_argument(
        "--initial-capital", 
        type=float, 
        default=100000,
        help="Initial capital for the backtest (default: 100000)"
    )
    parser.add_argument(
        "--no-filter", 
        action="store_true",
        help="Disable volatility and market direction filters"
    )
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('charts', exist_ok=True)
    
    # Get input data
    if args.synthetic_data:
        data = generate_synthetic_data(days=60, volatility=0.01, trend=0.0005)
        data_file = "data/synthetic_nifty_5min_60days.csv"
        
        # Save synthetic data
        data.to_csv(data_file)
        print(f"Saved synthetic data to {data_file}")
    else:
        # Load historical data
        data = load_and_prepare_data(args.data_file)
    
    # Initialize and run strategy
    strategy = ManciniOptimized(
        opening_range_minutes=args.open_range_minutes,
        take_profit_r_level=args.take_profit_level,
        stop_loss_pct=args.stop_loss_pct,
        risk_per_trade_pct=args.risk_per_trade,
        min_volatility_filter=not args.no_filter,
        check_market_direction=not args.no_filter,
        initial_capital=args.initial_capital
    )
    
    # Run backtest
    backtest_results = strategy.backtest(data)