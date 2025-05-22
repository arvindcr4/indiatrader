#!/usr/bin/env python3
"""
Adam Mancini Key Level Breakdown Trading Strategy Implementation

This strategy focuses on breakdowns below key support levels, rather than
opening range breakouts. It identifies key price levels and trades breakdowns
below these levels with appropriate risk management.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Set plotting style
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

class ManciniKeyLevelStrategy:
    """
    Adam Mancini Key Level Breakdown Strategy implementation.
    """
    
    def __init__(
        self,
        support_level_lookback=5,
        confirmation_candles=2,
        stop_loss_pct=1.0,
        take_profit_r_ratio=2.0,
        risk_per_trade_pct=1.0,
        check_market_direction=True,
        initial_capital=100000
    ):
        """
        Initialize the strategy with the given parameters.
        
        Parameters
        ----------
        support_level_lookback : int
            Number of bars to look back for identifying support levels
        confirmation_candles : int
            Number of candles needed to confirm a breakdown
        stop_loss_pct : float
            Stop loss as percentage of entry price
        take_profit_r_ratio : float
            Take profit as a ratio of risk (2.0 = 2:1 reward-to-risk)
        risk_per_trade_pct : float
            Risk per trade as percentage of capital
        check_market_direction : bool
            Whether to check for market direction alignment
        initial_capital : float
            Initial capital for backtesting
        """
        self.support_level_lookback = support_level_lookback
        self.confirmation_candles = confirmation_candles
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_r_ratio = take_profit_r_ratio
        self.risk_per_trade_pct = risk_per_trade_pct
        self.check_market_direction = check_market_direction
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
    
    def identify_key_levels(self, data, window=20):
        """
        Identify key support and resistance levels.
        
        Parameters
        ----------
        data : pd.DataFrame
            Historical price data
        window : int
            Window size for finding local minima/maxima
            
        Returns
        -------
        dict
            Dictionary of key support and resistance levels
        """
        # Initialize containers
        support_levels = []
        resistance_levels = []
        
        # Find local minima (support) and maxima (resistance)
        for i in range(window, len(data) - window):
            # Check if this candle is a local minimum
            if (data['low'].iloc[i] <= data['low'].iloc[i-window:i].min() and
                data['low'].iloc[i] <= data['low'].iloc[i+1:i+window+1].min()):
                support_levels.append((data.index[i], data['low'].iloc[i]))
            
            # Check if this candle is a local maximum
            if (data['high'].iloc[i] >= data['high'].iloc[i-window:i].max() and
                data['high'].iloc[i] >= data['high'].iloc[i+1:i+window+1].max()):
                resistance_levels.append((data.index[i], data['high'].iloc[i]))
        
        # Return as dictionary
        return {
            'support': support_levels,
            'resistance': resistance_levels
        }
    
    def find_recent_support(self, data, current_idx, lookback=5):
        """
        Find the most recent support level.
        
        Parameters
        ----------
        data : pd.DataFrame
            Price data
        current_idx : int
            Current index in the data
        lookback : int
            Number of bars to look back
            
        Returns
        -------
        float
            Recent support level price
        """
        if current_idx < lookback:
            return None
        
        # Get the lookback window
        lookback_data = data.iloc[current_idx-lookback:current_idx]
        
        # Find the lowest low in the lookback period
        return lookback_data['low'].min()
    
    def determine_market_direction(self, data, window=20):
        """Determine market direction using moving averages."""
        # Calculate short and long-term moving averages
        short_ma = data['close'].rolling(window=window//2).mean()
        long_ma = data['close'].rolling(window=window).mean()
        
        # Determine direction
        direction = np.where(short_ma > long_ma, 1, np.where(short_ma < long_ma, -1, 0))
        
        return pd.Series(direction, index=data.index)
    
    def calculate_position_size(self, entry_price, stop_price):
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
            Backtest results
        """
        print(f"Running Mancini Key Level Breakdown backtest...")
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Add date column for grouping
        data = data.copy()
        data['date'] = data.index.date
        
        # Calculate market direction
        if self.check_market_direction:
            data['market_direction'] = self.determine_market_direction(data)
        else:
            data['market_direction'] = 0  # Neutral if not checking
        
        # Dictionary to store backtest results
        results = {
            'daily_pnl': {},
            'trades': [],
            'equity_curve': [],
            'metrics': {},
            'key_levels': []
        }
        
        # Initialize position tracking
        position = 0  # 0: flat, -1: short
        entry_price = 0
        entry_time = None
        stop_loss = 0
        take_profit = 0
        position_size = 0
        trade_id = 0
        daily_pnl = {}
        
        # Process data
        for i in range(len(data)):
            current_row = data.iloc[i]
            current_idx = i
            current_date_str = current_row['date'].strftime("%Y-%m-%d")
            
            # Initialize daily PnL if not existing
            if current_date_str not in daily_pnl:
                daily_pnl[current_date_str] = 0
            
            # Check if we have an open position that needs management
            if position != 0:
                # Calculate current P&L
                if position == -1:  # Short position
                    unrealized_pnl = (entry_price - current_row['close']) * position_size
                    
                    # Check stop loss
                    if current_row['high'] >= stop_loss:
                        # Stop loss hit
                        exit_price = stop_loss  # Use stop loss price for exit
                        trade_pnl = (entry_price - exit_price) * position_size
                        exit_reason = "Stop Loss"
                        
                        # Update capital
                        self.current_capital += trade_pnl
                        
                        # Record trade
                        trade = {
                            'id': trade_id,
                            'entry_time': entry_time,
                            'entry_price': entry_price,
                            'exit_time': current_row.name,
                            'exit_price': exit_price,
                            'position_size': position_size,
                            'pnl': trade_pnl,
                            'position': 'SHORT',
                            'exit_reason': exit_reason,
                            'market_direction': current_row['market_direction']
                        }
                        results['trades'].append(trade)
                        daily_pnl[current_date_str] += trade_pnl
                        
                        # Reset position
                        position = 0
                        trade_id += 1
                        
                    # Check take profit
                    elif current_row['low'] <= take_profit:
                        # Take profit hit
                        exit_price = take_profit  # Use take profit price for exit
                        trade_pnl = (entry_price - exit_price) * position_size
                        exit_reason = "Take Profit"
                        
                        # Update capital
                        self.current_capital += trade_pnl
                        
                        # Record trade
                        trade = {
                            'id': trade_id,
                            'entry_time': entry_time,
                            'entry_price': entry_price,
                            'exit_time': current_row.name,
                            'exit_price': exit_price,
                            'position_size': position_size,
                            'pnl': trade_pnl,
                            'position': 'SHORT',
                            'exit_reason': exit_reason,
                            'market_direction': current_row['market_direction']
                        }
                        results['trades'].append(trade)
                        daily_pnl[current_date_str] += trade_pnl
                        
                        # Reset position
                        position = 0
                        trade_id += 1
            
            # Check for entry signals if we don't have a position
            if position == 0 and current_idx >= self.support_level_lookback:
                # Find recent support level
                support_level = self.find_recent_support(data, current_idx, self.support_level_lookback)
                
                if support_level is not None:
                    # Check for breakdown below support
                    breakdown = False
                    
                    # To confirm breakdown, we need X consecutive candles closing below support
                    if current_idx >= self.confirmation_candles:
                        breakdown = True
                        for j in range(self.confirmation_candles):
                            if data.iloc[current_idx-j]['close'] > support_level:
                                breakdown = False
                                break
                    
                    # Generate short signal
                    short_signal = breakdown
                    
                    # Check market direction if enabled
                    if self.check_market_direction and short_signal:
                        # Only short in downtrends or neutral markets
                        if current_row['market_direction'] > 0:
                            short_signal = False
                    
                    if short_signal:
                        # Short entry signal
                        position = -1
                        entry_price = current_row['close']
                        entry_time = current_row.name
                        
                        # Add key level to results
                        results['key_levels'].append({
                            'time': current_row.name,
                            'price': support_level,
                            'type': 'support_breakdown'
                        })
                        
                        # Set stop loss and take profit
                        stop_loss = entry_price * (1 + self.stop_loss_pct / 100)
                        take_profit = entry_price - (stop_loss - entry_price) * self.take_profit_r_ratio
                        
                        # Calculate position size based on risk
                        position_size = self.calculate_position_size(entry_price, stop_loss)
        
        # Close any open position at the end of the data
        if position != 0:
            exit_price = data.iloc[-1]['close']
            
            if position == -1:  # Short position
                trade_pnl = (entry_price - exit_price) * position_size
            
            # Update capital
            self.current_capital += trade_pnl
            
            # Record trade
            trade = {
                'id': trade_id,
                'entry_time': entry_time,
                'entry_price': entry_price,
                'exit_time': data.index[-1],
                'exit_price': exit_price,
                'position_size': position_size,
                'pnl': trade_pnl,
                'position': 'SHORT' if position == -1 else 'LONG',
                'exit_reason': "End of Data",
                'market_direction': data.iloc[-1]['market_direction']
            }
            results['trades'].append(trade)
            
            # Update daily PnL
            date_str = data.iloc[-1]['date'].strftime("%Y-%m-%d")
            if date_str not in daily_pnl:
                daily_pnl[date_str] = 0
            daily_pnl[date_str] += trade_pnl
        
        # Store daily PnL
        results['daily_pnl'] = daily_pnl
        
        # Calculate equity curve
        equity = self.initial_capital
        equity_curve = []
        
        for day, pnl in daily_pnl.items():
            equity += pnl
            equity_curve.append({'date': day, 'equity': equity})
            
        results['equity_curve'] = pd.DataFrame(equity_curve)
        
        # Calculate metrics
        self.calculate_metrics(results)
        
        # Visualize results if needed
        if visualize:
            self.visualize_results(results)
        
        return results
    
    def calculate_metrics(self, results):
        """Calculate performance metrics."""
        if not results['trades']:
            results['metrics'] = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_return': 0,
                'return_pct': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }
            return
        
        # Convert trades to DataFrame for analysis
        trades_df = pd.DataFrame(results['trades'])
        
        # Basic statistics
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
        
        # Drawdown calculation
        if not results['equity_curve'].empty:
            equity_data = results['equity_curve']['equity']
            running_max = equity_data.cummax()
            drawdown = (equity_data - running_max) / running_max * 100
            max_drawdown = drawdown.min()
        else:
            max_drawdown = 0
            
        # Sharpe ratio (simplified)
        if not results['equity_curve'].empty:
            equity_df = results['equity_curve']
            equity_df['pct_change'] = equity_df['equity'].pct_change().fillna(0)
            
            daily_returns = equity_df['pct_change']
            sharpe_ratio = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Exit reasons
        exit_reasons = trades_df['exit_reason'].value_counts().to_dict()
        
        # Direction analysis
        direction_results = {}
        for direction in [-1, 0, 1]:
            dir_trades = trades_df[trades_df['market_direction'] == direction]
            if len(dir_trades) > 0:
                direction_results[direction] = {
                    'count': len(dir_trades),
                    'win_rate': len(dir_trades[dir_trades['pnl'] > 0]) / len(dir_trades),
                    'avg_return': dir_trades['pnl'].mean()
                }
        
        # Store metrics
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
            'exit_reasons': exit_reasons,
            'direction_results': direction_results,
            'final_capital': self.current_capital
        }
        
        # Print summary
        print("\nStrategy Performance Summary:")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Total Return: ${total_return:,.2f} ({return_pct:.2f}%)")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Capital: ${self.current_capital:,.2f}")
    
    def visualize_results(self, results):
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
            
            plt.savefig('charts/key_level_equity_curve.png')
        
        # 2. Daily P&L
        plt.figure(figsize=(12, 6))
        
        days = []
        pnls = []
        
        for day, pnl in results['daily_pnl'].items():
            if pnl != 0:  # Only include days with trades
                days.append(day)
                pnls.append(pnl)
        
        if days:
            colors = ['green' if p > 0 else 'red' for p in pnls]
            plt.bar(days, pnls, color=colors)
            
            plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            plt.title('Daily P&L', fontsize=16)
            plt.xlabel('Date')
            plt.ylabel('P&L ($)')
            plt.xticks(rotation=45)
            plt.grid(axis='y')
            plt.tight_layout()
            
            plt.savefig('charts/key_level_daily_pnl.png')
        
        # 3. Trade P&L Distribution
        plt.figure(figsize=(10, 6))
        
        trades_df = pd.DataFrame(results['trades'])
        if len(trades_df) > 0:
            sns.histplot(trades_df['pnl'], bins=20, kde=True, color='skyblue')
            
            plt.axvline(x=0, color='red', linestyle='--', linewidth=1)
            plt.title('P&L Distribution', fontsize=16)
            plt.xlabel('P&L ($)')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plt.savefig('charts/key_level_pnl_distribution.png')
        
        # 4. Performance Summary Table
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')
        
        metrics = results['metrics']
        
        # Create table with metrics
        data = [
            ['Total Trades', f"{metrics['total_trades']}"],
            ['Win Rate', f"{metrics['win_rate']:.2%}"],
            ['Total Return', f"${metrics['total_return']:,.2f} ({metrics['return_pct']:.2f}%)"],
            ['Average Trade', f"${metrics['avg_return']:,.2f}"],
            ['Average Win', f"${metrics['avg_win']:,.2f}"],
            ['Average Loss', f"${metrics['avg_loss']:,.2f}"],
            ['Win/Loss Ratio', f"{metrics['win_loss_ratio']:.2f}"],
            ['Profit Factor', f"{metrics['profit_factor']:.2f}"],
            ['Max Drawdown', f"{metrics['max_drawdown']:.2f}%"],
            ['Sharpe Ratio', f"{metrics['sharpe_ratio']:.2f}"],
            ['Initial Capital', f"${self.initial_capital:,.2f}"],
            ['Final Capital', f"${metrics['final_capital']:,.2f}"]
        ]
        
        table = ax.table(cellText=data, colLabels=['Metric', 'Value'],
                        loc='center', cellLoc='left', colWidths=[0.4, 0.3])
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)
        
        plt.tight_layout()
        plt.savefig('charts/key_level_performance_summary.png')
        
        print("\nVisualization complete. Charts saved to 'charts' directory.")

def load_data(data_file):
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

def create_final_report(strategy, results, params=None):
    """Create a comprehensive final report."""
    # Create a report file
    report_path = 'mancini_key_level_strategy_report.md'
    
    with open(report_path, 'w') as f:
        f.write("# Adam Mancini Key Level Breakdown Strategy - Final Report\n\n")
        
        f.write("## Strategy Configuration\n\n")
        f.write(f"- **Support Level Lookback**: {strategy.support_level_lookback} bars\n")
        f.write(f"- **Confirmation Candles**: {strategy.confirmation_candles}\n")
        f.write(f"- **Stop Loss**: {strategy.stop_loss_pct}% of Entry Price\n")
        f.write(f"- **Take Profit R-Ratio**: {strategy.take_profit_r_ratio}:1\n")
        f.write(f"- **Risk Per Trade**: {strategy.risk_per_trade_pct}% of Capital\n")
        f.write(f"- **Initial Capital**: ${strategy.initial_capital:,.2f}\n")
        f.write(f"- **Market Direction Filter**: {'Enabled' if strategy.check_market_direction else 'Disabled'}\n\n")
        
        metrics = results['metrics']
        
        f.write("## Performance Summary\n\n")
        f.write(f"- **Total Trades**: {metrics['total_trades']}\n")
        f.write(f"- **Winning Trades**: {metrics['winning_trades']} ({metrics['win_rate']:.2%})\n")
        f.write(f"- **Losing Trades**: {metrics['losing_trades']} ({1-metrics['win_rate']:.2%})\n")
        f.write(f"- **Total Return**: ${metrics['total_return']:,.2f} ({metrics['return_pct']:.2f}%)\n")
        f.write(f"- **Profit Factor**: {metrics['profit_factor']:.2f}\n")
        f.write(f"- **Win/Loss Ratio**: {metrics['win_loss_ratio']:.2f}\n")
        f.write(f"- **Max Drawdown**: {metrics['max_drawdown']:.2f}%\n")
        f.write(f"- **Sharpe Ratio**: {metrics['sharpe_ratio']:.2f}\n")
        f.write(f"- **Average Trade P&L**: ${metrics['avg_return']:,.2f}\n")
        f.write(f"- **Average Win**: ${metrics['avg_win']:,.2f}\n")
        f.write(f"- **Average Loss**: ${metrics['avg_loss']:,.2f}\n")
        f.write(f"- **Final Capital**: ${metrics['final_capital']:,.2f}\n\n")
        
        # Add exit reasons
        f.write("## Exit Reason Analysis\n\n")
        f.write("| Exit Reason | Count | Percentage |\n")
        f.write("|------------|-------|------------|\n")
        
        for reason, count in metrics.get('exit_reasons', {}).items():
            percentage = count / metrics['total_trades'] * 100 if metrics['total_trades'] > 0 else 0
            f.write(f"| {reason} | {count} | {percentage:.2f}% |\n")
        
        # Add market direction analysis
        f.write("\n## Market Direction Analysis\n\n")
        f.write("| Market Direction | Trades | Win Rate | Avg Return |\n")
        f.write("|------------------|--------|----------|------------|\n")
        
        direction_labels = {-1: 'Downtrend', 0: 'Sideways', 1: 'Uptrend'}
        
        for direction, stats in metrics.get('direction_results', {}).items():
            f.write(f"| {direction_labels.get(direction, str(direction))} | {stats['count']} | {stats['win_rate']:.2%} | ${stats['avg_return']:,.2f} |\n")
        
        # Add recommendations
        f.write("\n## Implementation Recommendations\n\n")
        
        if metrics['win_rate'] > 0.5:
            f.write("- The strategy shows a positive win rate, indicating good trade selection.\n")
        else:
            f.write("- The lower win rate suggests focusing on proper position sizing and risk management.\n")
        
        if metrics['profit_factor'] > 1.5:
            f.write("- With a strong profit factor, this strategy demonstrates good risk/reward characteristics.\n")
        elif metrics['profit_factor'] > 1:
            f.write("- The profit factor is positive but moderate, suggesting room for optimization.\n")
        else:
            f.write("- The low profit factor indicates the need for further refinement or different market conditions.\n")
        
        # Add conclusion based on performance
        f.write("\n## Conclusion\n\n")
        
        if metrics['return_pct'] > 0 and metrics['win_rate'] > 0.5:
            f.write("The Adam Mancini key level breakdown strategy has demonstrated positive results in the tested period. ")
            f.write("With proper risk management and continued monitoring, this strategy shows promise ")
            f.write("for implementation in live trading environments.\n\n")
        else:
            f.write("The Adam Mancini key level breakdown strategy shows mixed results in the tested period. ")
            f.write("Further optimization and testing in different market conditions is recommended ")
            f.write("before implementing in a live trading environment.\n\n")
        
        f.write("### Next Steps\n\n")
        f.write("1. **Paper Trading**: Test the strategy in a simulated environment with real-time data\n")
        f.write("2. **Parameter Refinement**: Continue to optimize parameters based on market conditions\n")
        f.write("3. **Risk Management**: Implement strict risk controls to protect capital\n")
        f.write("4. **Diversification**: Consider using this strategy as part of a broader trading approach\n")
    
    print(f"Final report saved to {report_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Adam Mancini Key Level Breakdown strategy")
    
    parser.add_argument(
        "--data-file", 
        type=str, 
        default="data/nifty_5min_30days.csv",
        help="Path to CSV file with historical data"
    )
    parser.add_argument(
        "--lookback", 
        type=int, 
        default=5,
        help="Number of bars to look back for support levels (default: 5)"
    )
    parser.add_argument(
        "--confirmation", 
        type=int, 
        default=2,
        help="Number of candles needed to confirm breakdown (default: 2)"
    )
    parser.add_argument(
        "--stop-loss", 
        type=float, 
        default=1.0,
        help="Stop loss as percentage of entry price (default: 1.0)"
    )
    parser.add_argument(
        "--take-profit-ratio", 
        type=float, 
        default=2.0,
        help="Take profit as ratio of risk (default: 2.0)"
    )
    parser.add_argument(
        "--risk-per-trade", 
        type=float, 
        default=1.0,
        help="Risk per trade as percentage of capital (default: 1.0)"
    )
    parser.add_argument(
        "--no-direction-filter", 
        action="store_true",
        help="Disable market direction filter"
    )
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('charts', exist_ok=True)
    
    # Load data
    data = load_data(args.data_file)
    
    # Initialize strategy
    strategy = ManciniKeyLevelStrategy(
        support_level_lookback=args.lookback,
        confirmation_candles=args.confirmation,
        stop_loss_pct=args.stop_loss,
        take_profit_r_ratio=args.take_profit_ratio,
        risk_per_trade_pct=args.risk_per_trade,
        check_market_direction=not args.no_direction_filter
    )
    
    # Run backtest
    results = strategy.backtest(data)
    
    # Create final report
    create_final_report(strategy, results)