#!/usr/bin/env python3
"""
Final analysis of Adam Mancini trading strategy with realistic synthetic data.
This script will:
1. Generate realistic market data with true breakout patterns
2. Test various parameter combinations
3. Produce a comprehensive final report
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

# Import our optimized strategy implementation
from mancini_specialized_analysis import ManciniOptimized

def generate_realistic_market_data(days=60, bars_per_day=78, base_price=24000):
    """
    Generate realistic synthetic market data with intentional breakout patterns.
    
    Parameters
    ----------
    days : int
        Number of trading days to generate
    bars_per_day : int
        Number of 5-minute bars per day (78 for full trading day)
    base_price : float
        Starting price level
        
    Returns
    -------
    pd.DataFrame
        Synthetic OHLC data with breakout patterns
    """
    print(f"Generating {days} days of realistic market data with breakout patterns...")
    
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
    
    # Create data structure
    data = []
    
    # Track current price level
    current_price = base_price
    
    # Define common market patterns
    patterns = [
        'trend_up',
        'trend_down',
        'consolidation',
        'breakout_up',
        'breakout_down',
        'failed_breakout_up',
        'failed_breakout_down',
        'gap_up',
        'gap_down',
        'choppy'
    ]
    
    # Process each day with intentional patterns
    day_indices = [i for i, ts in enumerate(timestamps) if ts.time() == market_open.time()]
    
    for day_idx in range(len(day_indices)):
        # Determine this day's pattern and volatility
        pattern = np.random.choice(patterns)
        volatility = np.random.uniform(0.005, 0.015)  # 0.5% to 1.5% daily volatility
        
        # Day start and end indices
        start_idx = day_indices[day_idx]
        end_idx = day_indices[day_idx + 1] if day_idx < len(day_indices) - 1 else len(timestamps)
        day_bars = end_idx - start_idx
        
        # Gap open (from previous close)
        if day_idx > 0:
            previous_close = data[start_idx - 1]['close']
            
            if pattern == 'gap_up':
                gap_size = np.random.uniform(0.005, 0.015)  # 0.5% to 1.5% gap
                current_price = previous_close * (1 + gap_size)
            elif pattern == 'gap_down':
                gap_size = np.random.uniform(0.005, 0.015)  # 0.5% to 1.5% gap
                current_price = previous_close * (1 - gap_size)
            else:
                gap_size = np.random.normal(0, 0.005)  # Small random gap
                current_price = previous_close * (1 + gap_size)
        
        # First bar of the day
        first_open = current_price
        
        # Create opening range (first 6 bars / 30 minutes)
        opening_range_size = volatility * current_price
        opening_range_high = first_open * (1 + np.random.uniform(0, opening_range_size))
        opening_range_low = first_open * (1 - np.random.uniform(0, opening_range_size))
        
        # Generate bars for this day based on the pattern
        for i in range(start_idx, end_idx):
            bar_num = i - start_idx
            minutes_since_open = bar_num * 5
            
            if bar_num == 0:
                # First bar of the day
                open_price = first_open
                bar_volatility = volatility / np.sqrt(day_bars) * 1.5  # Higher volatility at open
                
                if np.random.random() > 0.5:
                    close_price = open_price * (1 + np.random.uniform(0, bar_volatility))
                else:
                    close_price = open_price * (1 - np.random.uniform(0, bar_volatility))
                    
                high_price = max(open_price, close_price) * (1 + np.random.uniform(0, bar_volatility/2))
                low_price = min(open_price, close_price) * (1 - np.random.uniform(0, bar_volatility/2))
                
                # Update opening range
                opening_range_high = max(opening_range_high, high_price)
                opening_range_low = min(opening_range_low, low_price)
                
            elif bar_num < 6:
                # Part of opening range
                open_price = data[i-1]['close']
                bar_volatility = volatility / np.sqrt(day_bars) * 1.3  # Higher volatility during opening range
                
                if np.random.random() > 0.5:
                    close_price = open_price * (1 + np.random.uniform(0, bar_volatility))
                else:
                    close_price = open_price * (1 - np.random.uniform(0, bar_volatility))
                    
                high_price = max(open_price, close_price) * (1 + np.random.uniform(0, bar_volatility/2))
                low_price = min(open_price, close_price) * (1 - np.random.uniform(0, bar_volatility/2))
                
                # Update opening range
                opening_range_high = max(opening_range_high, high_price)
                opening_range_low = min(opening_range_low, low_price)
                
            else:
                # After opening range - implement pattern
                open_price = data[i-1]['close']
                bar_volatility = volatility / np.sqrt(day_bars)
                
                # Adjust volatility based on time of day
                if minutes_since_open > 360:  # Last hour
                    bar_volatility *= 1.2
                
                # Implement patterns
                if pattern == 'trend_up':
                    # Gradual uptrend
                    trend_factor = 0.0003 * (1 + 0.2 * np.sin(bar_num / 10))
                    close_price = open_price * (1 + trend_factor + np.random.normal(0, bar_volatility))
                
                elif pattern == 'trend_down':
                    # Gradual downtrend
                    trend_factor = 0.0003 * (1 + 0.2 * np.sin(bar_num / 10))
                    close_price = open_price * (1 - trend_factor + np.random.normal(0, bar_volatility))
                
                elif pattern == 'consolidation':
                    # Sideways movement
                    close_price = open_price * (1 + np.random.normal(0, bar_volatility * 0.6))
                
                elif pattern == 'breakout_up' and bar_num == 7:
                    # Upside breakout after opening range
                    breakout_size = (opening_range_high - opening_range_low) * 0.3
                    close_price = opening_range_high * (1 + np.random.uniform(0.001, 0.003))
                    
                    # Ensure it's a clear breakout
                    close_price = max(close_price, opening_range_high * 1.001)
                
                elif pattern == 'breakout_down' and bar_num == 7:
                    # Downside breakout after opening range
                    breakout_size = (opening_range_high - opening_range_low) * 0.3
                    close_price = opening_range_low * (1 - np.random.uniform(0.001, 0.003))
                    
                    # Ensure it's a clear breakout
                    close_price = min(close_price, opening_range_low * 0.999)
                
                elif pattern == 'failed_breakout_up' and bar_num == 7:
                    # Failed upside breakout
                    close_price = opening_range_high * (1 + np.random.uniform(0.0005, 0.001))
                elif pattern == 'failed_breakout_up' and bar_num > 7 and bar_num < 12:
                    # Price falls back into range
                    open_price = data[i-1]['close']
                    reversal_strength = np.random.uniform(0.001, 0.002)
                    close_price = open_price * (1 - reversal_strength)
                
                elif pattern == 'failed_breakout_down' and bar_num == 7:
                    # Failed downside breakout
                    close_price = opening_range_low * (1 - np.random.uniform(0.0005, 0.001))
                elif pattern == 'failed_breakout_down' and bar_num > 7 and bar_num < 12:
                    # Price rises back into range
                    open_price = data[i-1]['close']
                    reversal_strength = np.random.uniform(0.001, 0.002)
                    close_price = open_price * (1 + reversal_strength)
                
                elif pattern == 'choppy':
                    # Highly volatile sideways movement
                    close_price = open_price * (1 + np.random.normal(0, bar_volatility * 1.5))
                
                else:
                    # Default behavior for other conditions
                    close_price = open_price * (1 + np.random.normal(0, bar_volatility))
                
            # Generate high and low with some randomness
            bar_range = abs(close_price - open_price)
            extra_range = max(bar_range * 0.5, open_price * bar_volatility * 0.5)
            
            if close_price > open_price:
                high_price = max(high_price if 'high_price' in locals() else close_price, 
                               close_price + np.random.uniform(0, extra_range))
                low_price = min(low_price if 'low_price' in locals() else open_price, 
                              open_price - np.random.uniform(0, extra_range))
            else:
                high_price = max(high_price if 'high_price' in locals() else open_price, 
                               open_price + np.random.uniform(0, extra_range))
                low_price = min(low_price if 'low_price' in locals() else close_price, 
                              close_price - np.random.uniform(0, extra_range))
            
            # Ensure logical OHLC relationships
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            # Add bar to the data
            data.append({
                'datetime': timestamps[i],
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': np.random.randint(100, 1000),  # Dummy volume
                'pattern': pattern
            })
            
            # Update current price for next bar
            current_price = close_price
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df.set_index('datetime', inplace=True)
    
    print(f"Generated {len(df)} bars of realistic market data with pattern labels")
    return df

def run_multi_param_test(data, param_combinations=None):
    """
    Run backtest with multiple parameter combinations.
    
    Parameters
    ----------
    data : pd.DataFrame
        OHLC data for testing
    param_combinations : List[Dict], optional
        List of parameter dictionaries to test
        
    Returns
    -------
    pd.DataFrame
        Results of parameter testing
    """
    if param_combinations is None:
        # Default parameter combinations to test
        param_combinations = [
            {'name': 'Default', 'opening_range_minutes': 6, 'take_profit_r_level': 1, 'stop_loss_pct': 0.5, 'risk_per_trade_pct': 1.0},
            {'name': 'Conservative', 'opening_range_minutes': 6, 'take_profit_r_level': 1, 'stop_loss_pct': 0.3, 'risk_per_trade_pct': 0.5},
            {'name': 'Aggressive', 'opening_range_minutes': 3, 'take_profit_r_level': 2, 'stop_loss_pct': 0.7, 'risk_per_trade_pct': 2.0},
            {'name': 'Short OR', 'opening_range_minutes': 3, 'take_profit_r_level': 1, 'stop_loss_pct': 0.5, 'risk_per_trade_pct': 1.0},
            {'name': 'Long OR', 'opening_range_minutes': 12, 'take_profit_r_level': 1, 'stop_loss_pct': 0.5, 'risk_per_trade_pct': 1.0},
            {'name': 'R2 Target', 'opening_range_minutes': 6, 'take_profit_r_level': 2, 'stop_loss_pct': 0.5, 'risk_per_trade_pct': 1.0},
            {'name': 'R3 Target', 'opening_range_minutes': 6, 'take_profit_r_level': 3, 'stop_loss_pct': 0.5, 'risk_per_trade_pct': 1.0},
            {'name': 'Tight SL', 'opening_range_minutes': 6, 'take_profit_r_level': 1, 'stop_loss_pct': 0.3, 'risk_per_trade_pct': 1.0},
            {'name': 'Wide SL', 'opening_range_minutes': 6, 'take_profit_r_level': 1, 'stop_loss_pct': 0.7, 'risk_per_trade_pct': 1.0},
        ]
    
    print(f"Running tests with {len(param_combinations)} parameter combinations...")
    
    # Store results
    results = []
    
    for params in param_combinations:
        print(f"\nTesting {params['name']} configuration:")
        print(f"OR Minutes: {params['opening_range_minutes']}, TP Level: R{params['take_profit_r_level']}, " 
              f"SL: {params['stop_loss_pct']}, Risk: {params['risk_per_trade_pct']}%")
        
        # Initialize strategy with these parameters
        strategy = ManciniOptimized(
            opening_range_minutes=params['opening_range_minutes'],
            take_profit_r_level=params['take_profit_r_level'],
            stop_loss_pct=params['stop_loss_pct'],
            risk_per_trade_pct=params['risk_per_trade_pct'],
            min_volatility_filter=True,
            check_market_direction=True,
            initial_capital=100000
        )
        
        # Run backtest
        backtest_results = strategy.backtest(data, visualize=False)
        
        # Extract metrics
        metrics = backtest_results['metrics'] if 'metrics' in backtest_results else {}
        
        # Save to results
        result = {
            'name': params['name'],
            'opening_range_minutes': params['opening_range_minutes'],
            'take_profit_r_level': params['take_profit_r_level'],
            'stop_loss_pct': params['stop_loss_pct'],
            'risk_per_trade_pct': params['risk_per_trade_pct'],
            'total_trades': metrics.get('total_trades', 0),
            'win_rate': metrics.get('win_rate', 0),
            'total_return': metrics.get('total_return', 0),
            'return_pct': metrics.get('return_pct', 0),
            'max_drawdown': metrics.get('max_drawdown', 0),
            'profit_factor': metrics.get('profit_factor', 0),
            'sharpe_ratio': metrics.get('sharpe_ratio', 0),
            'final_capital': metrics.get('final_capital', 100000)
        }
        
        results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by return percentage (descending)
    results_df = results_df.sort_values('return_pct', ascending=False)
    
    # Save results
    results_df.to_csv('parameter_test_results.csv', index=False)
    
    return results_df

def visualize_param_results(results_df):
    """
    Create visualizations of parameter test results.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results of parameter testing
    """
    # Create directory for parameter test charts
    os.makedirs('charts/param_tests', exist_ok=True)
    
    # 1. Return Percentage Comparison
    plt.figure(figsize=(12, 8))
    
    # Sort by return percentage
    plot_df = results_df.sort_values('return_pct')
    
    # Create horizontal bar chart
    colors = ['green' if x > 0 else 'red' for x in plot_df['return_pct']]
    plt.barh(plot_df['name'], plot_df['return_pct'], color=colors)
    
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.title('Strategy Return by Parameter Configuration', fontsize=16)
    plt.xlabel('Return (%)')
    plt.ylabel('Configuration')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('charts/param_tests/return_by_config.png')
    
    # 2. Win Rate Comparison
    plt.figure(figsize=(12, 8))
    
    # Sort by win rate
    plot_df = results_df.sort_values('win_rate')
    
    plt.barh(plot_df['name'], plot_df['win_rate'] * 100, color='skyblue')
    
    plt.axvline(x=50, color='red', linestyle='--', linewidth=1, label='50% Win Rate')
    plt.title('Win Rate by Parameter Configuration', fontsize=16)
    plt.xlabel('Win Rate (%)')
    plt.ylabel('Configuration')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('charts/param_tests/win_rate_by_config.png')
    
    # 3. Scatter plot: Win Rate vs Return
    plt.figure(figsize=(10, 8))
    
    plt.scatter(results_df['win_rate'] * 100, results_df['return_pct'], 
               s=100, alpha=0.7, c=results_df['profit_factor'], cmap='viridis')
    
    # Add labels for each point
    for i, row in results_df.iterrows():
        plt.annotate(row['name'], 
                   (row['win_rate'] * 100, row['return_pct']),
                   xytext=(5, 5),
                   textcoords='offset points')
    
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.axvline(x=50, color='black', linestyle='-', linewidth=0.5)
    
    plt.colorbar(label='Profit Factor')
    plt.title('Win Rate vs Return with Profit Factor', fontsize=16)
    plt.xlabel('Win Rate (%)')
    plt.ylabel('Return (%)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('charts/param_tests/win_rate_vs_return.png')
    
    # 4. Parameter impact visualization
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # Opening Range Minutes vs Return
    sns.boxplot(x='opening_range_minutes', y='return_pct', data=results_df, ax=axs[0, 0])
    axs[0, 0].set_title('Opening Range Minutes vs Return')
    axs[0, 0].set_xlabel('Opening Range Minutes')
    axs[0, 0].set_ylabel('Return (%)')
    
    # Take Profit Level vs Return
    sns.boxplot(x='take_profit_r_level', y='return_pct', data=results_df, ax=axs[0, 1])
    axs[0, 1].set_title('Take Profit Level vs Return')
    axs[0, 1].set_xlabel('Take Profit Level (R1/R2/R3)')
    axs[0, 1].set_ylabel('Return (%)')
    
    # Stop Loss % vs Return
    sns.boxplot(x='stop_loss_pct', y='return_pct', data=results_df, ax=axs[1, 0])
    axs[1, 0].set_title('Stop Loss % vs Return')
    axs[1, 0].set_xlabel('Stop Loss (% of Opening Range)')
    axs[1, 0].set_ylabel('Return (%)')
    
    # Risk Per Trade vs Return
    sns.boxplot(x='risk_per_trade_pct', y='return_pct', data=results_df, ax=axs[1, 1])
    axs[1, 1].set_title('Risk Per Trade vs Return')
    axs[1, 1].set_xlabel('Risk Per Trade (%)')
    axs[1, 1].set_ylabel('Return (%)')
    
    plt.tight_layout()
    plt.savefig('charts/param_tests/parameter_impact.png')
    
    # 5. Summary table visualization
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis('off')
    
    # Top section - Parameter Test Results
    ax.text(0.5, 0.98, 'Adam Mancini Strategy - Parameter Test Results', 
            horizontalalignment='center', verticalalignment='center',
            fontsize=16, fontweight='bold')
    
    # Create table with top 3 configurations
    top_configs = results_df.head(3)
    
    # Prepare table data
    table_data = []
    for i, row in top_configs.iterrows():
        table_data.append([
            row['name'],
            f"{row['opening_range_minutes']}",
            f"R{row['take_profit_r_level']}",
            f"{row['stop_loss_pct']:.1f}",
            f"{row['risk_per_trade_pct']:.1f}%",
            f"{row['total_trades']}",
            f"{row['win_rate']:.2%}",
            f"{row['return_pct']:.2f}%",
            f"{row['profit_factor']:.2f}",
            f"{row['max_drawdown']:.2f}%",
            f"${row['final_capital']:,.2f}"
        ])
    
    # Create table
    columns = ['Config', 'OR Min', 'TP', 'SL', 'Risk', 'Trades', 'Win Rate', 
               'Return', 'PF', 'Max DD', 'Final $']
    
    table = ax.table(cellText=table_data, colLabels=columns,
                    loc='center', cellLoc='center', 
                    colWidths=[0.12] + [0.08] * 10)
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    # Color top row differently
    for j in range(len(columns)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white')
    
    plt.tight_layout()
    plt.savefig('charts/param_tests/top_configurations.png')
    
    print("\nParameter test visualizations complete. Charts saved to 'charts/param_tests' directory.")

def run_best_strategy(data, best_params):
    """
    Run and visualize the best strategy configuration.
    
    Parameters
    ----------
    data : pd.DataFrame
        OHLC data for testing
    best_params : Dict
        Best parameter configuration
    """
    print(f"\nRunning final backtest with best configuration: {best_params['name']}")
    
    # Initialize strategy with best parameters
    strategy = ManciniOptimized(
        opening_range_minutes=best_params['opening_range_minutes'],
        take_profit_r_level=best_params['take_profit_r_level'],
        stop_loss_pct=best_params['stop_loss_pct'],
        risk_per_trade_pct=best_params['risk_per_trade_pct'],
        min_volatility_filter=True,
        check_market_direction=True,
        initial_capital=100000
    )
    
    # Run backtest with visualization
    backtest_results = strategy.backtest(data, visualize=True)
    
    # Create a more comprehensive final report
    create_final_report(strategy, backtest_results, best_params)
    
    return backtest_results

def create_final_report(strategy, results, params):
    """
    Create a comprehensive final report.
    
    Parameters
    ----------
    strategy : ManciniOptimized
        Strategy instance
    results : Dict
        Backtest results
    params : Dict
        Parameter configuration
    """
    # Create a detailed report
    report_path = 'adam_mancini_strategy_report.md'
    
    with open(report_path, 'w') as f:
        f.write("# Adam Mancini Trading Strategy - Final Report\n\n")
        
        f.write("## Strategy Configuration\n\n")
        f.write(f"- **Opening Range Minutes**: {params['opening_range_minutes']}\n")
        f.write(f"- **Take Profit Level**: R{params['take_profit_r_level']}\n")
        f.write(f"- **Stop Loss**: {params['stop_loss_pct']} × Opening Range Size\n")
        f.write(f"- **Risk Per Trade**: {params['risk_per_trade_pct']}% of Capital\n")
        f.write(f"- **Initial Capital**: ${strategy.initial_capital:,.2f}\n")
        f.write(f"- **Market Direction Filter**: {'Enabled' if strategy.check_market_direction else 'Disabled'}\n")
        f.write(f"- **Volatility Filter**: {'Enabled' if strategy.min_volatility_filter else 'Disabled'}\n\n")
        
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
        f.write(f"- **Average Hold Time**: {metrics['avg_hold_time']:.2f} minutes\n")
        f.write(f"- **Final Capital**: ${metrics['final_capital']:,.2f}\n\n")
        
        f.write("## Exit Reason Analysis\n\n")
        f.write("| Exit Reason | Count | Percentage |\n")
        f.write("|------------|-------|------------|\n")
        
        for reason, count in metrics.get('exit_reasons', {}).items():
            percentage = count / metrics['total_trades'] * 100 if metrics['total_trades'] > 0 else 0
            f.write(f"| {reason} | {count} | {percentage:.2f}% |\n")
        
        f.write("\n## Market Direction Analysis\n\n")
        f.write("| Market Direction | Trades | Win Rate | Avg Return |\n")
        f.write("|------------------|--------|----------|------------|\n")
        
        direction_labels = {-1: 'Downtrend', 0: 'Sideways', 1: 'Uptrend'}
        
        for direction, stats in metrics.get('direction_results', {}).items():
            f.write(f"| {direction_labels.get(direction, str(direction))} | {stats['count']} | {stats['win_rate']:.2%} | ${stats['avg_return']:,.2f} |\n")
        
        f.write("\n## Implementation Recommendations\n\n")
        
        # Add recommendations based on results
        if metrics['win_rate'] > 0.5:
            f.write("- **Position Sizing**: This strategy has a positive win rate. Consider using a fixed position size relative to account size.\n")
        else:
            f.write("- **Position Sizing**: This strategy has a win rate below 50%. Careful position sizing is critical - risk no more than 1% per trade.\n")
        
        if metrics['avg_win'] > abs(metrics['avg_loss']):
            f.write("- **Trade Management**: Average win is larger than average loss. Consider trailing stops to maximize profitable trades.\n")
        else:
            f.write("- **Trade Management**: Average loss is larger than average win. Consider tightening stop losses or taking partial profits earlier.\n")
        
        # Direction-based recommendations
        direction_results = metrics.get('direction_results', {})
        if 1 in direction_results and direction_results[1]['win_rate'] > 0.5:
            f.write("- **Market Direction**: Strategy performs well in uptrends. Prioritize long signals during uptrending markets.\n")
        if -1 in direction_results and direction_results[-1]['win_rate'] > 0.5:
            f.write("- **Market Direction**: Strategy performs well in downtrends. Consider short signals during downtrending markets.\n")
        
        f.write("- **Time Frame**: The best results were achieved with a ")
        if params['opening_range_minutes'] <= 3:
            f.write("short opening range (3 minutes). This suggests the strategy works best when responding quickly to market movements.\n")
        elif params['opening_range_minutes'] >= 12:
            f.write("long opening range (12+ minutes). This suggests the strategy benefits from filtering out noise with a longer observation period.\n")
        else:
            f.write(f"balanced opening range ({params['opening_range_minutes']} minutes). This provides a good balance between responsiveness and noise filtering.\n")
        
        f.write("- **Risk Management**: ")
        if params['stop_loss_pct'] <= 0.3:
            f.write("The optimal stop loss is tight (30% of opening range size). Use precise entries and consider scaling in to positions to manage risk.\n")
        elif params['stop_loss_pct'] >= 0.7:
            f.write("The optimal stop loss is wide (70% of opening range size). This strategy may encounter larger drawdowns, so consider reducing position sizes.\n")
        else:
            f.write(f"A moderate stop loss ({params['stop_loss_pct'] * 100}% of opening range) provides the best balance between protection and allowing trades room to breathe.\n")
        
        f.write("\n## Conclusion\n\n")
        
        if metrics['return_pct'] > 0 and metrics['win_rate'] > 0.5 and metrics['profit_factor'] > 1.5:
            f.write("The Adam Mancini strategy has demonstrated strong potential in the tested market conditions. ")
            f.write("With a profitable track record, positive win rate, and good risk-adjusted returns, this strategy ")
            f.write("is recommended for implementation with proper risk management.\n\n")
        elif metrics['return_pct'] > 0 and metrics['profit_factor'] > 1:
            f.write("The Adam Mancini strategy has shown positive results, but with moderate performance metrics. ")
            f.write("This strategy may be suitable as part of a diversified trading approach, rather than as a standalone system. ")
            f.write("Consider additional filters or combining with other strategies for improved results.\n\n")
        else:
            f.write("The Adam Mancini strategy did not perform well in the tested market conditions. ")
            f.write("Further optimization or application in different market environments may be needed before live implementation. ")
            f.write("Consider revising the strategy rules or exploring alternative approaches.\n\n")
        
        f.write("### Next Steps\n\n")
        f.write("1. **Forward Testing**: Run the strategy in a paper trading environment for at least 30 days\n")
        f.write("2. **Parameter Monitoring**: Periodically review and optimize parameters based on recent market conditions\n")
        f.write("3. **Implementation Plan**: Start with small position sizes and gradually increase as performance is verified\n")
        f.write("4. **Performance Tracking**: Set up a system to monitor key metrics and evaluate strategy health\n")
    
    print(f"Comprehensive strategy report saved to {report_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run final Adam Mancini strategy analysis")
    
    parser.add_argument(
        "--real-data",
        type=str,
        help="Path to real historical data file (if not provided, will use synthetic data)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=60,
        help="Number of days to generate for synthetic data (default: 60)"
    )
    parser.add_argument(
        "--save-synthetic",
        action="store_true",
        help="Save the generated synthetic data"
    )
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('charts', exist_ok=True)
    
    # Get data
    if args.real_data:
        # Load real data
        from mancini_specialized_analysis import load_and_prepare_data
        data = load_and_prepare_data(args.real_data)
    else:
        # Generate realistic synthetic data
        data = generate_realistic_market_data(days=args.days)
        
        if args.save_synthetic:
            # Save synthetic data
            data_path = f"data/realistic_nifty_5min_{args.days}days.csv"
            data.to_csv(data_path)
            print(f"Saved synthetic data to {data_path}")
    
    # Run parameter tests
    results_df = run_multi_param_test(data)
    
    # Visualize parameter test results
    visualize_param_results(results_df)
    
    # Get best parameters
    best_params = results_df.iloc[0].to_dict()
    
    # Run and visualize best strategy
    final_results = run_best_strategy(data, best_params)