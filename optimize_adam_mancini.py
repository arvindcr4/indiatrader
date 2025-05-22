#!/usr/bin/env python3
"""
Optimize Adam Mancini strategy parameters for Indian markets.
This script tests multiple parameter combinations and finds the best performing set.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import subprocess
import json
import re

def parse_backtest_output(output):
    """Parse the backtest output to extract key metrics."""
    results = {}
    
    # Extract win rate
    win_rate_match = re.search(r'Win Rate: ([\d.]+)%', output)
    if win_rate_match:
        results['win_rate'] = float(win_rate_match.group(1)) / 100
    
    # Extract total return
    return_match = re.search(r'Total Return: \$([^(]+)\s+\(([-\d.]+)%\)', output)
    if return_match:
        results['total_return'] = float(return_match.group(1).replace(',', ''))
        results['return_pct'] = float(return_match.group(2))
    
    # Extract profit factor
    pf_match = re.search(r'Profit Factor: ([\d.]+)', output)
    if pf_match:
        results['profit_factor'] = float(pf_match.group(1))
    
    # Extract max drawdown
    dd_match = re.search(r'Max Drawdown: ([-\d.]+)%', output)
    if dd_match:
        results['max_drawdown'] = float(dd_match.group(1))
    
    # Extract Sharpe ratio
    sharpe_match = re.search(r'Sharpe Ratio: ([-\d.]+)', output)
    if sharpe_match:
        results['sharpe_ratio'] = float(sharpe_match.group(1))
    
    # Extract trade count
    trades_match = re.search(r'Total Trades: (\d+)', output)
    if trades_match:
        results['total_trades'] = int(trades_match.group(1))

    # Extract final capital
    capital_match = re.search(r'Final Capital: \$([\d,.]+)', output)
    if capital_match:
        results['final_capital'] = float(capital_match.group(1).replace(',', ''))
    
    return results

def run_backtest(data_file, or_minutes, tp_level, sl_pct, risk_pct, direction_filter=True):
    """Run a backtest with the given parameters."""
    cmd = [
        'python', 'final_mancini_analysis.py',
        '--data-file', data_file,
        '--open-range-minutes', str(or_minutes),
        '--take-profit-level', str(tp_level),
        '--stop-loss-pct', str(sl_pct),
        '--risk-per-trade', str(risk_pct)
    ]
    
    if not direction_filter:
        cmd.append('--no-direction-filter')
    
    # Run the command and capture output
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return parse_backtest_output(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running backtest: {e}")
        print(f"Stderr: {e.stderr}")
        return {}

def optimize_parameters(data_file):
    """Test multiple parameter combinations and find the best set."""
    # Parameter ranges to test
    or_minutes_range = [3, 6, 12]
    tp_level_range = [1, 2, 3]
    sl_pct_range = [0.3, 0.5, 0.7]
    risk_pct_range = [0.5, 1.0, 2.0]
    direction_filter_options = [True, False]
    
    # Store results
    results = []
    
    # Track best parameter set
    best_params = None
    best_return = -np.inf
    
    print(f"Testing {len(or_minutes_range) * len(tp_level_range) * len(sl_pct_range) * len(risk_pct_range) * len(direction_filter_options)} parameter combinations...")
    
    # Test all combinations
    for or_minutes in or_minutes_range:
        for tp_level in tp_level_range:
            for sl_pct in sl_pct_range:
                for risk_pct in risk_pct_range:
                    for direction_filter in direction_filter_options:
                        param_desc = (
                            f"OR: {or_minutes}min, TP: R{tp_level}, SL: {sl_pct}, "
                            f"Risk: {risk_pct}%, Direction Filter: {'On' if direction_filter else 'Off'}"
                        )
                        print(f"\nTesting {param_desc}...")
                        
                        # Run backtest
                        metrics = run_backtest(
                            data_file, or_minutes, tp_level, sl_pct, risk_pct, direction_filter
                        )
                        
                        if not metrics:
                            print("No results obtained, skipping...")
                            continue
                        
                        # Add parameters to results
                        metrics['or_minutes'] = or_minutes
                        metrics['tp_level'] = tp_level
                        metrics['sl_pct'] = sl_pct
                        metrics['risk_pct'] = risk_pct
                        metrics['direction_filter'] = direction_filter
                        metrics['param_desc'] = param_desc
                        
                        # Track best parameters by return percentage
                        if metrics.get('return_pct', -np.inf) > best_return:
                            best_return = metrics.get('return_pct', -np.inf)
                            best_params = metrics.copy()
                        
                        results.append(metrics)
                        
                        # Print summary
                        print(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
                        print(f"Return: {metrics.get('return_pct', 0):.2f}%")
                        print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
                        print(f"Trades: {metrics.get('total_trades', 0)}")
    
    # Convert to DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv('mancini_optimization_results.csv', index=False)
    
    # Print best parameters
    print("\n=== Best Parameter Set ===")
    if best_params:
        print(f"Parameters: {best_params['param_desc']}")
        print(f"Return: {best_params.get('return_pct', 0):.2f}%")
        print(f"Win Rate: {best_params.get('win_rate', 0):.2%}")
        print(f"Profit Factor: {best_params.get('profit_factor', 0):.2f}")
        print(f"Max Drawdown: {best_params.get('max_drawdown', 0):.2f}%")
        print(f"Trades: {best_params.get('total_trades', 0)}")
    else:
        print("No valid parameter set found!")
    
    return best_params, results_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize Adam Mancini strategy parameters")
    parser.add_argument(
        "--data-file", 
        type=str, 
        default="data/nifty_5min_30days.csv",
        help="Path to data file for backtesting"
    )
    
    args = parser.parse_args()
    
    # Run optimization
    best_params, results_df = optimize_parameters(args.data_file)
    
    # Create chart directory
    os.makedirs('charts/optimization', exist_ok=True)
    
    # Create performance visualization of top 5 parameter sets
    if not results_df.empty:
        top_results = results_df.sort_values('return_pct', ascending=False).head(5)
        
        plt.figure(figsize=(12, 8))
        plt.bar(top_results['param_desc'], top_results['return_pct'], color='blue', alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.title('Top 5 Parameter Sets by Return', fontsize=16)
        plt.xlabel('Parameters')
        plt.ylabel('Return (%)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('charts/optimization/top_parameters.png')
        
        # Create parameter impact visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Opening Range Minutes
        or_impact = results_df.groupby('or_minutes')['return_pct'].mean()
        axes[0, 0].bar(or_impact.index, or_impact.values, color='green', alpha=0.7)
        axes[0, 0].set_title('Impact of Opening Range Minutes')
        axes[0, 0].set_xlabel('Opening Range Minutes')
        axes[0, 0].set_ylabel('Average Return (%)')
        
        # Take Profit Level
        tp_impact = results_df.groupby('tp_level')['return_pct'].mean()
        axes[0, 1].bar(tp_impact.index, tp_impact.values, color='blue', alpha=0.7)
        axes[0, 1].set_title('Impact of Take Profit Level')
        axes[0, 1].set_xlabel('Take Profit Level (R1/R2/R3)')
        axes[0, 1].set_ylabel('Average Return (%)')
        
        # Stop Loss Percentage
        sl_impact = results_df.groupby('sl_pct')['return_pct'].mean()
        axes[1, 0].bar(sl_impact.index, sl_impact.values, color='purple', alpha=0.7)
        axes[1, 0].set_title('Impact of Stop Loss Percentage')
        axes[1, 0].set_xlabel('Stop Loss (% of Opening Range)')
        axes[1, 0].set_ylabel('Average Return (%)')
        
        # Direction Filter
        dir_impact = results_df.groupby('direction_filter')['return_pct'].mean()
        axes[1, 1].bar([str(x) for x in dir_impact.index], dir_impact.values, color='orange', alpha=0.7)
        axes[1, 1].set_title('Impact of Direction Filter')
        axes[1, 1].set_xlabel('Direction Filter (On/Off)')
        axes[1, 1].set_ylabel('Average Return (%)')
        
        plt.tight_layout()
        plt.savefig('charts/optimization/parameter_impact.png')
        
        print("\nOptimization analysis complete. Results saved to mancini_optimization_results.csv")
        print("Visualizations saved to charts/optimization/ directory")
    else:
        print("No valid results to visualize.")