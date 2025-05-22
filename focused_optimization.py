#!/usr/bin/env python3
"""
Focused optimization of Adam Mancini strategy parameters for Indian markets.
This script tests a limited set of parameter combinations to find potential improvements.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np

# Import the strategy directly to avoid subprocess overhead
from final_mancini_analysis import ManciniStrategy, load_data, create_final_report

def run_focused_optimization(data_file):
    """Run focused parameter optimization."""
    print(f"Loading data from {data_file}...")
    data = load_data(data_file)
    
    # Define a limited set of parameter combinations to test
    param_sets = [
        {"name": "Default", "or_min": 6, "tp_level": 1, "sl_pct": 0.5, "risk_pct": 1.0, "direction_filter": True},
        {"name": "Wide SL", "or_min": 6, "tp_level": 1, "sl_pct": 0.8, "risk_pct": 1.0, "direction_filter": True},
        {"name": "Tight SL", "or_min": 6, "tp_level": 1, "sl_pct": 0.3, "risk_pct": 1.0, "direction_filter": True},
        {"name": "Short OR", "or_min": 3, "tp_level": 1, "sl_pct": 0.5, "risk_pct": 1.0, "direction_filter": True},
        {"name": "Long OR", "or_min": 12, "tp_level": 1, "sl_pct": 0.5, "risk_pct": 1.0, "direction_filter": True},
        {"name": "No Direction Filter", "or_min": 6, "tp_level": 1, "sl_pct": 0.5, "risk_pct": 1.0, "direction_filter": False}
    ]
    
    results = []
    best_params = None
    best_return = -float('inf')
    
    for params in param_sets:
        print(f"\nTesting {params['name']} configuration...")
        print(f"OR Minutes: {params['or_min']}, TP Level: R{params['tp_level']}, " 
              f"SL: {params['sl_pct']}, Risk: {params['risk_pct']}%, "
              f"Direction Filter: {'On' if params['direction_filter'] else 'Off'}")
        
        # Create strategy with these parameters
        strategy = ManciniStrategy(
            opening_range_minutes=params['or_min'],
            take_profit_r_level=params['tp_level'],
            stop_loss_pct=params['sl_pct'],
            risk_per_trade_pct=params['risk_pct'],
            check_market_direction=params['direction_filter'],
            initial_capital=100000
        )
        
        # Run backtest without visualization
        backtest_results = strategy.backtest(data, visualize=False)
        metrics = backtest_results['metrics']
        
        # Store results
        result = {
            'name': params['name'],
            'or_min': params['or_min'],
            'tp_level': params['tp_level'],
            'sl_pct': params['sl_pct'],
            'risk_pct': params['risk_pct'],
            'direction_filter': params['direction_filter'],
            'total_trades': metrics.get('total_trades', 0),
            'win_rate': metrics.get('win_rate', 0),
            'total_return': metrics.get('total_return', 0),
            'return_pct': metrics.get('return_pct', 0),
            'profit_factor': metrics.get('profit_factor', 0),
            'max_drawdown': metrics.get('max_drawdown', 0),
            'sharpe_ratio': metrics.get('sharpe_ratio', 0),
            'final_capital': metrics.get('final_capital', 100000)
        }
        results.append(result)
        
        # Check if this is the best set so far
        if result['return_pct'] > best_return:
            best_return = result['return_pct']
            best_params = params.copy()
    
    # Convert to DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv('focused_optimization_results.csv', index=False)
    
    # Print best parameters
    print("\n=== Best Parameter Set ===")
    if best_params:
        best_idx = results_df['return_pct'].idxmax()
        best_result = results_df.iloc[best_idx]
        
        print(f"Configuration: {best_result['name']}")
        print(f"Opening Range: {best_result['or_min']} minutes")
        print(f"Take Profit: R{best_result['tp_level']}")
        print(f"Stop Loss: {best_result['sl_pct']} × Opening Range Size")
        print(f"Risk Per Trade: {best_result['risk_pct']}%")
        print(f"Direction Filter: {'On' if best_result['direction_filter'] else 'Off'}")
        print(f"Total Trades: {best_result['total_trades']}")
        print(f"Win Rate: {best_result['win_rate']:.2%}")
        print(f"Return: {best_result['return_pct']:.2f}%")
        print(f"Profit Factor: {best_result['profit_factor']:.2f}")
        print(f"Max Drawdown: {best_result['max_drawdown']:.2f}%")
        print(f"Final Capital: ${best_result['final_capital']:,.2f}")
    else:
        print("No valid parameter set found!")
    
    # Create charts
    if not results_df.empty:
        # Sort by return and create bar chart
        results_df = results_df.sort_values('return_pct', ascending=False)
        
        # Create directory
        os.makedirs('charts/optimization', exist_ok=True)
        
        # Return comparison
        plt.figure(figsize=(12, 8))
        colors = ['g' if x >= 0 else 'r' for x in results_df['return_pct']]
        plt.bar(results_df['name'], results_df['return_pct'], color=colors)
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.title('Strategy Return by Configuration', fontsize=16)
        plt.xlabel('Configuration')
        plt.ylabel('Return (%)')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('charts/optimization/return_comparison.png')
        
        # Create recommendation
        create_optimization_report(results_df, data_file)
    
    return results_df, best_params

def create_optimization_report(results_df, data_file):
    """Create a report with optimization findings and recommendations."""
    report_path = 'mancini_strategy_optimization_report.md'
    
    with open(report_path, 'w') as f:
        f.write("# Adam Mancini Strategy Optimization Report\n\n")
        
        f.write("## Optimization Summary\n\n")
        f.write(f"Data File: `{data_file}`\n")
        f.write(f"Configurations Tested: {len(results_df)}\n\n")
        
        # Best configuration
        best_idx = results_df['return_pct'].idxmax()
        best_result = results_df.iloc[best_idx]
        
        f.write("## Best Configuration\n\n")
        f.write(f"- **Configuration Name**: {best_result['name']}\n")
        f.write(f"- **Opening Range**: {best_result['or_min']} minutes\n")
        f.write(f"- **Take Profit Level**: R{best_result['tp_level']}\n")
        f.write(f"- **Stop Loss**: {best_result['sl_pct']} × Opening Range Size\n")
        f.write(f"- **Risk Per Trade**: {best_result['risk_pct']}%\n")
        f.write(f"- **Direction Filter**: {'Enabled' if best_result['direction_filter'] else 'Disabled'}\n\n")
        
        f.write("### Performance Metrics\n\n")
        f.write(f"- **Total Trades**: {best_result['total_trades']}\n")
        f.write(f"- **Win Rate**: {best_result['win_rate']:.2%}\n")
        f.write(f"- **Total Return**: {best_result['return_pct']:.2f}%\n")
        f.write(f"- **Profit Factor**: {best_result['profit_factor']:.2f}\n")
        f.write(f"- **Max Drawdown**: {best_result['max_drawdown']:.2f}%\n")
        f.write(f"- **Sharpe Ratio**: {best_result['sharpe_ratio']:.2f}\n")
        f.write(f"- **Final Capital**: ${best_result['final_capital']:,.2f}\n\n")
        
        # Results table
        f.write("## All Configuration Results\n\n")
        f.write("| Configuration | OR Min | TP Level | SL % | Risk % | Direction Filter | Trades | Win Rate | Return % | Profit Factor |\n")
        f.write("|--------------|--------|----------|------|--------|------------------|--------|----------|----------|---------------|\n")
        
        for _, row in results_df.iterrows():
            f.write(f"| {row['name']} | {row['or_min']} | R{row['tp_level']} | {row['sl_pct']} | {row['risk_pct']}% | {'On' if row['direction_filter'] else 'Off'} | {row['total_trades']} | {row['win_rate']:.2%} | {row['return_pct']:.2f}% | {row['profit_factor']:.2f} |\n")
        
        # Parameter impact
        f.write("\n## Parameter Impact Analysis\n\n")
        
        # Opening Range
        or_impact = results_df.groupby('or_min')['return_pct'].mean().reset_index()
        best_or = or_impact.loc[or_impact['return_pct'].idxmax(), 'or_min']
        
        f.write("### Opening Range Minutes\n")
        f.write(f"The optimal opening range duration appears to be **{best_or} minutes**.\n\n")
        
        # Stop Loss
        sl_impact = results_df.groupby('sl_pct')['return_pct'].mean().reset_index()
        best_sl = sl_impact.loc[sl_impact['return_pct'].idxmax(), 'sl_pct']
        
        f.write("### Stop Loss Percentage\n")
        f.write(f"The optimal stop loss percentage appears to be **{best_sl} × Opening Range Size**.\n\n")
        
        # Direction Filter
        dir_impact = results_df.groupby('direction_filter')['return_pct'].mean().reset_index()
        dir_filter_better = dir_impact.loc[dir_impact['return_pct'].idxmax(), 'direction_filter']
        
        f.write("### Direction Filter\n")
        f.write(f"The direction filter should be **{'enabled' if dir_filter_better else 'disabled'}** for optimal results.\n\n")
        
        # Overall recommendations
        f.write("## Recommendations\n\n")
        
        if best_result['return_pct'] > 0:
            f.write("The optimization process has identified a profitable parameter set. ")
            f.write("This configuration shows potential for live trading with appropriate risk management.\n\n")
        else:
            f.write("Despite optimization, none of the tested configurations showed profitability. ")
            f.write("This suggests that the Adam Mancini strategy may not be well-suited for the specific market conditions in the test data.\n\n")
        
        f.write("### Key Findings\n\n")
        
        if best_result['return_pct'] > 0:
            f.write("1. A ")
            if best_result['or_min'] <= 3:
                f.write("shorter opening range (3 minutes) ")
            elif best_result['or_min'] >= 12:
                f.write("longer opening range (12 minutes) ")
            else:
                f.write(f"balanced opening range ({best_result['or_min']} minutes) ")
            f.write("appears to be most effective.\n")
                
            f.write("2. ")
            if best_result['sl_pct'] <= 0.3:
                f.write("Tighter stop losses (0.3 × OR) result in better risk control.")
            elif best_result['sl_pct'] >= 0.7:
                f.write("Wider stop losses (0.7+ × OR) allow trades more room to develop.")
            else:
                f.write(f"Balanced stop losses ({best_result['sl_pct']} × OR) provide good risk/reward trade-offs.")
            f.write("\n")
            
            f.write("3. The market direction filter ")
            if best_result['direction_filter']:
                f.write("improves performance by filtering out counter-trend trades.")
            else:
                f.write("reduces performance by filtering out valid trade opportunities.")
            f.write("\n\n")
        else:
            f.write("1. None of the tested configurations showed positive returns, suggesting a fundamental incompatibility between the strategy and recent market conditions.\n")
            f.write("2. The least negative configuration was {best_result['name']}, which minimized losses to {best_result['return_pct']:.2f}%.\n")
            f.write("3. Consider exploring alternative strategies or significantly modifying the Adam Mancini approach for Indian markets.\n\n")
        
        f.write("### Next Steps\n\n")
        f.write("1. **Further Optimization**: Test more granular parameter variations around the best configuration\n")
        f.write("2. **Different Timeframes**: Evaluate the strategy on different time periods and market conditions\n")
        f.write("3. **Strategy Modifications**: Consider adding additional filters or modifying entry/exit criteria\n")
        f.write("4. **Paper Trading**: If pursuing implementation, start with paper trading to validate results\n")
        
    print(f"Optimization report saved to {report_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run focused optimization of Adam Mancini strategy")
    parser.add_argument(
        "--data-file", 
        type=str, 
        default="data/nifty_5min_30days.csv",
        help="Path to data file for backtesting"
    )
    
    args = parser.parse_args()
    
    # Run optimization
    results_df, best_params = run_focused_optimization(args.data_file)