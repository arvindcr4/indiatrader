# Adam Mancini Strategy Optimization Report

## Optimization Summary

Data File: `data/nifty_5min_30days.csv`
Configurations Tested: 6

## Best Configuration

- **Configuration Name**: Tight SL
- **Opening Range**: 6 minutes
- **Take Profit Level**: R1
- **Stop Loss**: 0.3 × Opening Range Size
- **Risk Per Trade**: 1.0%
- **Direction Filter**: Enabled

### Performance Metrics

- **Total Trades**: 8
- **Win Rate**: 0.00%
- **Total Return**: -72.66%
- **Profit Factor**: 0.00
- **Max Drawdown**: -72.66%
- **Sharpe Ratio**: -4.64
- **Final Capital**: $27,342.52

## All Configuration Results

| Configuration | OR Min | TP Level | SL % | Risk % | Direction Filter | Trades | Win Rate | Return % | Profit Factor |
|--------------|--------|----------|------|--------|------------------|--------|----------|----------|---------------|
| Long OR | 12 | R1 | 0.5 | 1.0% | On | 0 | 0.00% | 0.00% | 0.00 |
| Wide SL | 6 | R1 | 0.8 | 1.0% | On | 8 | 0.00% | -34.60% | 0.00 |
| Default | 6 | R1 | 0.5 | 1.0% | On | 8 | 0.00% | -50.87% | 0.00 |
| No Direction Filter | 6 | R1 | 0.5 | 1.0% | Off | 13 | 7.69% | -51.37% | 0.01 |
| Tight SL | 6 | R1 | 0.3 | 1.0% | On | 8 | 0.00% | -72.66% | 0.00 |
| Short OR | 3 | R1 | 0.5 | 1.0% | On | 15 | 0.00% | -74.27% | 0.00 |

## Parameter Impact Analysis

### Opening Range Minutes
The optimal opening range duration appears to be **12 minutes**.

### Stop Loss Percentage
The optimal stop loss percentage appears to be **0.8 × Opening Range Size**.

### Direction Filter
The direction filter should be **enabled** for optimal results.

## Recommendations

Despite optimization, none of the tested configurations showed profitability. This suggests that the Adam Mancini strategy may not be well-suited for the specific market conditions in the test data.

### Key Findings

1. None of the tested configurations showed positive returns, suggesting a fundamental incompatibility between the strategy and recent market conditions.
2. The least negative configuration was {best_result['name']}, which minimized losses to {best_result['return_pct']:.2f}%.
3. Consider exploring alternative strategies or significantly modifying the Adam Mancini approach for Indian markets.

### Next Steps

1. **Further Optimization**: Test more granular parameter variations around the best configuration
2. **Different Timeframes**: Evaluate the strategy on different time periods and market conditions
3. **Strategy Modifications**: Consider adding additional filters or modifying entry/exit criteria
4. **Paper Trading**: If pursuing implementation, start with paper trading to validate results
