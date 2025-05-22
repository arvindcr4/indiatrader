# Adam Mancini Key Level Breakdown Strategy - Final Report

## Strategy Configuration

- **Support Level Lookback**: 10 bars
- **Confirmation Candles**: 1
- **Stop Loss**: 0.5% of Entry Price
- **Take Profit R-Ratio**: 3.0:1
- **Risk Per Trade**: 1.0% of Capital
- **Initial Capital**: $100,000.00
- **Market Direction Filter**: Enabled

## Performance Summary

- **Total Trades**: 3
- **Winning Trades**: 1 (33.33%)
- **Losing Trades**: 2 (66.67%)
- **Total Return**: $-1,030.22 (-1.03%)
- **Profit Factor**: 0.47
- **Win/Loss Ratio**: 0.94
- **Max Drawdown**: -1.94%
- **Sharpe Ratio**: -2.06
- **Average Trade P&L**: $-343.41
- **Average Win**: $906.14
- **Average Loss**: $-968.18
- **Final Capital**: $98,969.78

## Exit Reason Analysis

| Exit Reason | Count | Percentage |
|------------|-------|------------|
| Stop Loss | 2 | 66.67% |
| End of Data | 1 | 33.33% |

## Market Direction Analysis

| Market Direction | Trades | Win Rate | Avg Return |
|------------------|--------|----------|------------|
| Downtrend | 2 | 0.00% | $-968.18 |
| Uptrend | 1 | 100.00% | $906.14 |

## Implementation Recommendations

- The lower win rate suggests focusing on proper position sizing and risk management.
- The low profit factor indicates the need for further refinement or different market conditions.

## Conclusion

The Adam Mancini key level breakdown strategy shows mixed results in the tested period. Further optimization and testing in different market conditions is recommended before implementing in a live trading environment.

### Next Steps

1. **Paper Trading**: Test the strategy in a simulated environment with real-time data
2. **Parameter Refinement**: Continue to optimize parameters based on market conditions
3. **Risk Management**: Implement strict risk controls to protect capital
4. **Diversification**: Consider using this strategy as part of a broader trading approach
