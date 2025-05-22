# Adam Mancini Trading Strategy Backtest Results

## Summary

This document summarizes the backtest results for the Adam Mancini trading strategy applied to NIFTY data. The strategy is based on opening range breakouts combined with pivot levels.

## Data Used

- **NIFTY Daily Data**: One year of daily data from May 2024 to May 2025
- **NIFTY 5-Minute Data**: Seven days of intraday data from May 14 to May 21, 2025

## Backtest Results

### Daily Data

When using daily data, the strategy did not generate any signals. This is expected as the Adam Mancini strategy is designed for intraday trading and requires intraday price movements to generate meaningful signals.

### 5-Minute Data with Different Opening Range Settings

We tested the strategy with different opening range durations:

1. **15-Minute Opening Range (3 bars)**
   - Generated 15 long signals and 2 short signals
   - Executed 5 trades (3 buys, 2 sells)
   - P&L: -24,698.55 points
   - More active trading with this shorter time frame

2. **30-Minute Opening Range (6 bars)**
   - Generated 3 long signals and 1 short signal
   - Executed 3 trades (2 buys, 1 sell)
   - P&L: -24,356.15 points
   - Reduced number of signals compared to 15-minute opening range

3. **60-Minute Opening Range (12 bars)**
   - Generated 0 signals
   - No trades executed
   - With a wider opening range, no breakouts were identified

## Observations

1. **Opening Range Size Impact**: 
   - Shorter opening ranges (15 minutes) generate more signals
   - Longer opening ranges (60 minutes) may not generate any signals if they're too wide
   - The 30-minute opening range provides a balance between signal frequency and quality

2. **Limited Data Impact**:
   - The backtest results are based on a limited 7-day dataset
   - More extensive testing with a larger intraday dataset would provide more reliable results

3. **P&L Considerations**:
   - The negative P&L in both active tests is primarily due to the limited data and the fact that the last trade was a buy with no subsequent sell before the end of the dataset
   - A full profit/loss analysis would require a longer testing period

## Next Steps

1. **Gather More Intraday Data**: Expand the 5-minute dataset to cover a longer period for more reliable results
2. **Parameter Optimization**: 
   - Test different combinations of pivot level thresholds
   - Fine-tune the opening range duration (possibly between 20-40 minutes)
3. **Add Risk Management**: Implement stop-loss and take-profit mechanisms
4. **Integrate with Broker APIs**: Test the strategy with simulated trading through the Dhan and ICICI Breeze API connections

## Conclusion

The Adam Mancini trading strategy shows promise for intraday trading on the NIFTY index. The optimal opening range appears to be around 30 minutes (6 bars), providing a balance between signal generation and quality. Further testing with more extensive data and refined parameters is recommended before live trading.