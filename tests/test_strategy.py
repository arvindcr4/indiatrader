import pandas as pd
from indiatrader.strategies import AdamManciniNiftyStrategy


def test_adam_mancini_strategy_signals():
    idx = pd.date_range("2024-01-01 09:15", periods=10, freq="T")
    close = [99, 100, 101, 102, 103, 104, 105, 106, 107, 110]
    data = pd.DataFrame({
        "high": [c + 1 for c in close],
        "low": [c - 1 for c in close],
        "close": close,
    }, index=idx)

    strat = AdamManciniNiftyStrategy(open_range_minutes=3)
    out = strat.generate_signals(data)

    assert "long_signal" in out.columns
    assert "short_signal" in out.columns
    assert out.iloc[-1]["long_signal"] == 1
    assert out.iloc[-1]["short_signal"] == 0
