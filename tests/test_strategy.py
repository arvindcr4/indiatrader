import pandas as pd
import numpy as np
import pytest

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
    assert "high_or" in out.columns
    assert "low_or" in out.columns
    assert out.iloc[3]["high_or"] == 103
    assert out.iloc[3]["low_or"] == 98
    assert out.iloc[-1]["long_signal"] == 1
    assert out.iloc[-1]["short_signal"] == 0


def test_adam_mancini_strategy_short_signal():
    idx = pd.date_range("2024-01-01 09:15", periods=10, freq="T")
    close = [110, 109, 108, 107, 106, 105, 104, 103, 102, 100]
    data = pd.DataFrame({
        "high": [c + 1 for c in close],
        "low": [c - 1 for c in close],
        "close": close,
    }, index=idx)

    strat = AdamManciniNiftyStrategy(open_range_minutes=3)
    out = strat.generate_signals(data)

    assert out.iloc[-1]["long_signal"] == 0
    assert out.iloc[-1]["short_signal"] == -1


def test_compute_levels_single_day():
    daily = pd.DataFrame({
        "high": [111],
        "low": [98],
        "close": [110],
    }, index=[pd.Timestamp("2024-01-01")])
    strat = AdamManciniNiftyStrategy()
    levels = strat.compute_levels(daily)

    pivot = (111 + 98 + 110) / 3
    r1 = 2 * pivot - 98
    r2 = pivot + (111 - 98)
    r3 = r1 + (111 - 98)
    s1 = 2 * pivot - 111
    s2 = pivot - (111 - 98)
    s3 = s1 - (111 - 98)

    expected = pd.DataFrame({
        "pivot": [pivot],
        "r1": [r1],
        "r2": [r2],
        "r3": [r3],
        "s1": [s1],
        "s2": [s2],
        "s3": [s3],
    }, index=daily.index)

    pd.testing.assert_frame_equal(levels, expected)


def test_generate_signals_invalid_columns():
    df = pd.DataFrame({"high": [1, 2, 3]})
    strat = AdamManciniNiftyStrategy()
    with pytest.raises(ValueError):
        strat.generate_signals(df)
