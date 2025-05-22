import pandas as pd

import numpy as np
import pytest


from indiatrader.strategies import AdamManciniNiftyStrategy


def test_adam_mancini_strategy_signals():
    # Create simple test data with realistic index
    idx = pd.date_range("2024-01-01 09:15", periods=10, freq="min")
    close = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    data = pd.DataFrame({
        "high": [c + 1 for c in close],
        "low": [c - 1 for c in close],
        "close": close,
    }, index=idx)

    strat = AdamManciniNiftyStrategy(open_range_minutes=3)
    out = strat.generate_signals(data)

    # Basic structure checks
    assert "long_signal" in out.columns
    assert "short_signal" in out.columns
    assert "high_or" in out.columns
    assert "low_or" in out.columns
    assert "pivot" in out.columns
    
    # Check that signals are numeric
    assert out["long_signal"].dtype in ['int64', 'float64']
    assert out["short_signal"].dtype in ['int64', 'float64']
    assert len(out) == len(data)


def test_adam_mancini_strategy_short_signal():
    # Create simple test data with descending prices
    idx = pd.date_range("2024-01-01 09:15", periods=10, freq="min")
    close = [110, 109, 108, 107, 106, 105, 104, 103, 102, 101]
    data = pd.DataFrame({
        "high": [c + 1 for c in close],
        "low": [c - 1 for c in close],
        "close": close,
    }, index=idx)

    strat = AdamManciniNiftyStrategy(open_range_minutes=3)
    out = strat.generate_signals(data)

    # Basic structure checks
    assert "long_signal" in out.columns
    assert "short_signal" in out.columns
    assert "high_or" in out.columns
    assert "low_or" in out.columns
    assert "pivot" in out.columns
    
    # Check that signals are numeric
    assert out["long_signal"].dtype in ['int64', 'float64']
    assert out["short_signal"].dtype in ['int64', 'float64']
    assert len(out) == len(data)
    
    # Check the basic structure is correct
    assert out["long_signal"].notna().all()
    assert out["short_signal"].notna().all()


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
