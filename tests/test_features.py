import pandas as pd
from indiatrader.features import FeaturePipeline


def test_feature_pipeline_market():
    df = pd.DataFrame({
        "open": [100, 101, 102, 103, 104],
        "high": [101, 102, 103, 104, 105],
        "low": [99, 100, 101, 102, 103],
        "close": [100, 101, 102, 103, 104],
        "volume": [10, 20, 30, 40, 50],
        "bid_volume": [6, 12, 18, 24, 30],
        "ask_volume": [4, 8, 12, 16, 20],
    })
    pipeline = FeaturePipeline()
    out_df, _ = pipeline.generate(df)
    assert "sma_5" in out_df.columns
    assert "vwap_5" in out_df.columns
    assert "order_imbalance" in out_df.columns


def test_feature_pipeline_nlp():
    text_df = pd.DataFrame({"text": ["Market is good", "Bad news for investors"]})
    pipeline = FeaturePipeline()
    _, out_df = pipeline.generate(pd.DataFrame({"close": []}), text_df)
    assert "sentiment_score" in out_df.columns
    # Check that embedding columns exist
    embedding_cols = [col for col in out_df.columns if col.startswith("embedding_")]
    assert len(embedding_cols) > 0
