"""Basic NLP feature generation utilities."""

import logging
import hashlib
import pandas as pd
from typing import Dict, List

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class NLPFeatureGenerator:
    """Generate NLP features for text data."""

    def generate_features(self, df: pd.DataFrame, feature_config: List[Dict]) -> pd.DataFrame:
        result_df = df.copy()
        for spec in feature_config:
            name = spec.get("name")
            params = spec.get("params", {})
            if name == "sentiment_score":
                result_df = self._add_sentiment_score(result_df)
            elif name == "news_embedding":
                result_df = self._add_embedding(result_df, params.get("dimensions", 16))
            else:
                logger.warning(f"Unknown NLP feature: {name}")
        return result_df

    def _add_sentiment_score(self, df: pd.DataFrame) -> pd.DataFrame:
        if "text" not in df.columns:
            logger.warning("Text column not found for sentiment analysis")
            return df
        positive = {"good", "great", "positive", "up", "bull", "gain"}
        negative = {"bad", "down", "negative", "bear", "loss"}

        def score(text: str) -> float:
            tokens = text.lower().split()
            pos = sum(1 for t in tokens if t in positive)
            neg = sum(1 for t in tokens if t in negative)
            total = len(tokens) if tokens else 1
            return (pos - neg) / total

        df["sentiment_score"] = df["text"].fillna("").apply(score)
        return df

    def _add_embedding(self, df: pd.DataFrame, dims: int) -> pd.DataFrame:
        if "text" not in df.columns:
            logger.warning("Text column not found for embeddings")
            return df

        def embed(text: str):
            digest = hashlib.sha256(text.encode()).digest()
            return [int(b) for b in digest[:dims]]

        df["news_embedding"] = df["text"].fillna("").apply(embed)
        return df
