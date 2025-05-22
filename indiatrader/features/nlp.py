"""NLP Feature Generator for text data."""

import logging
import hashlib
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class NLPFeatureGenerator:
    """Generate NLP features from text data."""

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize NLP feature generator."""
        # Set cache directory
        if cache_dir is None:
            self.cache_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "cache",
                "nlp"
            )
        else:
            self.cache_dir = cache_dir

        os.makedirs(self.cache_dir, exist_ok=True)

    def generate_features(self, df: pd.DataFrame, feature_config: List[Dict]) -> pd.DataFrame:
        """Generate features based on configuration.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with text data
        feature_config : List[Dict]
            List of feature specifications

        Returns
        -------
        pd.DataFrame
            DataFrame with added features
        """
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
        """Add sentiment scores to text data.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with text data

        Returns
        -------
        pd.DataFrame
            DataFrame with sentiment scores
        """
        if "text" not in df.columns:
            logger.warning("Text column not found for sentiment analysis")
            return df

        positive = {"good", "great", "positive", "up", "bull", "gain", "bullish", "rally", "buy"}
        negative = {"bad", "down", "negative", "bear", "loss", "bearish", "sell", "short"}

        def score(text: str) -> float:
            if pd.isna(text):
                return 0.0
            tokens = text.lower().split()
            pos = sum(1 for t in tokens if t in positive)
            neg = sum(1 for t in tokens if t in negative)
            total = len(tokens) if tokens else 1
            return (pos - neg) / total

        df["sentiment_score"] = df["text"].fillna("").apply(score)
        return df

    def _add_embedding(self, df: pd.DataFrame, dims: int = 16) -> pd.DataFrame:
        """Add text embeddings to text data.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with text data
        dims : int, optional
            Dimensionality of embeddings

        Returns
        -------
        pd.DataFrame
            DataFrame with embeddings
        """
        if "text" not in df.columns:
            logger.warning("Text column not found for embeddings")
            return df

        def embed(text: str):
            if pd.isna(text):
                return [0] * dims
            digest = hashlib.sha256(text.encode()).digest()
            # Ensure we don't exceed the digest length (32 bytes)
            actual_dims = min(dims, len(digest))
            result = [float(b) for b in digest[:actual_dims]]
            # Pad with zeros if needed
            if len(result) < dims:
                result.extend([0.0] * (dims - len(result)))
            return result

        embeddings = df["text"].fillna("").apply(embed).tolist()
        embedding_matrix = np.array(embeddings)

        # Create embedding columns efficiently
        embedding_df = pd.DataFrame(
            embedding_matrix,
            columns=[f"embedding_{i + 1}" for i in range(dims)],
            index=df.index
        )

        # Concatenate with original dataframe
        return pd.concat([df, embedding_df], axis=1)
