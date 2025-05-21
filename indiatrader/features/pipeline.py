"""Feature generation pipeline."""

from typing import Optional, Tuple
import pandas as pd

from indiatrader.data.config import load_config
from indiatrader.features.technical import TechnicalFeatureGenerator
from indiatrader.features.orderflow import OrderFlowFeatureGenerator
from indiatrader.features.nlp import NLPFeatureGenerator


class FeaturePipeline:
    """Pipeline that orchestrates feature generation based on config.yaml."""

    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path)
        self.tech = TechnicalFeatureGenerator()
        self.order = OrderFlowFeatureGenerator()
        self.nlp = NLPFeatureGenerator()

    def generate(self, market_df: pd.DataFrame, text_df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        features_conf = self.config.get("features", {})
        if "technical" in features_conf:
            market_df = self.tech.generate_features(market_df, features_conf["technical"])
        if "order_flow" in features_conf:
            market_df = self.order.generate_features(market_df, features_conf["order_flow"])
        if text_df is not None and "nlp" in features_conf:
            text_df = self.nlp.generate_features(text_df, features_conf["nlp"])
        return market_df, text_df
