"""
IndiaTrader: AI-Driven Intraday Trading Platform for Indian Stock Markets
"""

__version__ = "0.1.1"

from .strategies.adam_mancini import AdamManciniNiftyStrategy
from .features import FeaturePipeline, TechnicalFeatureGenerator, OrderFlowFeatureGenerator, NLPFeatureGenerator
from .models.patchtst import PatchTSTModel, train_model as train_patchtst
from .reinforcement.xlstm_ppo import TradingEnv, xLSTMPPO, train_agent
from .backtesting.framework import Backtester
from .broker.paper import PaperBroker

__all__ = [
    "AdamManciniNiftyStrategy",
    "FeaturePipeline",
    "TechnicalFeatureGenerator",
    "OrderFlowFeatureGenerator",
    "NLPFeatureGenerator",
    "PatchTSTModel",
    "train_patchtst",
    "TradingEnv",
    "xLSTMPPO",
    "train_agent",
    "Backtester",
    "PaperBroker",
]
