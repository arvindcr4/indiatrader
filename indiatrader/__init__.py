"""
IndiaTrader: AI-Driven Intraday Trading Platform for Indian Stock Markets
"""

__version__ = "0.1.1"

from .strategies.adam_mancini import AdamManciniNiftyStrategy
from .features import FeaturePipeline, TechnicalFeatureGenerator, OrderFlowFeatureGenerator, NLPFeatureGenerator
from .backtesting.framework import Backtester
from .broker.paper import PaperBroker

__all__ = [
    "AdamManciniNiftyStrategy",
    "FeaturePipeline",
    "TechnicalFeatureGenerator",
    "OrderFlowFeatureGenerator",
    "NLPFeatureGenerator",
    "Backtester",
    "PaperBroker",
]

# Optional imports with fallback
try:
    from .models.patchtst import PatchTSTModel, train_model as train_patchtst
    __all__.extend(["PatchTSTModel", "train_patchtst"])
except ImportError:
    pass

try:
    from .reinforcement.xlstm_ppo import TradingEnv, xLSTMPPO, train_agent
    __all__.extend(["TradingEnv", "xLSTMPPO", "train_agent"])
except ImportError:
    pass
