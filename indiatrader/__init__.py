"""
IndiaTrader: AI-Driven Intraday Trading Platform for Indian Stock Markets
"""

__version__ = "0.1.1"

from .strategies.adam_mancini import AdamManciniNiftyStrategy

__all__ = ["AdamManciniNiftyStrategy"]
