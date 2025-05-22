"""Broker API integrations for IndiaTrader."""

__all__ = []

# Optional imports to avoid network calls during module import
try:
    from .dhan import DhanClient
    __all__.append("DhanClient")
except ImportError:
    pass

try:
    from .icici import ICICIBreezeClient
    __all__.append("ICICIBreezeClient")
except ImportError:
    pass