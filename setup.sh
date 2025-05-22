#!/bin/bash

# Setup script for IndiaTrader project

# Create logs directory
mkdir -p logs

# Create data directories
mkdir -p data/nse/ohlcv/{1m,5m,15m,1h,1d}
mkdir -p data/nse/orderbook
mkdir -p data/bse/ohlcv/{1m,5m,15m,1h,1d}
mkdir -p data/bse/orderbook
mkdir -p data/news
mkdir -p data/social/{twitter,reddit}

# Create feature directories
mkdir -p features/nse/{1m,5m,15m,1h,1d}
mkdir -p features/bse/{1m,5m,15m,1h,1d}

# Create models directory
mkdir -p models/{nse,bse}

# Create Docker and deployment directories
mkdir -p deployment/docker

# Create virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install requirements
pip install -e .

# Display setup information
echo "IndiaTrader setup completed successfully!"
echo "Activate the virtual environment with: source venv/bin/activate"
echo "Run the CLI with: python -m indiatrader.deployment.cli"