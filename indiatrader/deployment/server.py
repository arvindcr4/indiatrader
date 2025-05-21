"""
FastAPI server for deploying trading models.
"""

import os
import logging
import pandas as pd
import numpy as np
import json
import torch
import time
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query, Path
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import threading
import queue

from indiatrader.models.transformer import PatchTSTModel
from indiatrader.models.reinforcement import PPOAgent
from indiatrader.features.processor import FeatureProcessor
from indiatrader.data.market_data import NSEConnector, BSEConnector

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# Models
class SignalRequest(BaseModel):
    """
    Request model for generating signals.
    """
    symbol: str = Field(..., description="Trading symbol")
    exchange: str = Field("nse", description="Exchange name ('nse' or 'bse')")
    interval: str = Field("1d", description="Data interval ('1m', '5m', '15m', '1h', '1d')")
    model_type: str = Field("transformer", description="Model type ('transformer' or 'rl')")
    model_path: Optional[str] = Field(None, description="Path to model file")


class BacktestRequest(BaseModel):
    """
    Request model for running a backtest.
    """
    symbol: str = Field(..., description="Trading symbol")
    exchange: str = Field("nse", description="Exchange name ('nse' or 'bse')")
    interval: str = Field("1d", description="Data interval ('1m', '5m', '15m', '1h', '1d')")
    strategy: str = Field(..., description="Strategy name")
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")
    initial_capital: float = Field(100000.0, description="Initial capital")
    params: Optional[Dict[str, Any]] = Field(None, description="Strategy parameters")


class ZerodhaCredentials(BaseModel):
    """
    Zerodha API credentials.
    """
    api_key: str = Field(..., description="Zerodha API key")
    api_secret: str = Field(..., description="Zerodha API secret")
    access_token: Optional[str] = Field(None, description="Zerodha access token")


class ZerodhaOrderRequest(BaseModel):
    """
    Request model for placing a Zerodha order.
    """
    symbol: str = Field(..., description="Trading symbol")
    exchange: str = Field("NSE", description="Exchange name ('NSE' or 'BSE')")
    transaction_type: str = Field(..., description="Transaction type ('BUY' or 'SELL')")
    quantity: int = Field(..., description="Order quantity")
    order_type: str = Field("MARKET", description="Order type ('MARKET' or 'LIMIT')")
    price: Optional[float] = Field(None, description="Order price (required for LIMIT orders)")
    product: str = Field("CNC", description="Product type ('CNC', 'MIS', or 'NRML')")
    validity: str = Field("DAY", description="Order validity ('DAY' or 'IOC')")
    disclosed_quantity: Optional[int] = Field(None, description="Disclosed quantity")
    trigger_price: Optional[float] = Field(None, description="Trigger price for SL orders")


# Application
app = FastAPI(
    title="IndiaTrader API",
    description="API for IndiaTrader trading platform",
    version="0.1.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
models = {}
data_connectors = {}
zerodha_credentials = None
signal_queue = queue.Queue()
trading_active = False


# Background tasks
def process_signals():
    """
    Process signals from the queue and execute trades.
    """
    global trading_active
    
    logger.info("Signal processing thread started")
    
    while trading_active:
        try:
            # Get signal from queue with timeout
            signal = signal_queue.get(timeout=1)
            
            # Execute trade based on signal
            if zerodha_credentials:
                # Execute trade here
                logger.info(f"Executing trade: {signal}")
                
                # Implement actual trade execution logic here
                execute_trade(signal)
            else:
                logger.warning("No Zerodha credentials configured. Can't execute trade.")
            
            # Mark task as done
            signal_queue.task_done()
        
        except queue.Empty:
            # No signals in queue, continue waiting
            pass
        
        except Exception as e:
            logger.exception(f"Error processing signal: {str(e)}")
    
    logger.info("Signal processing thread stopped")


def execute_trade(signal: Dict[str, Any]):
    """
    Execute trade using Zerodha API.
    
    Args:
        signal: Trading signal
    """
    if not zerodha_credentials:
        logger.warning("No Zerodha credentials configured. Can't execute trade.")
        return
    
    try:
        from kiteconnect import KiteConnect
        
        # Initialize Kite Connect
        kite = KiteConnect(api_key=zerodha_credentials.api_key)
        
        # Set access token
        if zerodha_credentials.access_token:
            kite.set_access_token(zerodha_credentials.access_token)
        else:
            logger.warning("No access token provided. Authentication required.")
            return
        
        # Prepare order params
        order_params = {
            "tradingsymbol": signal["symbol"],
            "exchange": signal["exchange"],
            "transaction_type": "BUY" if signal["direction"] > 0 else "SELL",
            "quantity": signal["quantity"],
            "order_type": "MARKET",
            "product": "MIS"  # Intraday
        }
        
        # Place order
        order_id = kite.place_order(variety="regular", **order_params)
        
        logger.info(f"Order placed successfully. Order ID: {order_id}")
        
        # Store order details for tracking
        signal["order_id"] = order_id
        signal["status"] = "PLACED"
        signal["timestamp"] = datetime.now().isoformat()
        
        # Store the order
        # (In a production system, this would be stored in a database)
    
    except Exception as e:
        logger.exception(f"Error executing trade: {str(e)}")
        
        # Log failed order
        signal["status"] = "FAILED"
        signal["error"] = str(e)
        signal["timestamp"] = datetime.now().isoformat()


# Start background task for signal processing
@app.on_event("startup")
async def startup_event():
    """
    Startup event for the API server.
    """
    global trading_active
    
    # Initialize data connectors
    data_connectors["nse"] = NSEConnector()
    data_connectors["bse"] = BSEConnector()
    
    # Start signal processing thread
    trading_active = True
    threading.Thread(target=process_signals, daemon=True).start()
    
    logger.info("IndiaTrader API server started")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Shutdown event for the API server.
    """
    global trading_active
    
    # Stop signal processing thread
    trading_active = False
    
    logger.info("IndiaTrader API server shutdown")


# API Endpoints
@app.get("/")
async def root():
    """
    Root endpoint.
    """
    return {"message": "Welcome to IndiaTrader API"}


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.post("/api/signals")
async def generate_signals(request: SignalRequest):
    """
    Generate trading signals for a symbol.
    
    Args:
        request: Signal request
    
    Returns:
        Trading signals
    """
    try:
        symbol = request.symbol
        exchange = request.exchange.lower()
        interval = request.interval
        model_type = request.model_type.lower()
        model_path = request.model_path
        
        # Check if model is already loaded
        model_key = f"{model_type}_{exchange}_{symbol}_{interval}"
        
        if model_path and model_key not in models:
            # Load model
            if model_type == "transformer":
                models[model_key] = PatchTSTModel.load(model_path)
            elif model_type == "rl":
                models[model_key] = PPOAgent.load(model_path)
            else:
                raise HTTPException(status_code=400, detail=f"Unknown model type: {model_type}")
            
            logger.info(f"Loaded model: {model_key}")
        
        # Get latest market data
        if exchange not in data_connectors:
            raise HTTPException(status_code=400, detail=f"Unknown exchange: {exchange}")
        
        connector = data_connectors[exchange]
        
        # Get historical data
        days_back = 30  # Get data for last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Fetch historical data
        historical_data = connector.get_historical_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval
        )
        
        if historical_data.empty:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
        
        # Process features
        processor = FeatureProcessor()
        features_df = processor.create_model_ready_features(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            include_news=True,
            start_date=start_date,
            end_date=end_date,
            save_to_file=False
        )
        
        if features_df.empty:
            raise HTTPException(status_code=404, detail=f"No features available for {symbol}")
        
        # Generate signals
        signals = []
        
        if model_type == "transformer":
            model = models[model_key]
            
            # Prepare input data
            # (This would depend on the specific model architecture)
            # This is a simplified example
            feature_cols = [col for col in features_df.columns 
                          if col not in ["timestamp", "open", "high", "low", "close", "volume"]]
            
            X = features_df[feature_cols].values
            X = X.reshape(1, X.shape[0], X.shape[1])  # Add batch dimension
            
            # Generate predictions
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)
                prediction = np.argmax(probs, axis=1)[0]
                confidence = probs[0, prediction]
            else:
                prediction = model.predict(X)[0]
                confidence = 0.7  # Default confidence
            
            # Map prediction to signal
            if prediction == 1:  # Buy signal
                signal = {
                    "symbol": symbol,
                    "exchange": exchange.upper(),
                    "direction": 1,
                    "confidence": float(confidence),
                    "timestamp": datetime.now().isoformat(),
                    "price": features_df["close"].iloc[-1],
                    "interval": interval
                }
                signals.append(signal)
            elif prediction == 0:  # Sell signal
                signal = {
                    "symbol": symbol,
                    "exchange": exchange.upper(),
                    "direction": -1,
                    "confidence": float(confidence),
                    "timestamp": datetime.now().isoformat(),
                    "price": features_df["close"].iloc[-1],
                    "interval": interval
                }
                signals.append(signal)
        
        elif model_type == "rl":
            model = models[model_key]
            
            # Prepare state for RL model
            # (This would depend on the specific model architecture)
            # This is a simplified example
            feature_cols = [col for col in features_df.columns 
                          if col not in ["timestamp", "open", "high", "low", "close", "volume"]]
            
            state = features_df[feature_cols].values[-1]  # Latest state
            
            # Get action from PPO agent
            action, prob, _ = model.choose_action(state)
            
            # Map action to signal
            if action == 1:  # Buy signal
                signal = {
                    "symbol": symbol,
                    "exchange": exchange.upper(),
                    "direction": 1,
                    "confidence": float(prob),
                    "timestamp": datetime.now().isoformat(),
                    "price": features_df["close"].iloc[-1],
                    "interval": interval
                }
                signals.append(signal)
            elif action == 2:  # Sell signal
                signal = {
                    "symbol": symbol,
                    "exchange": exchange.upper(),
                    "direction": -1,
                    "confidence": float(prob),
                    "timestamp": datetime.now().isoformat(),
                    "price": features_df["close"].iloc[-1],
                    "interval": interval
                }
                signals.append(signal)
        
        return {"signals": signals}
    
    except Exception as e:
        logger.exception(f"Error generating signals: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/backtest")
async def run_backtest(request: BacktestRequest):
    """
    Run backtest for a strategy.
    
    Args:
        request: Backtest request
    
    Returns:
        Backtest results
    """
    try:
        # Import here to avoid circular imports
        from indiatrader.backtesting.run import run_strategy_backtest
        
        # Run backtest
        results = run_strategy_backtest(
            symbol=request.symbol,
            strategy_name=request.strategy,
            exchange=request.exchange,
            interval=request.interval,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
            strategy_params=request.params,
            use_features=True
        )
        
        # Process results for API response
        api_results = {
            "symbol": request.symbol,
            "strategy": request.strategy,
            "exchange": request.exchange,
            "interval": request.interval,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "initial_capital": request.initial_capital,
            "final_value": results["final_value"],
            "profit": results["profit"],
            "return_pct": results["return_pct"],
            "metrics": results["metrics"]
        }
        
        # Add trades if available
        if "trades" in results and not results["trades"].empty:
            trades_df = results["trades"]
            
            # Convert to list of dicts (handling datetime conversion)
            trades = []
            for _, row in trades_df.iterrows():
                trade = row.to_dict()
                
                # Convert datetime objects to ISO format strings
                for key, value in trade.items():
                    if isinstance(value, datetime):
                        trade[key] = value.isoformat()
                
                trades.append(trade)
            
            api_results["trades"] = trades
        
        return api_results
    
    except Exception as e:
        logger.exception(f"Error running backtest: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/zerodha/setup")
async def setup_zerodha(credentials: ZerodhaCredentials):
    """
    Set up Zerodha credentials.
    
    Args:
        credentials: Zerodha API credentials
    
    Returns:
        Success message
    """
    global zerodha_credentials
    
    try:
        # Store credentials (in a production system, these should be encrypted)
        zerodha_credentials = credentials
        
        # Test the credentials
        from kiteconnect import KiteConnect
        
        kite = KiteConnect(api_key=credentials.api_key)
        
        # If access token is provided, try to get user profile
        if credentials.access_token:
            kite.set_access_token(credentials.access_token)
            profile = kite.profile()
            
            return {
                "status": "success",
                "message": "Zerodha credentials configured successfully",
                "user": profile["user_name"]
            }
        
        # If no access token, generate login URL
        login_url = kite.login_url()
        
        return {
            "status": "partial",
            "message": "API key and secret configured. Access token required.",
            "login_url": login_url
        }
    
    except Exception as e:
        logger.exception(f"Error setting up Zerodha: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/zerodha/authorize")
async def authorize_zerodha(request_token: str = Query(..., description="Request token from Zerodha callback")):
    """
    Authorize Zerodha with request token.
    
    Args:
        request_token: Request token from Zerodha callback
    
    Returns:
        Success message with access token
    """
    global zerodha_credentials
    
    if not zerodha_credentials:
        raise HTTPException(status_code=400, detail="Zerodha credentials not configured")
    
    try:
        from kiteconnect import KiteConnect
        
        kite = KiteConnect(api_key=zerodha_credentials.api_key)
        
        # Generate access token
        data = kite.generate_session(request_token, api_secret=zerodha_credentials.api_secret)
        access_token = data["access_token"]
        
        # Update credentials
        zerodha_credentials.access_token = access_token
        
        # Set access token
        kite.set_access_token(access_token)
        
        # Get user profile
        profile = kite.profile()
        
        return {
            "status": "success",
            "message": "Zerodha authorized successfully",
            "access_token": access_token,
            "user": profile["user_name"]
        }
    
    except Exception as e:
        logger.exception(f"Error authorizing Zerodha: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/zerodha/order")
async def place_order(order: ZerodhaOrderRequest):
    """
    Place order on Zerodha.
    
    Args:
        order: Order details
    
    Returns:
        Order ID
    """
    if not zerodha_credentials or not zerodha_credentials.access_token:
        raise HTTPException(status_code=400, detail="Zerodha not configured or authenticated")
    
    try:
        from kiteconnect import KiteConnect
        
        kite = KiteConnect(api_key=zerodha_credentials.api_key)
        kite.set_access_token(zerodha_credentials.access_token)
        
        # Prepare order params
        order_params = {
            "tradingsymbol": order.symbol,
            "exchange": order.exchange,
            "transaction_type": order.transaction_type,
            "quantity": order.quantity,
            "order_type": order.order_type,
            "product": order.product,
            "validity": order.validity
        }
        
        # Add optional parameters if provided
        if order.order_type == "LIMIT" and order.price:
            order_params["price"] = order.price
        
        if order.disclosed_quantity:
            order_params["disclosed_quantity"] = order.disclosed_quantity
        
        if order.trigger_price:
            order_params["trigger_price"] = order.trigger_price
        
        # Place order
        order_id = kite.place_order(variety="regular", **order_params)
        
        return {
            "status": "success",
            "message": "Order placed successfully",
            "order_id": order_id
        }
    
    except Exception as e:
        logger.exception(f"Error placing order: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/execute")
async def execute_signal(signal: Dict[str, Any], background_tasks: BackgroundTasks):
    """
    Execute a trading signal.
    
    Args:
        signal: Trading signal
        background_tasks: FastAPI background tasks
    
    Returns:
        Success message
    """
    # Validate signal
    required_fields = ["symbol", "exchange", "direction"]
    for field in required_fields:
        if field not in signal:
            raise HTTPException(status_code=400, detail=f"Missing field: {field}")
    
    try:
        # Add to signal queue for processing
        signal_queue.put(signal)
        
        return {
            "status": "success",
            "message": "Signal queued for execution",
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.exception(f"Error queuing signal: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/trading/start")
async def start_trading():
    """
    Start automated trading.
    
    Returns:
        Success message
    """
    global trading_active
    
    if trading_active:
        return {"status": "success", "message": "Trading is already active"}
    
    # Start trading
    trading_active = True
    
    # Start signal processing thread if it's not running
    if not any(t.name == "SignalProcessor" for t in threading.enumerate()):
        thread = threading.Thread(target=process_signals, daemon=True, name="SignalProcessor")
        thread.start()
    
    return {
        "status": "success",
        "message": "Automated trading started",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/trading/stop")
async def stop_trading():
    """
    Stop automated trading.
    
    Returns:
        Success message
    """
    global trading_active
    
    if not trading_active:
        return {"status": "success", "message": "Trading is already stopped"}
    
    # Stop trading
    trading_active = False
    
    return {
        "status": "success",
        "message": "Automated trading stopped",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/models")
async def list_models():
    """
    List available models.
    
    Returns:
        List of models
    """
    # In a production system, this would list models from a database or file system
    model_files = []
    
    # Get the models directory
    models_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "models"
    )
    
    # Scan for model files
    for root, _, files in os.walk(models_dir):
        for file in files:
            if file.endswith(".pt") or file.endswith("_policy.pt"):
                model_path = os.path.join(root, file)
                rel_path = os.path.relpath(model_path, models_dir)
                
                # Determine model type
                model_type = "rl" if file.endswith("_policy.pt") else "transformer"
                
                # Extract symbol and exchange from path
                parts = rel_path.split(os.path.sep)
                symbol = parts[-1].split("_")[0] if len(parts) > 0 else "unknown"
                exchange = parts[-2] if len(parts) > 1 else "unknown"
                
                model_files.append({
                    "path": model_path,
                    "type": model_type,
                    "symbol": symbol,
                    "exchange": exchange
                })
    
    return {"models": model_files}


def main():
    """
    Main entry point for the API server.
    """
    # Configure logging
    log_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "logs"
    )
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"api_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Run server
    uvicorn.run(
        "indiatrader.deployment.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )


if __name__ == "__main__":
    main()