
"""
Entry point for serving model inference.
"""

import os
import logging
import argparse
import time
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import json
from typing import Dict, List, Optional, Any, Union

from indiatrader.data.market_data import NSEConnector, BSEConnector
from indiatrader.features.processor import FeatureProcessor
from indiatrader.models.transformer import PatchTSTModel
from indiatrader.models.reinforcement import PPOAgent

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class ModelServer:
    """
    Server for model inference.
    """
    
    def __init__(self, 
                model_path: str,
                model_type: str = "transformer",
                device: str = "cpu",
                batch_size: int = 16,
                use_tensorrt: bool = False):
        """
        Initialize model server.
        
        Args:
            model_path: Path to model file
            model_type: Type of model ('transformer' or 'rl')
            device: Device to use ('cpu' or 'cuda')
            batch_size: Batch size for inference
            use_tensorrt: Whether to use TensorRT for acceleration
        """
        self.model_path = model_path
        self.model_type = model_type
        self.device = device
        self.batch_size = batch_size
        self.use_tensorrt = use_tensorrt
        
        # Load model
        self._load_model()
        
        # Initialize data connectors
        self.connectors = {
            "nse": NSEConnector(),
            "bse": BSEConnector()
        }
        
        # Initialize feature processor
        self.processor = FeatureProcessor()
    
    def _load_model(self):
        """
        Load model from file.
        """
        try:
            logger.info(f"Loading {self.model_type} model from {self.model_path}")
            
            if self.model_type == "transformer":
                self.model = PatchTSTModel.load(self.model_path)
                
                # Move model to device
                if hasattr(self.model, "model"):
                    self.model.model.to(self.device)
            
            elif self.model_type == "rl":
                self.model = PPOAgent.load(self.model_path)
                
                # Move model to device
                if hasattr(self.model, "policy"):
                    self.model.policy.to(self.device)
                if hasattr(self.model, "value"):
                    self.model.value.to(self.device)
            
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            # Optimize with TensorRT if requested
            if self.use_tensorrt:
                try:
                    import torch_tensorrt
                    
                    if self.model_type == "transformer" and hasattr(self.model, "model"):
                        logger.info("Optimizing model with TensorRT")
                        
                        # Create example input
                        example_input = torch.randn(
                            1, 
                            self.model.model.context_length, 
                            self.model.model.input_dim, 
                            device=self.device
                        )
                        
                        # Convert to TensorRT
                        self.model.model = torch_tensorrt.compile(
                            self.model.model,
                            inputs=[example_input],
                            enabled_precisions={torch.float32, torch.float16}
                        )
                        
                        logger.info("Model optimized with TensorRT")
                    
                    else:
                        logger.warning("TensorRT optimization not supported for this model type")
                
                except ImportError:
                    logger.warning("TensorRT not available. Skipping optimization.")
            
            logger.info("Model loaded successfully")
        
        except Exception as e:
            logger.exception(f"Error loading model: {str(e)}")
            raise
    
    def predict(self, 
               symbol: str, 
               exchange: str = "nse",
               interval: str = "1d",
               include_features: bool = False) -> Dict[str, Any]:
        """
        Generate predictions for a symbol.
        
        Args:
            symbol: Symbol to predict
            exchange: Exchange name
            interval: Data interval
            include_features: Whether to include features in output
        
        Returns:
            Prediction results
        """
        start_time = time.time()
        
        try:
            # Get connector
            connector = self.connectors.get(exchange.lower())
            if not connector:
                raise ValueError(f"Unknown exchange: {exchange}")
            
            # Get latest market data
            logger.info(f"Fetching latest data for {symbol} from {exchange}")
            
            # Calculate date range (last 30 days)
            from datetime import datetime, timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            # Get historical data
            market_data = connector.get_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval=interval
            )
            
            if market_data.empty:
                raise ValueError(f"No data available for {symbol}")
            
            # Process features
            logger.info("Processing features")
            features_df = self.processor.create_model_ready_features(
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                include_news=True,
                start_date=start_date,
                end_date=end_date,
                save_to_file=False
            )
            
            if features_df.empty:
                raise ValueError(f"No features available for {symbol}")
            
            # Generate predictions
            logger.info("Generating predictions")
            
            if self.model_type == "transformer":
                # Prepare input data for transformer model
                feature_cols = [col for col in features_df.columns 
                              if col not in ["timestamp", "open", "high", "low", "close", "volume"]]
                
                X = features_df[feature_cols].values
                X = X.reshape(1, X.shape[0], X.shape[1])  # Add batch dimension
                
                # Generate predictions
                if hasattr(self.model, "predict_proba"):
                    probs = self.model.predict_proba(X)
                    prediction = np.argmax(probs, axis=1)[0]
                    confidence = float(probs[0, prediction])
                else:
                    prediction = int(self.model.predict(X)[0])
                    confidence = 0.7  # Default confidence
                
                # Map prediction to signal
                if prediction == 1:  # Buy signal
                    signal = "BUY"
                elif prediction == 0:  # Sell signal
                    signal = "SELL"
                else:
                    signal = "HOLD"
            
            elif self.model_type == "rl":
                # Prepare state for RL model
                feature_cols = [col for col in features_df.columns 
                              if col not in ["timestamp", "open", "high", "low", "close", "volume"]]
                
                state = features_df[feature_cols].values[-1]  # Latest state
                
                # Get action from PPO agent
                action, prob, _ = self.model.choose_action(state)
                
                # Map action to signal
                if action == 1:  # Buy signal
                    signal = "BUY"
                    confidence = float(prob)
                elif action == 2:  # Sell signal
                    signal = "SELL"
                    confidence = float(prob)
                else:  # Hold signal
                    signal = "HOLD"
                    confidence = float(prob)
            
            # Prepare result
            current_price = features_df["close"].iloc[-1]
            timestamp = features_df["timestamp"].iloc[-1] if "timestamp" in features_df.columns else datetime.now()
            
            result = {
                "symbol": symbol,
                "exchange": exchange,
                "signal": signal,
                "confidence": confidence,
                "price": float(current_price),
                "timestamp": timestamp.isoformat() if hasattr(timestamp, "isoformat") else timestamp,
                "model_type": self.model_type,
                "prediction_time": time.time() - start_time
            }
            
            # Include features if requested
            if include_features:
                result["features"] = {
                    col: float(features_df[col].iloc[-1]) 
                    for col in feature_cols[:20]  # Limit to first 20 features
                }
            
            return result
        
        except Exception as e:
            logger.exception(f"Error generating prediction: {str(e)}")
            
            return {
                "symbol": symbol,
                "exchange": exchange,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "prediction_time": time.time() - start_time
            }
    
    def predict_batch(self, 
                     symbols: List[str], 
                     exchange: str = "nse",
                     interval: str = "1d") -> List[Dict[str, Any]]:
        """
        Generate predictions for multiple symbols.
        
        Args:
            symbols: List of symbols to predict
            exchange: Exchange name
            interval: Data interval
        
        Returns:
            List of prediction results
        """
        results = []
        
        for symbol in symbols:
            try:
                result = self.predict(symbol, exchange, interval)
                results.append(result)
            except Exception as e:
                logger.exception(f"Error predicting {symbol}: {str(e)}")
                
                results.append({
                    "symbol": symbol,
                    "exchange": exchange,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        return results
    
    def benchmark(self, 
                symbol: str, 
                exchange: str = "nse",
                interval: str = "1d",
                num_iterations: int = 100) -> Dict[str, Any]:
        """
        Benchmark model inference performance.
        
        Args:
            symbol: Symbol to predict
            exchange: Exchange name
            interval: Data interval
            num_iterations: Number of inference iterations
        
        Returns:
            Benchmark results
        """
        try:
            # Get data for benchmarking
            result = self.predict(symbol, exchange, interval, include_features=True)
            
            if "error" in result:
                return {
                    "symbol": symbol,
                    "exchange": exchange,
                    "error": result["error"],
                    "timestamp": datetime.now().isoformat()
                }
            
            # Extract features for repeated inference
            feature_data = result.get("features", {})
            
            if not feature_data:
                return {
                    "symbol": symbol,
                    "exchange": exchange,
                    "error": "No features available for benchmarking",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Benchmark inference time
            logger.info(f"Running benchmark with {num_iterations} iterations")
            
            latencies = []
            
            for _ in range(num_iterations):
                if self.model_type == "transformer":
                    # Prepare input tensor
                    feature_values = list(feature_data.values())
                    X = torch.tensor(feature_values, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    
                    # Measure inference time
                    start_time = time.time()
                    
                    with torch.no_grad():
                        _ = self.model.model(X)
                    
                    latency = time.time() - start_time
                    latencies.append(latency)
                
                elif self.model_type == "rl":
                    # Prepare state
                    feature_values = list(feature_data.values())
                    state = np.array(feature_values)
                    
                    # Measure inference time
                    start_time = time.time()
                    
                    _ = self.model.choose_action(state)
                    
                    latency = time.time() - start_time
                    latencies.append(latency)
            
            # Calculate statistics
            avg_latency = np.mean(latencies) * 1000  # ms
            min_latency = np.min(latencies) * 1000  # ms
            max_latency = np.max(latencies) * 1000  # ms
            p95_latency = np.percentile(latencies, 95) * 1000  # ms
            p99_latency = np.percentile(latencies, 99) * 1000  # ms
            throughput = 1 / avg_latency * 1000  # inferences per second
            
            return {
                "symbol": symbol,
                "exchange": exchange,
                "model_type": self.model_type,
                "device": self.device,
                "use_tensorrt": self.use_tensorrt,
                "num_iterations": num_iterations,
                "avg_latency_ms": avg_latency,
                "min_latency_ms": min_latency,
                "max_latency_ms": max_latency,
                "p95_latency_ms": p95_latency,
                "p99_latency_ms": p99_latency,
                "throughput_fps": throughput,
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.exception(f"Error benchmarking model: {str(e)}")
            
            return {
                "symbol": symbol,
                "exchange": exchange,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


def main():
    """
    Main entry point for model serving.
    """
    parser = argparse.ArgumentParser(description="Serve model for inference")
    
    parser.add_argument("--model-path", type=str, required=True, help="Path to model file")
    parser.add_argument("--model-type", type=str, default="transformer", choices=["transformer", "rl"],
                      help="Type of model ('transformer' or 'rl')")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                      help="Device to use for inference")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--use-tensorrt", action="store_true", help="Use TensorRT for acceleration")
    parser.add_argument("--symbol", type=str, help="Symbol to predict")
    parser.add_argument("--exchange", type=str, default="nse", help="Exchange name")
    parser.add_argument("--interval", type=str, default="1d", help="Data interval")
    parser.add_argument("--symbols-file", type=str, help="Path to file with symbols list")
    parser.add_argument("--output-file", type=str, help="Path to output file")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--benchmark-iterations", type=int, default=100, help="Number of benchmark iterations")
    
    args = parser.parse_args()
    
    # Configure logging to file and console
    log_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
        "logs"
    )
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"serve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    try:
        # Initialize server
        logger.info("Initializing model server")
        server = ModelServer(
            model_path=args.model_path,
            model_type=args.model_type,
            device=args.device,
            batch_size=args.batch_size,
            use_tensorrt=args.use_tensorrt
        )
        
        if args.benchmark:
            # Run benchmark
            if args.symbol:
                # Benchmark single symbol
                result = server.benchmark(
                    symbol=args.symbol,
                    exchange=args.exchange,
                    interval=args.interval,
                    num_iterations=args.benchmark_iterations
                )
                
                print(json.dumps(result, indent=2))
                
                if args.output_file:
                    with open(args.output_file, "w") as f:
                        json.dump(result, f, indent=2)
            else:
                logger.error("Symbol required for benchmark")
        
        elif args.symbols_file:
            # Predict for multiple symbols
            try:
                with open(args.symbols_file, "r") as f:
                    symbols = [line.strip() for line in f.readlines()]
                
                logger.info(f"Predicting for {len(symbols)} symbols")
                results = server.predict_batch(symbols, args.exchange, args.interval)
                
                if args.output_file:
                    with open(args.output_file, "w") as f:
                        json.dump(results, f, indent=2)
                else:
                    print(json.dumps(results, indent=2))
            
            except Exception as e:
                logger.exception(f"Error reading symbols file: {str(e)}")
        
        elif args.symbol:
            # Predict for single symbol
            result = server.predict(
                symbol=args.symbol,
                exchange=args.exchange,
                interval=args.interval,
                include_features=True
            )
            
            if args.output_file:
                with open(args.output_file, "w") as f:
                    json.dump(result, f, indent=2)
            else:
                print(json.dumps(result, indent=2))
        
        else:
            logger.error("No symbol or symbols file specified")
    
    except Exception as e:
        logger.exception(f"Error in model serving: {str(e)}")


if __name__ == "__main__":
    main()

