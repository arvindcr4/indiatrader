"""
Command-line interface for IndiaTrader.
"""

import os
import sys
import logging
import argparse
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tabulate

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def setup_logging(verbose: bool = False):
    """
    Set up logging configuration.
    
    Args:
        verbose: Whether to enable verbose logging
    """
    # Create logs directory
    log_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "logs"
    )
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up log file
    log_file = os.path.join(log_dir, f"indiatrader_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Configure logging
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    root_logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(
        '%(levelname)s: %(message)s'
    ))
    root_logger.addHandler(console_handler)
    
    return log_file


def ingest_data(args):
    """
    Ingest market data.
    
    Args:
        args: Command-line arguments
    """
    from indiatrader.data.ingest import DataIngestionPipeline
    
    try:
        # Initialize ingestion pipeline
        pipeline = DataIngestionPipeline(args.config)
        
        # Run ingestion
        if args.type == "all":
            logger.info("Starting full data ingestion")
            stats = pipeline.run_full_ingestion()
        else:
            if args.type == "market":
                logger.info(f"Ingesting {args.exchange} market data")
                stats = pipeline.ingest_market_data(
                    exchange=args.exchange,
                    symbols=args.symbols.split(",") if args.symbols else None,
                    intervals=args.intervals.split(",") if args.intervals else None,
                    days_back=args.days
                )
            elif args.type == "news":
                logger.info("Ingesting news data")
                stats = pipeline.ingest_news_data(
                    sources=args.sources.split(",") if args.sources else None,
                    limit=args.limit
                )
            elif args.type == "social":
                logger.info(f"Ingesting {args.platform} data")
                stats = pipeline.ingest_social_media_data(args.platform)
        
        # Print statistics
        if isinstance(stats, dict):
            print("\nIngestion Statistics:")
            for key, value in stats.items():
                if key not in ["market_data_files", "news_data_files", "social_data_files", "errors"]:
                    print(f"{key}: {value}")
            
            print(f"\nFiles created: {stats.get('market_data_files', 0) + stats.get('news_data_files', 0) + stats.get('social_data_files', 0)}")
            print(f"Errors: {stats.get('errors', 0)}")
        
        logger.info("Data ingestion completed")
    
    except Exception as e:
        logger.exception(f"Error in data ingestion: {str(e)}")
        sys.exit(1)


def process_features(args):
    """
    Process features for a symbol.
    
    Args:
        args: Command-line arguments
    """
    from indiatrader.features.processor import FeatureProcessor
    
    try:
        # Initialize feature processor
        processor = FeatureProcessor(args.config)
        
        # Process features
        logger.info(f"Processing features for {args.symbol}")
        features = processor.create_model_ready_features(
            symbol=args.symbol,
            exchange=args.exchange,
            interval=args.interval,
            include_news=not args.no_news,
            start_date=args.start_date,
            end_date=args.end_date,
            target_horizon=args.horizon,
            save_to_file=True
        )
        
        # Print feature statistics
        if not features.empty:
            print(f"\nFeatures created for {args.symbol}:")
            print(f"Shape: {features.shape}")
            print(f"Date range: {features['timestamp'].min()} to {features['timestamp'].max()}")
            print(f"Number of features: {len(features.columns) - 1}")  # Excluding timestamp
            
            # Print feature groups
            tech_features = [col for col in features.columns if col.startswith(("sma", "ema", "rsi", "bb", "macd", "stoch", "atr"))]
            order_flow_features = [col for col in features.columns if col.startswith(("vwap", "delta_vol", "vp", "order"))]
            sentiment_features = [col for col in features.columns if col.startswith(("sentiment", "news"))]
            target_features = [col for col in features.columns if col.startswith("target")]
            
            print(f"\nFeature groups:")
            print(f"Technical indicators: {len(tech_features)}")
            print(f"Order flow features: {len(order_flow_features)}")
            print(f"Sentiment features: {len(sentiment_features)}")
            print(f"Target variables: {len(target_features)}")
        
        logger.info("Feature processing completed")
    
    except Exception as e:
        logger.exception(f"Error in feature processing: {str(e)}")
        sys.exit(1)


def train_model(args):
    """
    Train a model.
    
    Args:
        args: Command-line arguments
    """
    try:
        if args.model_type == "transformer":
            logger.info(f"Training {args.model_type} model for {args.symbol}")
            
            from indiatrader.models.transformer import run_patchtst_experiment
            from indiatrader.features.processor import FeatureProcessor
            
            # Load or process features
            processor = FeatureProcessor()
            
            # Create feature directory path
            feature_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "features", args.exchange, args.interval
            )
            
            # Check if features exist
            import glob
            feature_files = glob.glob(os.path.join(feature_dir, f"{args.symbol}_features_*.parquet"))
            
            if feature_files and not args.reprocess:
                # Use existing features
                latest_file = max(feature_files)
                logger.info(f"Using existing features from {latest_file}")
                features_df = pd.read_parquet(latest_file)
            else:
                # Process features
                logger.info("Processing features")
                features_df = processor.create_model_ready_features(
                    symbol=args.symbol,
                    exchange=args.exchange,
                    interval=args.interval,
                    include_news=not args.no_news,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    target_horizon=args.horizon,
                    save_to_file=True
                )
            
            if features_df.empty:
                logger.error("No features available for training")
                sys.exit(1)
            
            # Create models directory
            models_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "models", args.exchange, args.symbol
            )
            os.makedirs(models_dir, exist_ok=True)
            
            # Determine model path
            model_path = os.path.join(models_dir, f"{args.symbol}_{args.model_type}_{datetime.now().strftime('%Y%m%d')}")
            
            # Train model
            target_col = f"target_direction_{args.horizon}"
            
            if target_col not in features_df.columns:
                logger.error(f"Target column {target_col} not found in features")
                sys.exit(1)
            
            # Set patch parameters based on data size
            patch_len = min(16, features_df.shape[0] // 10)
            context_length = min(args.context_length, features_df.shape[0] - args.horizon)
            
            results = run_patchtst_experiment(
                data=features_df,
                target_col=target_col,
                model_type="classifier",
                test_size=0.2,
                val_size=0.1,
                context_length=context_length,
                patch_len=patch_len,
                stride=patch_len // 2,
                d_model=args.d_model,
                n_heads=args.n_heads,
                n_layers=args.n_layers,
                batch_size=args.batch_size,
                max_epochs=args.epochs,
                save_path=model_path
            )
            
            # Print training metrics
            print("\nTraining Results:")
            print(f"Training accuracy: {results['history'].get('val_acc', 0):.4f}")
            print(f"Training loss: {results['history'].get('val_loss', 0):.4f}")
            
            # Print test metrics
            print("\nTest Results:")
            for metric, value in results['test_results'].items():
                print(f"{metric}: {value:.4f}")
            
            print(f"\nModel saved to: {model_path}")
        
        elif args.model_type == "rl":
            logger.info(f"Training {args.model_type} model for {args.symbol}")
            
            from indiatrader.models.reinforcement import train_rl_model
            from indiatrader.features.processor import FeatureProcessor
            
            # Load or process features
            processor = FeatureProcessor()
            
            # Create feature directory path
            feature_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "features", args.exchange, args.interval
            )
            
            # Check if features exist
            import glob
            feature_files = glob.glob(os.path.join(feature_dir, f"{args.symbol}_features_*.parquet"))
            
            if feature_files and not args.reprocess:
                # Use existing features
                latest_file = max(feature_files)
                logger.info(f"Using existing features from {latest_file}")
                features_df = pd.read_parquet(latest_file)
            else:
                # Process features
                logger.info("Processing features")
                features_df = processor.create_model_ready_features(
                    symbol=args.symbol,
                    exchange=args.exchange,
                    interval=args.interval,
                    include_news=not args.no_news,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    target_horizon=args.horizon,
                    save_to_file=True
                )
            
            if features_df.empty:
                logger.error("No features available for training")
                sys.exit(1)
            
            # Create models directory
            models_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "models", args.exchange, args.symbol
            )
            os.makedirs(models_dir, exist_ok=True)
            
            # Determine model path
            model_path = os.path.join(models_dir, f"{args.symbol}_{args.model_type}_{datetime.now().strftime('%Y%m%d')}")
            
            # Get feature columns
            feature_cols = [col for col in features_df.columns 
                         if col not in ["timestamp", "open", "high", "low", "close", "volume"] and 
                         not col.startswith("target")]
            
            # Train model
            results = train_rl_model(
                data=features_df,
                window_size=args.context_length,
                initial_balance=args.initial_capital,
                commission=args.commission,
                features=feature_cols,
                lstm_units=args.lstm_units,
                lstm_layers=args.lstm_layers,
                max_episodes=args.episodes,
                save_path=model_path
            )
            
            # Print training metrics
            print("\nTraining Results:")
            print(f"Episodes: {args.episodes}")
            print(f"Final reward: {results['episode_rewards'][-1]:.4f}")
            print(f"Final profit: {results['episode_profits'][-1]:.4%}")
            print(f"Best reward: {results['best_reward']:.4f}")
            
            print(f"\nModel saved to: {model_path}")
        
        else:
            logger.error(f"Unknown model type: {args.model_type}")
            sys.exit(1)
        
        logger.info("Model training completed")
    
    except Exception as e:
        logger.exception(f"Error in model training: {str(e)}")
        sys.exit(1)


def backtest_strategy(args):
    """
    Backtest a trading strategy.
    
    Args:
        args: Command-line arguments
    """
    try:
        from indiatrader.backtesting.run import run_strategy_backtest
        
        # Parse strategy parameters
        if args.params:
            try:
                strategy_params = json.loads(args.params)
            except json.JSONDecodeError:
                logger.error("Invalid JSON format for strategy parameters")
                sys.exit(1)
        else:
            strategy_params = {}
        
        # Create output directory if specified
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
        
        # Run backtest
        logger.info(f"Running backtest for {args.symbol} with {args.strategy} strategy")
        results = run_strategy_backtest(
            symbol=args.symbol,
            strategy_name=args.strategy,
            exchange=args.exchange,
            interval=args.interval,
            start_date=args.start_date,
            end_date=args.end_date,
            initial_capital=args.initial_capital,
            strategy_params=strategy_params,
            commission_model=args.commission_model,
            position_sizing=args.position_sizing,
            use_features=not args.no_features,
            output_dir=args.output_dir
        )
        
        if "error" in results:
            logger.error(f"Backtest error: {results['error']}")
            sys.exit(1)
        
        # Print results
        print("\nBacktest Results:")
        print(f"Symbol: {args.symbol}")
        print(f"Strategy: {args.strategy}")
        print(f"Initial Capital: ${args.initial_capital:.2f}")
        print(f"Final Value: ${results['final_value']:.2f}")
        print(f"Profit: ${results['profit']:.2f} ({results['return_pct']:.2f}%)")
        
        # Print metrics
        print("\nPerformance Metrics:")
        for metric, value in results["metrics"].items():
            if isinstance(value, (int, float)):
                print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
        
        # Print trade statistics if available
        if "trades" in results and not results["trades"].empty:
            trades_df = results["trades"]
            print(f"\nTrades: {len(trades_df)}")
            print(f"Win Rate: {results['metrics'].get('win_rate', 0):.2f}%")
            print(f"Profit Factor: {results['metrics'].get('profit_factor', 0):.2f}")
            print(f"Average Winning Trade: {results['metrics'].get('avg_winning_trade', 0):.2f}%")
            print(f"Average Losing Trade: {results['metrics'].get('avg_losing_trade', 0):.2f}%")
        
        logger.info("Backtest completed")
    
    except Exception as e:
        logger.exception(f"Error in backtest: {str(e)}")
        sys.exit(1)


def optimize_strategy(args):
    """
    Optimize a trading strategy using walk-forward optimization.
    
    Args:
        args: Command-line arguments
    """
    try:
        from indiatrader.backtesting.run import run_walk_forward_optimization
        
        # Parse parameter grid
        try:
            param_grid = json.loads(args.param_grid)
        except json.JSONDecodeError:
            logger.error("Invalid JSON format for parameter grid")
            sys.exit(1)
        
        # Create output directory if specified
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
        
        # Run optimization
        logger.info(f"Running walk-forward optimization for {args.symbol} with {args.strategy} strategy")
        results = run_walk_forward_optimization(
            symbol=args.symbol,
            strategy_name=args.strategy,
            param_grid=param_grid,
            exchange=args.exchange,
            interval=args.interval,
            start_date=args.start_date,
            end_date=args.end_date,
            train_window=args.train_window,
            test_window=args.test_window,
            step_size=args.step_size,
            initial_capital=args.initial_capital,
            optimization_metric=args.optimization_metric,
            commission_model=args.commission_model,
            position_sizing=args.position_sizing,
            use_features=not args.no_features,
            output_dir=args.output_dir
        )
        
        if "error" in results:
            logger.error(f"Optimization error: {results['error']}")
            sys.exit(1)
        
        # Print results
        print("\nOptimization Results:")
        print(f"Symbol: {args.symbol}")
        print(f"Strategy: {args.strategy}")
        print(f"Optimization Metric: {args.optimization_metric}")
        
        # Print robust parameters
        print("\nRobust Parameters:")
        for param, value in results["robust_params"].items():
            print(f"  {param}: {value}")
        
        # Print parameter statistics
        print("\nParameter Statistics:")
        for param, stats in results["param_stats"].items():
            if "mean" in stats:
                print(f"  {param}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, min={stats['min']}, max={stats['max']}")
            else:
                print(f"  {param}: most_common={stats['most_common']}")
        
        # Print overall metrics
        print(f"\nOverall Train Metric: {results['overall_train_metric']:.4f}")
        print(f"Overall Test Metric: {results['overall_test_metric']:.4f}")
        print(f"Optimization Time: {results['elapsed_time']:.2f} seconds")
        
        # Print backtest results if available
        if "backtest_results" in results:
            backtest = results["backtest_results"]
            print("\nFull Backtest with Robust Parameters:")
            print(f"Initial Capital: ${args.initial_capital:.2f}")
            print(f"Final Value: ${backtest['final_value']:.2f}")
            print(f"Profit: ${backtest['profit']:.2f} ({backtest['return_pct']:.2f}%)")
            print(f"Sharpe Ratio: {backtest['metrics'].get('sharpe_ratio', 0):.2f}")
            print(f"Max Drawdown: {backtest['metrics'].get('max_drawdown', 0):.2f}%")
        
        logger.info("Optimization completed")
    
    except Exception as e:
        logger.exception(f"Error in optimization: {str(e)}")
        sys.exit(1)


def serve_model(args):
    """
    Serve a model for inference.
    
    Args:
        args: Command-line arguments
    """
    try:
        from indiatrader.deployment.serve import ModelServer
        
        # Initialize server
        logger.info(f"Initializing model server for {args.model_path}")
        server = ModelServer(
            model_path=args.model_path,
            model_type=args.model_type,
            device=args.device,
            batch_size=args.batch_size,
            use_tensorrt=args.use_tensorrt
        )
        
        if args.benchmark:
            # Run benchmark
            logger.info(f"Running benchmark with {args.benchmark_iterations} iterations")
            result = server.benchmark(
                symbol=args.symbol,
                exchange=args.exchange,
                interval=args.interval,
                num_iterations=args.benchmark_iterations
            )
            
            # Print benchmark results
            print("\nBenchmark Results:")
            for key, value in result.items():
                if key not in ["symbol", "exchange", "error"]:
                    if isinstance(value, (int, float)):
                        print(f"{key}: {value:.4f}")
                    else:
                        print(f"{key}: {value}")
            
            # Save results if output file is specified
            if args.output_file:
                with open(args.output_file, "w") as f:
                    json.dump(result, f, indent=2)
        
        elif args.symbols_file:
            # Predict for multiple symbols
            try:
                with open(args.symbols_file, "r") as f:
                    symbols = [line.strip() for line in f.readlines()]
                
                logger.info(f"Predicting for {len(symbols)} symbols")
                results = server.predict_batch(symbols, args.exchange, args.interval)
                
                # Print results in table format
                table_data = []
                for result in results:
                    if "error" not in result:
                        table_data.append([
                            result["symbol"],
                            result["signal"],
                            f"{result['confidence']:.4f}",
                            f"{result['price']:.2f}",
                            result["timestamp"]
                        ])
                    else:
                        table_data.append([
                            result["symbol"],
                            "ERROR",
                            "0.0000",
                            "0.00",
                            result["timestamp"]
                        ])
                
                headers = ["Symbol", "Signal", "Confidence", "Price", "Timestamp"]
                print("\nPrediction Results:")
                print(tabulate.tabulate(table_data, headers=headers, tablefmt="grid"))
                
                # Save results if output file is specified
                if args.output_file:
                    with open(args.output_file, "w") as f:
                        json.dump(results, f, indent=2)
            
            except Exception as e:
                logger.exception(f"Error reading symbols file: {str(e)}")
                sys.exit(1)
        
        else:
            # Predict for single symbol
            logger.info(f"Predicting for {args.symbol}")
            result = server.predict(
                symbol=args.symbol,
                exchange=args.exchange,
                interval=args.interval,
                include_features=True
            )
            
            # Print prediction result
            if "error" not in result:
                print("\nPrediction Result:")
                print(f"Symbol: {result['symbol']}")
                print(f"Exchange: {result['exchange']}")
                print(f"Signal: {result['signal']}")
                print(f"Confidence: {result['confidence']:.4f}")
                print(f"Price: {result['price']:.2f}")
                print(f"Timestamp: {result['timestamp']}")
                print(f"Model Type: {result['model_type']}")
                print(f"Prediction Time: {result['prediction_time']:.4f} seconds")
                
                if "features" in result:
                    print("\nTop Features:")
                    for feature, value in list(result["features"].items())[:10]:
                        print(f"  {feature}: {value:.4f}")
            else:
                print(f"\nError: {result['error']}")
            
            # Save result if output file is specified
            if args.output_file:
                with open(args.output_file, "w") as f:
                    json.dump(result, f, indent=2)
        
        logger.info("Model serving completed")
    
    except Exception as e:
        logger.exception(f"Error serving model: {str(e)}")
        sys.exit(1)


def start_server(args):
    """
    Start API server.
    
    Args:
        args: Command-line arguments
    """
    try:
        from indiatrader.deployment.server import main as server_main
        
        logger.info(f"Starting API server on port {args.port}")
        
        # Set environment variables for server
        os.environ["PORT"] = str(args.port)
        os.environ["HOST"] = args.host
        os.environ["LOG_LEVEL"] = "debug" if args.verbose else "info"
        
        # Start server
        server_main()
    
    except Exception as e:
        logger.exception(f"Error starting server: {str(e)}")
        sys.exit(1)


def main():
    """
    Main entry point for the CLI.
    """
    parser = argparse.ArgumentParser(description="IndiaTrader Command-Line Interface")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="IndiaTrader commands")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest market data")
    ingest_parser.add_argument("--type", choices=["market", "news", "social", "all"], default="all",
                             help="Type of data to ingest")
    ingest_parser.add_argument("--exchange", choices=["nse", "bse", "all"], default="all",
                             help="Exchange to ingest data from")
    ingest_parser.add_argument("--symbols", type=str, help="Comma-separated list of symbols to ingest")
    ingest_parser.add_argument("--intervals", type=str, default="1d",
                             help="Comma-separated list of intervals to ingest")
    ingest_parser.add_argument("--days", type=int, default=30, help="Number of days to ingest")
    ingest_parser.add_argument("--sources", type=str, help="Comma-separated list of news sources")
    ingest_parser.add_argument("--platform", choices=["twitter", "reddit"], default="twitter",
                             help="Social media platform to ingest from")
    ingest_parser.add_argument("--limit", type=int, default=100, help="Maximum number of items to ingest")
    ingest_parser.add_argument("--config", type=str, help="Path to configuration file")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process features for a symbol")
    process_parser.add_argument("--symbol", type=str, required=True, help="Symbol to process")
    process_parser.add_argument("--exchange", choices=["nse", "bse"], default="nse",
                              help="Exchange name")
    process_parser.add_argument("--interval", type=str, default="1d", help="Data interval")
    process_parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    process_parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    process_parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon")
    process_parser.add_argument("--no-news", action="store_true", help="Exclude news features")
    process_parser.add_argument("--config", type=str, help="Path to configuration file")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--symbol", type=str, required=True, help="Symbol to train model for")
    train_parser.add_argument("--exchange", choices=["nse", "bse"], default="nse",
                            help="Exchange name")
    train_parser.add_argument("--interval", type=str, default="1d", help="Data interval")
    train_parser.add_argument("--model-type", choices=["transformer", "rl"], default="transformer",
                            help="Type of model to train")
    train_parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    train_parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    train_parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon")
    train_parser.add_argument("--no-news", action="store_true", help="Exclude news features")
    train_parser.add_argument("--reprocess", action="store_true", 
                            help="Reprocess features even if they exist")
    train_parser.add_argument("--context-length", type=int, default=60,
                            help="Context length for sequence models")
    train_parser.add_argument("--initial-capital", type=float, default=100000.0,
                            help="Initial capital for RL models")
    train_parser.add_argument("--commission", type=float, default=0.001,
                            help="Commission rate for RL models")
    
    # Transformer-specific parameters
    train_parser.add_argument("--d-model", type=int, default=128,
                            help="Model dimension for transformer models")
    train_parser.add_argument("--n-heads", type=int, default=8,
                            help="Number of attention heads for transformer models")
    train_parser.add_argument("--n-layers", type=int, default=3,
                            help="Number of transformer layers")
    train_parser.add_argument("--batch-size", type=int, default=32,
                            help="Batch size for training")
    train_parser.add_argument("--epochs", type=int, default=100,
                            help="Maximum number of epochs for training")
    
    # RL-specific parameters
    train_parser.add_argument("--lstm-units", type=int, default=64,
                            help="Number of LSTM units for RL models")
    train_parser.add_argument("--lstm-layers", type=int, default=2,
                            help="Number of LSTM layers for RL models")
    train_parser.add_argument("--episodes", type=int, default=100,
                            help="Number of episodes for RL training")
    
    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Backtest a trading strategy")
    backtest_parser.add_argument("--symbol", type=str, required=True, help="Symbol to backtest")
    backtest_parser.add_argument("--strategy", type=str, required=True,
                               choices=["moving_average_crossover", "rsi", "bollinger_bands", "macd", 
                                       "breakout", "trend_following", "dual_momentum", "vwap", 
                                       "mean_reversion", "sentiment", "volatility_breakout", "combined"],
                               help="Strategy to backtest")
    backtest_parser.add_argument("--exchange", choices=["nse", "bse"], default="nse",
                               help="Exchange name")
    backtest_parser.add_argument("--interval", type=str, default="1d", help="Data interval")
    backtest_parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    backtest_parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    backtest_parser.add_argument("--initial-capital", type=float, default=100000.0,
                               help="Initial capital")
    backtest_parser.add_argument("--params", type=str, help="Strategy parameters as JSON string")
    backtest_parser.add_argument("--commission-model", type=str, default="percentage",
                               choices=["flat_fee", "percentage", "percentage_plus_fee", "indian_exchange"],
                               help="Transaction cost model to use")
    backtest_parser.add_argument("--position-sizing", type=str, default="fixed_size",
                               choices=["fixed_size", "fixed_risk", "kelly_criterion", "volatility_sizing"],
                               help="Position sizing method to use")
    backtest_parser.add_argument("--no-features", action="store_true",
                               help="Use raw market data instead of processed features")
    backtest_parser.add_argument("--output-dir", type=str, help="Directory to save results")
    
    # Optimize command
    optimize_parser = subparsers.add_parser("optimize", help="Optimize a trading strategy")
    optimize_parser.add_argument("--symbol", type=str, required=True, help="Symbol to optimize")
    optimize_parser.add_argument("--strategy", type=str, required=True,
                               choices=["moving_average_crossover", "rsi", "bollinger_bands", "macd", 
                                       "breakout", "trend_following", "dual_momentum", "vwap", 
                                       "mean_reversion", "sentiment", "volatility_breakout", "combined"],
                               help="Strategy to optimize")
    optimize_parser.add_argument("--param-grid", type=str, required=True,
                               help="Parameter grid as JSON string")
    optimize_parser.add_argument("--exchange", choices=["nse", "bse"], default="nse",
                               help="Exchange name")
    optimize_parser.add_argument("--interval", type=str, default="1d", help="Data interval")
    optimize_parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    optimize_parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    optimize_parser.add_argument("--train-window", type=int, default=252,
                               help="Number of bars for training window")
    optimize_parser.add_argument("--test-window", type=int, default=63,
                               help="Number of bars for test window")
    optimize_parser.add_argument("--step-size", type=int, default=63,
                               help="Number of bars to step forward between windows")
    optimize_parser.add_argument("--initial-capital", type=float, default=100000.0,
                               help="Initial capital")
    optimize_parser.add_argument("--optimization-metric", type=str, default="sharpe_ratio",
                               choices=["sharpe_ratio", "sortino_ratio", "calmar_ratio", 
                                        "max_drawdown", "total_return", "win_rate"],
                               help="Metric to optimize")
    optimize_parser.add_argument("--commission-model", type=str, default="percentage",
                               choices=["flat_fee", "percentage", "percentage_plus_fee", "indian_exchange"],
                               help="Transaction cost model to use")
    optimize_parser.add_argument("--position-sizing", type=str, default="fixed_size",
                               choices=["fixed_size", "fixed_risk", "kelly_criterion", "volatility_sizing"],
                               help="Position sizing method to use")
    optimize_parser.add_argument("--no-features", action="store_true",
                               help="Use raw market data instead of processed features")
    optimize_parser.add_argument("--output-dir", type=str, help="Directory to save results")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Serve a model for inference")
    serve_parser.add_argument("--model-path", type=str, required=True, help="Path to model file")
    serve_parser.add_argument("--model-type", choices=["transformer", "rl"], default="transformer",
                            help="Type of model")
    serve_parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                            help="Device to use for inference")
    serve_parser.add_argument("--batch-size", type=int, default=16, help="Batch size for inference")
    serve_parser.add_argument("--use-tensorrt", action="store_true", 
                            help="Use TensorRT for acceleration")
    serve_parser.add_argument("--symbol", type=str, help="Symbol to predict")
    serve_parser.add_argument("--exchange", choices=["nse", "bse"], default="nse",
                            help="Exchange name")
    serve_parser.add_argument("--interval", type=str, default="1d", help="Data interval")
    serve_parser.add_argument("--symbols-file", type=str, help="Path to file with symbols list")
    serve_parser.add_argument("--output-file", type=str, help="Path to output file")
    serve_parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    serve_parser.add_argument("--benchmark-iterations", type=int, default=100,
                            help="Number of benchmark iterations")
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Start API server")
    server_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    server_parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logging
    log_file = setup_logging(args.verbose)
    
    # Execute command
    if args.command == "ingest":
        ingest_data(args)
    elif args.command == "process":
        process_features(args)
    elif args.command == "train":
        train_model(args)
    elif args.command == "backtest":
        backtest_strategy(args)
    elif args.command == "optimize":
        optimize_strategy(args)
    elif args.command == "serve":
        serve_model(args)
    elif args.command == "server":
        start_server(args)
    else:
        parser.print_help()
        sys.exit(1)
    
    logger.info(f"Command '{args.command}' completed")
    logger.info(f"Log file: {log_file}")


if __name__ == "__main__":
    main()