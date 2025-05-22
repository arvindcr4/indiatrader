"""
Run backtesting evaluations.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

from indiatrader.data.config import load_config
from indiatrader.features.processor import FeatureProcessor
from indiatrader.backtesting.engine import (
    BacktestEngine, 
    WalkForwardOptimizer, 
    MultiStrategyBacktest,
    TransactionCostModel,
    PositionSizer
)
from indiatrader.backtesting.strategies import (
    moving_average_crossover,
    rsi_strategy,
    bollinger_band_strategy,
    macd_strategy,
    breakout_strategy,
    trend_following_strategy,
    dual_momentum_strategy,
    vwap_strategy,
    mean_reversion_with_zscore,
    sentiment_based_strategy,
    volatility_breakout_strategy,
    combined_strategy
)

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def load_data(symbol: str, 
             exchange: str = "nse",
             interval: str = "1d",
             start_date: Optional[str] = None,
             end_date: Optional[str] = None,
             use_features: bool = True) -> pd.DataFrame:
    """
    Load market data for backtesting.
    
    Args:
        symbol: Symbol to load
        exchange: Exchange name
        interval: Data interval
        start_date: Start date for filtering
        end_date: End date for filtering
        use_features: Whether to use processed features
    
    Returns:
        DataFrame with market data
    """
    if use_features:
        # Load feature-processed data
        feature_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "features", exchange, interval
        )
        
        # Find feature files for the symbol
        import glob
        feature_files = glob.glob(os.path.join(feature_dir, f"{symbol}_features_*.parquet"))
        
        if not feature_files:
            logger.warning(f"No feature files found for {symbol}. Processing features...")
            
            # Process features
            processor = FeatureProcessor()
            features_df = processor.create_model_ready_features(
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                include_news=True,
                start_date=start_date,
                end_date=end_date,
                save_to_file=True
            )
            
            return features_df
        else:
            # Use the most recent feature file
            latest_file = max(feature_files)
            logger.info(f"Loading features from {latest_file}")
            
            features_df = pd.read_parquet(latest_file)
            
            # Filter by date if needed
            if (start_date or end_date) and "timestamp" in features_df.columns:
                features_df["timestamp"] = pd.to_datetime(features_df["timestamp"])
                
                if start_date:
                    start_date = pd.to_datetime(start_date)
                    features_df = features_df[features_df["timestamp"] >= start_date]
                
                if end_date:
                    end_date = pd.to_datetime(end_date)
                    features_df = features_df[features_df["timestamp"] <= end_date]
            
            return features_df
    else:
        # Load raw market data
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", exchange, "ohlcv", interval
        )
        
        # Find data files for the symbol
        import glob
        data_files = glob.glob(os.path.join(data_dir, f"{symbol}_*.parquet"))
        
        if not data_files:
            logger.error(f"No data files found for {symbol}")
            return pd.DataFrame()
        
        # Load and combine all files
        dfs = []
        
        for file_path in data_files:
            try:
                df = pd.read_parquet(file_path)
                dfs.append(df)
            except Exception as e:
                logger.error(f"Failed to read Parquet file {file_path}: {str(e)}")
        
        if not dfs:
            logger.error(f"Failed to load any data for {symbol}")
            return pd.DataFrame()
        
        # Combine dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Remove duplicates
        if "timestamp" in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset=["timestamp"])
            combined_df = combined_df.sort_values("timestamp")
        
        # Filter by date if needed
        if (start_date or end_date) and "timestamp" in combined_df.columns:
            combined_df["timestamp"] = pd.to_datetime(combined_df["timestamp"])
            
            if start_date:
                start_date = pd.to_datetime(start_date)
                combined_df = combined_df[combined_df["timestamp"] >= start_date]
            
            if end_date:
                end_date = pd.to_datetime(end_date)
                combined_df = combined_df[combined_df["timestamp"] <= end_date]
        
        return combined_df


def run_strategy_backtest(
    symbol: str,
    strategy_name: str,
    exchange: str = "nse",
    interval: str = "1d",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    initial_capital: float = 100000.0,
    strategy_params: Optional[Dict[str, Any]] = None,
    commission_model: str = "percentage",
    position_sizing: str = "fixed_size",
    use_features: bool = True,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run backtest for a specific strategy.
    
    Args:
        symbol: Symbol to backtest
        strategy_name: Name of strategy to use
        exchange: Exchange name
        interval: Data interval
        start_date: Start date for backtest
        end_date: End date for backtest
        initial_capital: Initial capital
        strategy_params: Strategy parameters
        commission_model: Transaction cost model to use
        position_sizing: Position sizing method to use
        use_features: Whether to use feature-processed data
        output_dir: Directory to save results
    
    Returns:
        Dictionary of backtest results
    """
    # Load data
    data = load_data(
        symbol=symbol,
        exchange=exchange,
        interval=interval,
        start_date=start_date,
        end_date=end_date,
        use_features=use_features
    )
    
    if data.empty:
        logger.error(f"No data available for {symbol}")
        return {"error": "No data available"}
    
    # Get strategy function
    strategy_funcs = {
        "moving_average_crossover": moving_average_crossover,
        "rsi": rsi_strategy,
        "bollinger_bands": bollinger_band_strategy,
        "macd": macd_strategy,
        "breakout": breakout_strategy,
        "trend_following": trend_following_strategy,
        "dual_momentum": dual_momentum_strategy,
        "vwap": vwap_strategy,
        "mean_reversion": mean_reversion_with_zscore,
        "sentiment": sentiment_based_strategy,
        "volatility_breakout": volatility_breakout_strategy,
        "combined": combined_strategy
    }
    
    if strategy_name not in strategy_funcs:
        logger.error(f"Unknown strategy: {strategy_name}")
        return {"error": f"Unknown strategy: {strategy_name}"}
    
    strategy_func = strategy_funcs[strategy_name]
    
    # Set default parameters if not provided
    if strategy_params is None:
        strategy_params = {}
    
    # Select commission model
    commission_models = {
        "flat_fee": TransactionCostModel.flat_fee,
        "percentage": TransactionCostModel.percentage,
        "percentage_plus_fee": TransactionCostModel.percentage_plus_fee,
        "indian_exchange": TransactionCostModel.indian_exchange
    }
    
    selected_commission_model = commission_models.get(commission_model, TransactionCostModel.percentage)
    
    # Select position sizing method
    position_sizing_methods = {
        "fixed_size": PositionSizer.fixed_size,
        "fixed_risk": PositionSizer.fixed_risk,
        "kelly_criterion": PositionSizer.kelly_criterion,
        "volatility_sizing": PositionSizer.volatility_sizing
    }
    
    selected_position_sizing = position_sizing_methods.get(position_sizing, PositionSizer.fixed_size)
    
    # Create backtest engine
    engine = BacktestEngine(
        data=data,
        initial_capital=initial_capital,
        commission_model=selected_commission_model,
        position_sizing=selected_position_sizing
    )
    
    # Run backtest
    results = engine.run(
        strategy=strategy_func,
        start_date=start_date,
        end_date=end_date,
        strategy_params=strategy_params
    )
    
    # Plot and save results if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{symbol}_{strategy_name}")
        engine.plot_results(results, output_path)
        engine.save_results(results, output_path)
    
    return results


def run_walk_forward_optimization(
    symbol: str,
    strategy_name: str,
    param_grid: Dict[str, List[Any]],
    exchange: str = "nse",
    interval: str = "1d",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    train_window: int = 252,
    test_window: int = 63,
    step_size: int = 63,
    initial_capital: float = 100000.0,
    optimization_metric: str = "sharpe_ratio",
    commission_model: str = "percentage",
    position_sizing: str = "fixed_size",
    use_features: bool = True,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run walk-forward optimization for a strategy.
    
    Args:
        symbol: Symbol to backtest
        strategy_name: Name of strategy to use
        param_grid: Dictionary of parameter names and values to try
        exchange: Exchange name
        interval: Data interval
        start_date: Start date for backtest
        end_date: End date for backtest
        train_window: Number of bars for training window
        test_window: Number of bars for test window
        step_size: Number of bars to step forward between windows
        initial_capital: Initial capital
        optimization_metric: Metric to optimize
        commission_model: Transaction cost model to use
        position_sizing: Position sizing method to use
        use_features: Whether to use feature-processed data
        output_dir: Directory to save results
    
    Returns:
        Dictionary of optimization results
    """
    # Load data
    data = load_data(
        symbol=symbol,
        exchange=exchange,
        interval=interval,
        start_date=start_date,
        end_date=end_date,
        use_features=use_features
    )
    
    if data.empty:
        logger.error(f"No data available for {symbol}")
        return {"error": "No data available"}
    
    # Get strategy function
    strategy_funcs = {
        "moving_average_crossover": moving_average_crossover,
        "rsi": rsi_strategy,
        "bollinger_bands": bollinger_band_strategy,
        "macd": macd_strategy,
        "breakout": breakout_strategy,
        "trend_following": trend_following_strategy,
        "dual_momentum": dual_momentum_strategy,
        "vwap": vwap_strategy,
        "mean_reversion": mean_reversion_with_zscore,
        "sentiment": sentiment_based_strategy,
        "volatility_breakout": volatility_breakout_strategy,
        "combined": combined_strategy
    }
    
    if strategy_name not in strategy_funcs:
        logger.error(f"Unknown strategy: {strategy_name}")
        return {"error": f"Unknown strategy: {strategy_name}"}
    
    strategy_func = strategy_funcs[strategy_name]
    
    # Select commission model
    commission_models = {
        "flat_fee": TransactionCostModel.flat_fee,
        "percentage": TransactionCostModel.percentage,
        "percentage_plus_fee": TransactionCostModel.percentage_plus_fee,
        "indian_exchange": TransactionCostModel.indian_exchange
    }
    
    selected_commission_model = commission_models.get(commission_model, TransactionCostModel.percentage)
    
    # Select position sizing method
    position_sizing_methods = {
        "fixed_size": PositionSizer.fixed_size,
        "fixed_risk": PositionSizer.fixed_risk,
        "kelly_criterion": PositionSizer.kelly_criterion,
        "volatility_sizing": PositionSizer.volatility_sizing
    }
    
    selected_position_sizing = position_sizing_methods.get(position_sizing, PositionSizer.fixed_size)
    
    # Create walk-forward optimizer
    optimizer = WalkForwardOptimizer(
        data=data,
        train_window=train_window,
        test_window=test_window,
        step_size=step_size,
        initial_capital=initial_capital,
        commission_model=selected_commission_model,
        position_sizing=selected_position_sizing
    )
    
    # Run optimization
    optimization_results = optimizer.optimize(
        strategy=strategy_func,
        param_grid=param_grid,
        optimization_metric=optimization_metric
    )
    
    # Run backtest with robust parameters
    if "error" not in optimization_results:
        backtest_results = optimizer.run_robust_backtest(
            strategy=strategy_func,
            robust_params=optimization_results["robust_params"],
            plot_results=True,
            save_path=os.path.join(output_dir, f"{symbol}_{strategy_name}_wfo") if output_dir else None
        )
        
        optimization_results["backtest_results"] = backtest_results
    
    return optimization_results


def run_multi_strategy_backtest(
    symbol: str,
    strategies: List[str],
    exchange: str = "nse",
    interval: str = "1d",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    initial_capital: float = 100000.0,
    weights: Optional[Dict[str, float]] = None,
    optimize_weights: bool = False,
    strategy_params: Optional[Dict[str, Dict[str, Any]]] = None,
    commission_model: str = "percentage",
    position_sizing: str = "fixed_size",
    use_features: bool = True,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run backtest with multiple strategies.
    
    Args:
        symbol: Symbol to backtest
        strategies: List of strategy names to use
        exchange: Exchange name
        interval: Data interval
        start_date: Start date for backtest
        end_date: End date for backtest
        initial_capital: Initial capital
        weights: Dictionary of strategy names and weights
        optimize_weights: Whether to optimize weights
        strategy_params: Dictionary of strategy names and parameters
        commission_model: Transaction cost model to use
        position_sizing: Position sizing method to use
        use_features: Whether to use feature-processed data
        output_dir: Directory to save results
    
    Returns:
        Dictionary of backtest results
    """
    # Load data
    data = load_data(
        symbol=symbol,
        exchange=exchange,
        interval=interval,
        start_date=start_date,
        end_date=end_date,
        use_features=use_features
    )
    
    if data.empty:
        logger.error(f"No data available for {symbol}")
        return {"error": "No data available"}
    
    # Get strategy functions
    strategy_funcs = {
        "moving_average_crossover": moving_average_crossover,
        "rsi": rsi_strategy,
        "bollinger_bands": bollinger_band_strategy,
        "macd": macd_strategy,
        "breakout": breakout_strategy,
        "trend_following": trend_following_strategy,
        "dual_momentum": dual_momentum_strategy,
        "vwap": vwap_strategy,
        "mean_reversion": mean_reversion_with_zscore,
        "sentiment": sentiment_based_strategy,
        "volatility_breakout": volatility_breakout_strategy,
        "combined": combined_strategy
    }
    
    # Validate strategies
    invalid_strategies = [strat for strat in strategies if strat not in strategy_funcs]
    if invalid_strategies:
        logger.error(f"Unknown strategies: {invalid_strategies}")
        return {"error": f"Unknown strategies: {invalid_strategies}"}
    
    # Set default parameters if not provided
    if strategy_params is None:
        strategy_params = {}
    
    # Create strategies dictionary
    strategies_dict = {}
    for strat_name in strategies:
        strat_params = strategy_params.get(strat_name, {})
        strategies_dict[strat_name] = (strategy_funcs[strat_name], strat_params)
    
    # Set weights if not provided
    if weights is None:
        weights = {strat_name: 1.0 / len(strategies) for strat_name in strategies}
    
    # Select commission model
    commission_models = {
        "flat_fee": TransactionCostModel.flat_fee,
        "percentage": TransactionCostModel.percentage,
        "percentage_plus_fee": TransactionCostModel.percentage_plus_fee,
        "indian_exchange": TransactionCostModel.indian_exchange
    }
    
    selected_commission_model = commission_models.get(commission_model, TransactionCostModel.percentage)
    
    # Select position sizing method
    position_sizing_methods = {
        "fixed_size": PositionSizer.fixed_size,
        "fixed_risk": PositionSizer.fixed_risk,
        "kelly_criterion": PositionSizer.kelly_criterion,
        "volatility_sizing": PositionSizer.volatility_sizing
    }
    
    selected_position_sizing = position_sizing_methods.get(position_sizing, PositionSizer.fixed_size)
    
    # Create multi-strategy backtest
    multi_backtest = MultiStrategyBacktest(
        data=data,
        strategies=strategies_dict,
        weights=weights,
        initial_capital=initial_capital,
        commission_model=selected_commission_model,
        position_sizing=selected_position_sizing
    )
    
    # Optimize weights if requested
    if optimize_weights:
        optimization_results = multi_backtest.optimize_weights(
            metric="sharpe_ratio",
            method="grid_search" if len(strategies) <= 3 else "random_search"
        )
        
        weights = optimization_results["optimal_weights"]
        multi_backtest.weights = weights
    
    # Run backtest
    results = multi_backtest.run(
        start_date=start_date,
        end_date=end_date
    )
    
    # Plot and save results if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{symbol}_multi_strategy")
        multi_backtest.plot_results(results, output_path)
    
    return results


def main():
    """
    Main entry point for running backtests.
    """
    parser = argparse.ArgumentParser(description="Run trading strategy backtests")
    
    # Common arguments
    parser.add_argument("--symbol", type=str, required=True, help="Symbol to backtest")
    parser.add_argument("--exchange", type=str, default="nse", help="Exchange name")
    parser.add_argument("--interval", type=str, default="1d", help="Data interval")
    parser.add_argument("--start-date", type=str, help="Start date for backtest (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date for backtest (YYYY-MM-DD)")
    parser.add_argument("--initial-capital", type=float, default=100000.0, help="Initial capital")
    parser.add_argument("--commission-model", type=str, default="percentage", 
                       choices=["flat_fee", "percentage", "percentage_plus_fee", "indian_exchange"],
                       help="Transaction cost model to use")
    parser.add_argument("--position-sizing", type=str, default="fixed_size",
                       choices=["fixed_size", "fixed_risk", "kelly_criterion", "volatility_sizing"],
                       help="Position sizing method to use")
    parser.add_argument("--use-features", action="store_true", help="Use feature-processed data")
    parser.add_argument("--output-dir", type=str, help="Directory to save results")
    
    # Subparsers for different backtest types
    subparsers = parser.add_subparsers(dest="command", help="Backtest command")
    
    # Backtest a single strategy
    backtest_parser = subparsers.add_parser("backtest", help="Run backtest for a single strategy")
    backtest_parser.add_argument("--strategy", type=str, required=True, 
                              choices=["moving_average_crossover", "rsi", "bollinger_bands", "macd", 
                                      "breakout", "trend_following", "dual_momentum", "vwap", 
                                      "mean_reversion", "sentiment", "volatility_breakout", "combined"],
                              help="Strategy to backtest")
    backtest_parser.add_argument("--params", type=str, help="Strategy parameters as JSON string")
    
    # Walk-forward optimization
    optimize_parser = subparsers.add_parser("optimize", help="Run walk-forward optimization")
    optimize_parser.add_argument("--strategy", type=str, required=True,
                               choices=["moving_average_crossover", "rsi", "bollinger_bands", "macd", 
                                       "breakout", "trend_following", "dual_momentum", "vwap", 
                                       "mean_reversion", "sentiment", "volatility_breakout", "combined"],
                               help="Strategy to optimize")
    optimize_parser.add_argument("--param-grid", type=str, required=True, 
                               help="Parameter grid as JSON string")
    optimize_parser.add_argument("--train-window", type=int, default=252,
                               help="Number of bars for training window")
    optimize_parser.add_argument("--test-window", type=int, default=63,
                               help="Number of bars for test window")
    optimize_parser.add_argument("--step-size", type=int, default=63,
                               help="Number of bars to step forward between windows")
    optimize_parser.add_argument("--optimization-metric", type=str, default="sharpe_ratio",
                               choices=["sharpe_ratio", "sortino_ratio", "calmar_ratio", 
                                        "max_drawdown", "total_return", "win_rate"],
                               help="Metric to optimize")
    
    # Multi-strategy backtest
    multi_parser = subparsers.add_parser("multi-strategy", help="Run backtest with multiple strategies")
    multi_parser.add_argument("--strategies", type=str, required=True, 
                            help="Comma-separated list of strategies")
    multi_parser.add_argument("--weights", type=str, 
                            help="Strategy weights as JSON string")
    multi_parser.add_argument("--optimize-weights", action="store_true",
                            help="Optimize strategy weights")
    multi_parser.add_argument("--params", type=str,
                            help="Strategy parameters as JSON string")
    
    args = parser.parse_args()
    
    # Configure logging to file and console
    log_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
        "logs"
    )
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Parse JSON inputs
    import json
    
    # Create output directory if provided
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        if args.command == "backtest":
            # Parse strategy parameters
            strategy_params = json.loads(args.params) if args.params else None
            
            # Run backtest
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
                use_features=args.use_features,
                output_dir=args.output_dir
            )
            
            # Print summary
            if "error" not in results:
                print("\nBacktest Results:")
                print(f"Symbol: {args.symbol}")
                print(f"Strategy: {args.strategy}")
                print(f"Initial Capital: ${args.initial_capital:.2f}")
                print(f"Final Value: ${results['final_value']:.2f}")
                print(f"Profit: ${results['profit']:.2f} ({results['return_pct']:.2f}%)")
                print(f"Sharpe Ratio: {results['metrics'].get('sharpe_ratio', 0):.2f}")
                print(f"Max Drawdown: {results['metrics'].get('max_drawdown', 0):.2f}%")
                print(f"Number of Trades: {results['metrics'].get('num_trades', 0)}")
                print(f"Win Rate: {results['metrics'].get('win_rate', 0):.2f}%")
            else:
                print(f"Error: {results['error']}")
        
        elif args.command == "optimize":
            # Parse parameter grid
            param_grid = json.loads(args.param_grid)
            
            # Run optimization
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
                use_features=args.use_features,
                output_dir=args.output_dir
            )
            
            # Print summary
            if "error" not in results:
                print("\nOptimization Results:")
                print(f"Symbol: {args.symbol}")
                print(f"Strategy: {args.strategy}")
                print(f"Optimization Metric: {args.optimization_metric}")
                print("\nRobust Parameters:")
                for param, value in results["robust_params"].items():
                    print(f"  {param}: {value}")
                
                print("\nParameter Statistics:")
                for param, stats in results["param_stats"].items():
                    if "mean" in stats:
                        print(f"  {param}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, min={stats['min']}, max={stats['max']}")
                    else:
                        print(f"  {param}: most_common={stats['most_common']}")
                
                print(f"\nOverall Train Metric: {results['overall_train_metric']:.4f}")
                print(f"Overall Test Metric: {results['overall_test_metric']:.4f}")
                print(f"Optimization Time: {results['elapsed_time']:.2f} seconds")
                
                if "backtest_results" in results:
                    backtest = results["backtest_results"]
                    print("\nFull Backtest with Robust Parameters:")
                    print(f"Initial Capital: ${args.initial_capital:.2f}")
                    print(f"Final Value: ${backtest['final_value']:.2f}")
                    print(f"Profit: ${backtest['profit']:.2f} ({backtest['return_pct']:.2f}%)")
                    print(f"Sharpe Ratio: {backtest['metrics'].get('sharpe_ratio', 0):.2f}")
                    print(f"Max Drawdown: {backtest['metrics'].get('max_drawdown', 0):.2f}%")
            else:
                print(f"Error: {results['error']}")
        
        elif args.command == "multi-strategy":
            # Parse strategies list
            strategies = args.strategies.split(",")
            
            # Parse weights and parameters
            weights = json.loads(args.weights) if args.weights else None
            strategy_params = json.loads(args.params) if args.params else None
            
            # Run multi-strategy backtest
            results = run_multi_strategy_backtest(
                symbol=args.symbol,
                strategies=strategies,
                exchange=args.exchange,
                interval=args.interval,
                start_date=args.start_date,
                end_date=args.end_date,
                initial_capital=args.initial_capital,
                weights=weights,
                optimize_weights=args.optimize_weights,
                strategy_params=strategy_params,
                commission_model=args.commission_model,
                position_sizing=args.position_sizing,
                use_features=args.use_features,
                output_dir=args.output_dir
            )
            
            # Print summary
            if "error" not in results:
                print("\nMulti-Strategy Backtest Results:")
                print(f"Symbol: {args.symbol}")
                print(f"Strategies: {', '.join(strategies)}")
                print("\nStrategy Weights:")
                for strat, weight in results["weights"].items():
                    print(f"  {strat}: {weight:.2f}")
                
                print(f"\nInitial Capital: ${args.initial_capital:.2f}")
                print(f"Final Value: ${results['final_value']:.2f}")
                print(f"Profit: ${results['profit']:.2f} ({results['return_pct']:.2f}%)")
                print(f"Sharpe Ratio: {results['metrics'].get('sharpe_ratio', 0):.2f}")
                print(f"Sortino Ratio: {results['metrics'].get('sortino_ratio', 0):.2f}")
                print(f"Max Drawdown: {results['metrics'].get('max_drawdown', 0):.2f}%")
            else:
                print(f"Error: {results['error']}")
        
        else:
            parser.print_help()
    
    except Exception as e:
        logger.exception(f"Error in backtest: {str(e)}")
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()