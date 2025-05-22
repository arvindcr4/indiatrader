"""
Backtesting engine for trading strategies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import json
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from datetime import datetime, timedelta
import multiprocessing as mp
from functools import partial

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class PositionSizer:
    """
    Position sizing strategies for backtesting.
    """
    
    @staticmethod
    def fixed_size(context: Dict[str, Any], data: pd.DataFrame, percent: float = 1.0) -> float:
        """
        Fixed position size as a percentage of available capital.
        
        Args:
            context: Trading context dictionary
            data: Market data DataFrame
            percent: Percentage of available capital to use
        
        Returns:
            Position size in currency
        """
        return context["capital"] * percent
    
    @staticmethod
    def fixed_risk(context: Dict[str, Any], 
                 data: pd.DataFrame, 
                 risk_pct: float = 0.01, 
                 stop_loss_pct: float = 0.02) -> float:
        """
        Position size based on fixed risk percentage.
        
        Args:
            context: Trading context dictionary
            data: Market data DataFrame
            risk_pct: Percentage of capital to risk
            stop_loss_pct: Stop loss percentage
        
        Returns:
            Position size in currency
        """
        # Calculate risk amount
        risk_amount = context["capital"] * risk_pct
        
        # Calculate position size based on stop loss
        current_price = data.iloc[-1]["close"]
        stop_loss_amount = current_price * stop_loss_pct
        
        # Position size = risk amount / stop loss amount per share
        position_size = risk_amount / stop_loss_amount * current_price
        
        return min(position_size, context["capital"])
    
    @staticmethod
    def kelly_criterion(context: Dict[str, Any], 
                      data: pd.DataFrame, 
                      win_rate: float, 
                      win_loss_ratio: float,
                      max_size: float = 0.25) -> float:
        """
        Position size based on Kelly Criterion.
        
        Args:
            context: Trading context dictionary
            data: Market data DataFrame
            win_rate: Expected win rate (0.0 to 1.0)
            win_loss_ratio: Ratio of average win to average loss
            max_size: Maximum position size as fraction of capital
        
        Returns:
            Position size in currency
        """
        # Calculate Kelly fraction
        kelly_fraction = win_rate - (1 - win_rate) / win_loss_ratio
        
        # Apply half-Kelly for more conservative sizing
        kelly_fraction = kelly_fraction * 0.5
        
        # Limit maximum position size
        kelly_fraction = min(kelly_fraction, max_size)
        
        # Ensure positive position size
        kelly_fraction = max(0, kelly_fraction)
        
        return context["capital"] * kelly_fraction
    
    @staticmethod
    def volatility_sizing(context: Dict[str, Any], 
                        data: pd.DataFrame, 
                        atr_periods: int = 14,
                        volatility_factor: float = 1.0,
                        max_risk_pct: float = 0.02) -> float:
        """
        Position size based on volatility (ATR).
        
        Args:
            context: Trading context dictionary
            data: Market data DataFrame
            atr_periods: Number of periods for ATR calculation
            volatility_factor: Multiplier for ATR
            max_risk_pct: Maximum risk percentage
        
        Returns:
            Position size in currency
        """
        # Ensure we have enough data
        if len(data) < atr_periods:
            return context["capital"] * 0.01
        
        # Calculate ATR if not already in data
        if "atr" not in data.columns:
            # Calculate True Range
            high = data["high"].values
            low = data["low"].values
            close = data["close"].values
            
            tr1 = high[-atr_periods:] - low[-atr_periods:]
            tr2 = np.abs(high[-atr_periods:] - close[-atr_periods-1:-1])
            tr3 = np.abs(low[-atr_periods:] - close[-atr_periods-1:-1])
            
            tr = np.vstack([tr1, tr2, tr3]).max(axis=0)
            atr = np.mean(tr)
        else:
            atr = data.iloc[-1]["atr"]
        
        # Calculate position size based on ATR
        current_price = data.iloc[-1]["close"]
        risk_per_share = atr * volatility_factor
        risk_amount = context["capital"] * max_risk_pct
        
        # Position size = risk amount / risk per share
        position_size = risk_amount / risk_per_share * current_price
        
        return min(position_size, context["capital"])


class TransactionCostModel:
    """
    Transaction cost models for backtesting.
    """
    
    @staticmethod
    def flat_fee(context: Dict[str, Any], price: float, shares: float, cost_per_trade: float = 5.0) -> float:
        """
        Flat fee per trade transaction cost model.
        
        Args:
            context: Trading context dictionary
            price: Current price
            shares: Number of shares traded
            cost_per_trade: Flat fee per trade
        
        Returns:
            Transaction cost
        """
        return cost_per_trade
    
    @staticmethod
    def percentage(context: Dict[str, Any], price: float, shares: float, commission_pct: float = 0.001) -> float:
        """
        Percentage-based transaction cost model.
        
        Args:
            context: Trading context dictionary
            price: Current price
            shares: Number of shares traded
            commission_pct: Commission percentage
        
        Returns:
            Transaction cost
        """
        return price * shares * commission_pct
    
    @staticmethod
    def percentage_plus_fee(context: Dict[str, Any], price: float, shares: float, 
                          commission_pct: float = 0.0005, min_fee: float = 1.0, max_fee: float = 100.0) -> float:
        """
        Percentage-based transaction cost model with minimum and maximum fees.
        
        Args:
            context: Trading context dictionary
            price: Current price
            shares: Number of shares traded
            commission_pct: Commission percentage
            min_fee: Minimum fee per trade
            max_fee: Maximum fee per trade
        
        Returns:
            Transaction cost
        """
        cost = price * shares * commission_pct
        return max(min_fee, min(cost, max_fee))
    
    @staticmethod
    def indian_exchange(context: Dict[str, Any], price: float, shares: float, 
                      exchange: str = "nse",
                      brokerage_pct: float = 0.0003,
                      stt_pct: float = 0.00025,
                      exchange_charge_pct: float = 0.0000325,
                      sebi_charge_pct: float = 0.0000005,
                      stamp_duty_pct: float = 0.00003,
                      gst_pct: float = 0.18) -> float:
        """
        Indian exchange transaction cost model with all applicable charges.
        
        Args:
            context: Trading context dictionary
            price: Current price
            shares: Number of shares traded
            exchange: Exchange name ("nse" or "bse")
            brokerage_pct: Brokerage percentage
            stt_pct: Securities Transaction Tax percentage
            exchange_charge_pct: Exchange transaction charge percentage
            sebi_charge_pct: SEBI turnover fee percentage
            stamp_duty_pct: Stamp duty percentage
            gst_pct: GST percentage on (brokerage + exchange charges)
        
        Returns:
            Transaction cost
        """
        # Calculate trade value
        trade_value = price * shares
        
        # Calculate each component
        brokerage = trade_value * brokerage_pct
        
        # STT is applicable on sell trades only and on the selling price
        stt = trade_value * stt_pct if context.get("side") == "sell" else 0
        
        # Exchange transaction charges
        exchange_charge = trade_value * exchange_charge_pct
        
        # SEBI charges
        sebi_charge = trade_value * sebi_charge_pct
        
        # Stamp duty (applicable on buy trades only)
        stamp_duty = trade_value * stamp_duty_pct if context.get("side") == "buy" else 0
        
        # GST on brokerage and exchange charges
        gst = (brokerage + exchange_charge) * gst_pct
        
        # Total cost
        total_cost = brokerage + stt + exchange_charge + sebi_charge + stamp_duty + gst
        
        return total_cost
    
    @staticmethod
    def slippage_model(context: Dict[str, Any], price: float, shares: float, 
                     slippage_pct: float = 0.001, side: str = None) -> float:
        """
        Price slippage model to adjust execution price.
        
        Args:
            context: Trading context dictionary
            price: Current price
            shares: Number of shares traded
            slippage_pct: Slippage percentage
            side: Trade side ("buy" or "sell")
        
        Returns:
            Adjusted price after slippage
        """
        side = side or context.get("side", "buy")
        
        # Apply slippage based on trade direction
        if side == "buy":
            # Buy at higher price
            adjusted_price = price * (1 + slippage_pct)
        else:
            # Sell at lower price
            adjusted_price = price * (1 - slippage_pct)
        
        return adjusted_price


class BacktestEngine:
    """
    Engine for backtesting trading strategies.
    """
    
    def __init__(self, 
                data: pd.DataFrame,
                initial_capital: float = 100000.0,
                commission_model: Callable = TransactionCostModel.percentage,
                slippage_model: Callable = TransactionCostModel.slippage_model,
                position_sizing: Callable = PositionSizer.fixed_size,
                price_col: str = "close",
                date_col: str = "timestamp"):
        """
        Initialize backtest engine.
        
        Args:
            data: DataFrame with market data
            initial_capital: Initial capital
            commission_model: Transaction cost model function
            slippage_model: Slippage model function
            position_sizing: Position sizing function
            price_col: Name of price column
            date_col: Name of date column
        """
        # Validate data
        required_cols = [price_col, "high", "low", "volume"]
        for col in required_cols:
            if col not in data.columns:
                logger.warning(f"Required column '{col}' not found in data")
        
        # Store parameters
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.commission_model = commission_model
        self.slippage_model = slippage_model
        self.position_sizing = position_sizing
        self.price_col = price_col
        self.date_col = date_col
        
        # Ensure data is sorted
        if date_col in self.data.columns:
            self.data = self.data.sort_values(date_col)
        
        # Initialize trading context
        self.reset()
    
    def reset(self):
        """
        Reset trading context to initial state.
        """
        self.context = {
            "capital": self.initial_capital,
            "position": 0,
            "position_value": 0,
            "entry_price": 0,
            "entry_time": None,
            "trades": [],
            "equity_curve": [],
            "current_index": 0,
            "current_data": None
        }
    
    def run(self, 
           strategy: Callable,
           start_date: Optional[Union[str, datetime]] = None,
           end_date: Optional[Union[str, datetime]] = None,
           warmup_period: int = 20,
           strategy_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run backtest for a given strategy.
        
        Args:
            strategy: Strategy function that takes (context, data, params) and returns signals
            start_date: Start date for backtest
            end_date: End date for backtest
            warmup_period: Number of bars for warmup
            strategy_params: Parameters to pass to strategy
        
        Returns:
            Dictionary of backtest results
        """
        # Configure date range
        if self.date_col in self.data.columns:
            # Ensure datetime format
            if not pd.api.types.is_datetime64_any_dtype(self.data[self.date_col]):
                self.data[self.date_col] = pd.to_datetime(self.data[self.date_col])
            
            # Filter by date range
            if start_date is not None:
                if isinstance(start_date, str):
                    start_date = pd.to_datetime(start_date)
                self.data = self.data[self.data[self.date_col] >= start_date]
            
            if end_date is not None:
                if isinstance(end_date, str):
                    end_date = pd.to_datetime(end_date)
                self.data = self.data[self.data[self.date_col] <= end_date]
        
        # Ensure we have data
        if len(self.data) == 0:
            logger.error("No data available for the specified date range")
            return {"error": "No data available"}
        
        # Reset context
        self.reset()
        
        # Set up equity curve tracking
        equity_curve = []
        
        # Get strategy parameters
        params = strategy_params or {}
        
        # Run strategy
        for i in range(warmup_period, len(self.data)):
            # Update context
            self.context["current_index"] = i
            self.context["current_data"] = self.data.iloc[:i+1]
            
            # Calculate current portfolio value
            current_price = self.data.iloc[i][self.price_col]
            self.context["position_value"] = self.context["position"] * current_price
            portfolio_value = self.context["capital"] + self.context["position_value"]
            
            # Record equity curve
            equity_curve.append({
                "timestamp": self.data.iloc[i][self.date_col] if self.date_col in self.data.columns else i,
                "portfolio_value": portfolio_value,
                "capital": self.context["capital"],
                "position_value": self.context["position_value"],
                "price": current_price
            })
            
            # Execute strategy
            signal = strategy(self.context, self.data.iloc[:i+1], params)
            
            # Process signals
            if signal:
                self._process_signal(signal, i)
        
        # Calculate final portfolio value
        last_price = self.data.iloc[-1][self.price_col]
        final_position_value = self.context["position"] * last_price
        final_value = self.context["capital"] + final_position_value
        
        # Create equity curve DataFrame
        self.equity_curve = pd.DataFrame(equity_curve)
        
        # Calculate performance metrics
        metrics = self._calculate_metrics()
        
        # Return results
        results = {
            "equity_curve": self.equity_curve,
            "trades": pd.DataFrame(self.context["trades"]) if self.context["trades"] else pd.DataFrame(),
            "metrics": metrics,
            "initial_capital": self.initial_capital,
            "final_value": final_value,
            "profit": final_value - self.initial_capital,
            "return_pct": (final_value / self.initial_capital - 1) * 100
        }
        
        return results
    
    def _process_signal(self, signal: Dict[str, Any], index: int):
        """
        Process trading signal.
        
        Args:
            signal: Signal dictionary
            index: Current data index
        """
        # Extract signal info
        direction = signal.get("direction", 0)  # 1 for buy, -1 for sell, 0 for hold
        side = "buy" if direction > 0 else "sell" if direction < 0 else "hold"
        close_position = signal.get("close_position", False)
        
        # Current price and time
        current_price = self.data.iloc[index][self.price_col]
        current_time = self.data.iloc[index][self.date_col] if self.date_col in self.data.columns else index
        
        # No action for hold signal
        if direction == 0 and not close_position:
            return
        
        # Close existing position if requested or direction is opposite
        if (close_position or 
            (direction > 0 and self.context["position"] < 0) or 
            (direction < 0 and self.context["position"] > 0)):
            
            # Skip if no position
            if self.context["position"] == 0:
                return
            
            # Set up exit context
            exit_context = self.context.copy()
            exit_context["side"] = "sell" if self.context["position"] > 0 else "buy"
            
            # Apply slippage to exit price
            exit_price = self.slippage_model(
                exit_context, 
                current_price, 
                abs(self.context["position"]),
                slippage_pct=signal.get("slippage_pct", 0.001),
                side=exit_context["side"]
            )
            
            # Calculate exit value
            exit_value = abs(self.context["position"]) * exit_price
            
            # Calculate exit commission
            exit_commission = self.commission_model(
                exit_context,
                exit_price,
                abs(self.context["position"]),
                **signal.get("commission_params", {})
            )
            
            # Update capital
            if self.context["position"] > 0:
                # Selling long position
                self.context["capital"] += exit_value - exit_commission
            else:
                # Covering short position
                self.context["capital"] -= exit_value + exit_commission
            
            # Calculate profit/loss
            if self.context["position"] > 0:
                # Long position
                pnl = exit_value - self.context["entry_price"] * self.context["position"] - exit_commission
                pnl_pct = (exit_price / self.context["entry_price"] - 1) * 100
            else:
                # Short position
                pnl = self.context["entry_price"] * abs(self.context["position"]) - exit_value - exit_commission
                pnl_pct = (1 - exit_price / self.context["entry_price"]) * 100
            
            # Record trade
            trade = {
                "entry_time": self.context["entry_time"],
                "entry_price": self.context["entry_price"],
                "exit_time": current_time,
                "exit_price": exit_price,
                "position": self.context["position"],
                "direction": "long" if self.context["position"] > 0 else "short",
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "exit_commission": exit_commission
            }
            
            self.context["trades"].append(trade)
            
            # Reset position
            self.context["position"] = 0
            self.context["position_value"] = 0
            self.context["entry_price"] = 0
            self.context["entry_time"] = None
        
        # Open new position if direction is non-zero
        if direction != 0:
            # Set up entry context
            entry_context = self.context.copy()
            entry_context["side"] = side
            
            # Determine position size
            position_value = self.position_sizing(
                entry_context,
                self.data.iloc[:index+1],
                **signal.get("sizing_params", {})
            )
            
            # Apply slippage to entry price
            entry_price = self.slippage_model(
                entry_context,
                current_price,
                position_value / current_price,
                slippage_pct=signal.get("slippage_pct", 0.001),
                side=side
            )
            
            # Calculate position size in shares
            shares = position_value / entry_price
            
            # Calculate entry commission
            entry_commission = self.commission_model(
                entry_context,
                entry_price,
                shares,
                **signal.get("commission_params", {})
            )
            
            # Check if we have enough capital
            required_capital = position_value + entry_commission
            
            if required_capital > self.context["capital"]:
                # Scale down position if not enough capital
                scale_factor = self.context["capital"] / required_capital
                shares *= scale_factor
                position_value *= scale_factor
                entry_commission = self.commission_model(
                    entry_context,
                    entry_price,
                    shares,
                    **signal.get("commission_params", {})
                )
            
            # Update position and capital
            if direction > 0:
                # Long position
                self.context["position"] = shares
                self.context["capital"] -= position_value + entry_commission
            else:
                # Short position
                self.context["position"] = -shares
                self.context["capital"] += position_value - entry_commission
            
            # Record entry
            self.context["entry_price"] = entry_price
            self.context["entry_time"] = current_time
            self.context["position_value"] = shares * entry_price
    
    def _calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate backtest performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        # Check if we have equity curve
        if not hasattr(self, "equity_curve") or self.equity_curve.empty:
            return {}
        
        # Convert equity curve dates if needed
        if self.date_col in self.equity_curve.columns and not pd.api.types.is_datetime64_any_dtype(self.equity_curve[self.date_col]):
            self.equity_curve[self.date_col] = pd.to_datetime(self.equity_curve[self.date_col])
        
        # Calculate daily returns
        self.equity_curve["daily_return"] = self.equity_curve["portfolio_value"].pct_change()
        
        # Calculate cumulative returns
        self.equity_curve["cum_return"] = (1 + self.equity_curve["daily_return"]).cumprod() - 1
        
        # Calculate drawdown
        self.equity_curve["high_watermark"] = self.equity_curve["portfolio_value"].cummax()
        self.equity_curve["drawdown"] = (self.equity_curve["portfolio_value"] / self.equity_curve["high_watermark"] - 1) * 100
        
        # Trades metrics
        if not self.context["trades"]:
            num_trades = 0
            win_rate = 0
            avg_profit = 0
            avg_loss = 0
            profit_factor = 0
            avg_trade = 0
            avg_winning_trade = 0
            avg_losing_trade = 0
            max_consecutive_wins = 0
            max_consecutive_losses = 0
        else:
            trades_df = pd.DataFrame(self.context["trades"])
            num_trades = len(trades_df)
            winning_trades = trades_df[trades_df["pnl"] > 0]
            losing_trades = trades_df[trades_df["pnl"] <= 0]
            
            win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0
            avg_profit = winning_trades["pnl"].mean() if not winning_trades.empty else 0
            avg_loss = losing_trades["pnl"].mean() if not losing_trades.empty else 0
            
            total_profit = winning_trades["pnl"].sum() if not winning_trades.empty else 0
            total_loss = abs(losing_trades["pnl"].sum()) if not losing_trades.empty else 0
            
            profit_factor = total_profit / total_loss if total_loss != 0 else float('inf')
            avg_trade = trades_df["pnl"].mean()
            avg_winning_trade = winning_trades["pnl_pct"].mean() if not winning_trades.empty else 0
            avg_losing_trade = losing_trades["pnl_pct"].mean() if not losing_trades.empty else 0
            
            # Calculate consecutive wins/losses
            trades_df["win"] = trades_df["pnl"] > 0
            trades_df["streak"] = (trades_df["win"] != trades_df["win"].shift(1)).cumsum()
            
            streak_groups = trades_df.groupby(["streak", "win"]).size()
            
            max_consecutive_wins = streak_groups.xs(True, level=1).max() if True in streak_groups.index.get_level_values(1) else 0
            max_consecutive_losses = streak_groups.xs(False, level=1).max() if False in streak_groups.index.get_level_values(1) else 0
        
        # Calculate annualized metrics
        if self.date_col in self.equity_curve.columns:
            # Calculate trading days in test period
            start_date = self.equity_curve[self.date_col].min()
            end_date = self.equity_curve[self.date_col].max()
            days = (end_date - start_date).days
            years = days / 365
            
            # Annualized return
            total_return = self.equity_curve["portfolio_value"].iloc[-1] / self.equity_curve["portfolio_value"].iloc[0] - 1
            ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
            
            # Annualized volatility
            daily_std = self.equity_curve["daily_return"].std()
            ann_volatility = daily_std * np.sqrt(252)
            
            # Sharpe ratio
            risk_free_rate = 0.03  # Assumed risk-free rate
            excess_return = ann_return - risk_free_rate
            sharpe_ratio = excess_return / ann_volatility if ann_volatility != 0 else 0
            
            # Sortino ratio
            downside_returns = self.equity_curve["daily_return"][self.equity_curve["daily_return"] < 0]
            downside_std = downside_returns.std() if not downside_returns.empty else 0
            ann_downside_std = downside_std * np.sqrt(252)
            sortino_ratio = excess_return / ann_downside_std if ann_downside_std != 0 else 0
        else:
            # Use index-based calculations if no date column
            total_return = self.equity_curve["portfolio_value"].iloc[-1] / self.equity_curve["portfolio_value"].iloc[0] - 1
            ann_return = total_return  # Cannot annualize without dates
            ann_volatility = self.equity_curve["daily_return"].std() * np.sqrt(252)
            
            risk_free_rate = 0.03
            excess_return = ann_return - risk_free_rate
            sharpe_ratio = excess_return / ann_volatility if ann_volatility != 0 else 0
            
            downside_returns = self.equity_curve["daily_return"][self.equity_curve["daily_return"] < 0]
            downside_std = downside_returns.std() if not downside_returns.empty else 0
            ann_downside_std = downside_std * np.sqrt(252)
            sortino_ratio = excess_return / ann_downside_std if ann_downside_std != 0 else 0
        
        # Calculate max drawdown
        max_drawdown = self.equity_curve["drawdown"].min()
        
        # Calculate Calmar ratio
        calmar_ratio = ann_return / abs(max_drawdown / 100) if max_drawdown != 0 else 0
        
        # Calculate exposure
        exposure = (self.equity_curve["position_value"] != 0).mean() * 100
        
        # Compile metrics
        metrics = {
            "total_return": total_return * 100,
            "annualized_return": ann_return * 100,
            "annualized_volatility": ann_volatility * 100,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "max_drawdown": max_drawdown,
            "num_trades": num_trades,
            "win_rate": win_rate * 100,
            "profit_factor": profit_factor,
            "avg_trade": avg_trade,
            "avg_winning_trade": avg_winning_trade,
            "avg_losing_trade": avg_losing_trade,
            "max_consecutive_wins": max_consecutive_wins,
            "max_consecutive_losses": max_consecutive_losses,
            "exposure": exposure
        }
        
        return metrics
    
    def plot_results(self, results: Dict[str, Any], save_path: Optional[str] = None):
        """
        Plot backtest results.
        
        Args:
            results: Dictionary of backtest results
            save_path: Path to save the plots
        """
        # Set up matplotlib
        plt.style.use("seaborn-v0_8-darkgrid")
        plt.rcParams["figure.figsize"] = (12, 8)
        
        # Extract data
        equity_curve = results["equity_curve"]
        trades = results["trades"]
        metrics = results["metrics"]
        
        # Create figure for equity curve
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]}, figsize=(12, 8))
        
        # Plot equity curve
        ax1.plot(equity_curve["timestamp"], equity_curve["portfolio_value"], label="Portfolio Value")
        
        # Add buy/sell markers if trades are available
        if not trades.empty:
            for _, trade in trades.iterrows():
                if trade["direction"] == "long":
                    # Buy marker
                    ax1.plot(trade["entry_time"], trade["entry_price"] * trade["position"], 
                            "^", color="green", markersize=8)
                    # Sell marker
                    ax1.plot(trade["exit_time"], trade["exit_price"] * trade["position"], 
                            "v", color="red", markersize=8)
                else:
                    # Short marker
                    ax1.plot(trade["entry_time"], abs(trade["entry_price"] * trade["position"]), 
                            "v", color="red", markersize=8)
                    # Cover marker
                    ax1.plot(trade["exit_time"], abs(trade["exit_price"] * trade["position"]), 
                            "^", color="green", markersize=8)
        
        # Format equity curve plot
        ax1.set_title("Portfolio Equity Curve", fontsize=14)
        ax1.set_ylabel("Portfolio Value", fontsize=12)
        ax1.grid(True)
        ax1.legend()
        
        # Plot drawdown
        ax2.fill_between(equity_curve["timestamp"], 0, equity_curve["drawdown"], 
                        color="red", alpha=0.5)
        ax2.set_ylabel("Drawdown (%)", fontsize=12)
        ax2.set_xlabel("Date", fontsize=12)
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            plt.savefig(f"{save_path}_equity_curve.png", dpi=300, bbox_inches="tight")
        else:
            plt.show()
        
        # Create second figure for trade analysis
        if not trades.empty:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            # Plot trade P&L distribution
            sns.histplot(trades["pnl"], bins=20, kde=True, ax=ax1)
            ax1.set_title("Trade P&L Distribution", fontsize=14)
            ax1.set_xlabel("Profit/Loss", fontsize=12)
            ax1.axvline(0, color="black", linestyle="--")
            
            # Plot trade P&L by month
            if "entry_time" in trades.columns and pd.api.types.is_datetime64_any_dtype(trades["entry_time"]):
                trades["month"] = trades["entry_time"].dt.to_period("M")
                monthly_pnl = trades.groupby("month")["pnl"].sum()
                monthly_pnl.plot(kind="bar", ax=ax2)
                ax2.set_title("Monthly P&L", fontsize=14)
                ax2.set_xlabel("Month", fontsize=12)
                ax2.set_ylabel("Profit/Loss", fontsize=12)
                
                # Rotate x-axis labels
                plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
            
            plt.tight_layout()
            
            # Save or show the plot
            if save_path:
                plt.savefig(f"{save_path}_trade_analysis.png", dpi=300, bbox_inches="tight")
            else:
                plt.show()
        
        # Create third figure for performance metrics
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Key metrics to display
        key_metrics = [
            "total_return", "annualized_return", "max_drawdown", "sharpe_ratio",
            "win_rate", "profit_factor", "num_trades", "exposure"
        ]
        
        metric_values = [metrics.get(m, 0) for m in key_metrics]
        
        # Format metric names
        metric_names = [m.replace("_", " ").title() for m in key_metrics]
        
        # Create horizontal bar chart
        bars = ax.barh(metric_names, metric_values, color="skyblue")
        
        # Add values to bars
        for bar, value in zip(bars, metric_values):
            if abs(value) < 1:
                # Format as float with 2 decimal places
                text = f"{value:.2f}"
            else:
                # Format as integer if over 100, otherwise 2 decimal places
                text = f"{int(value)}" if abs(value) >= 100 else f"{value:.2f}"
            
            ax.text(bar.get_width() * 1.05, bar.get_y() + bar.get_height() / 2, 
                   text, va="center")
        
        ax.set_title("Performance Metrics", fontsize=14)
        ax.grid(axis="x")
        
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            plt.savefig(f"{save_path}_metrics.png", dpi=300, bbox_inches="tight")
        else:
            plt.show()
    
    def save_results(self, results: Dict[str, Any], path: str):
        """
        Save backtest results to file.
        
        Args:
            results: Dictionary of backtest results
            path: Path to save the results
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save equity curve
        results["equity_curve"].to_csv(f"{path}_equity_curve.csv", index=False)
        
        # Save trades if available
        if "trades" in results and not results["trades"].empty:
            results["trades"].to_csv(f"{path}_trades.csv", index=False)
        
        # Save metrics
        metrics_df = pd.DataFrame([results["metrics"]])
        metrics_df.to_csv(f"{path}_metrics.csv", index=False)
        
        # Save summary
        summary = {
            "initial_capital": results["initial_capital"],
            "final_value": results["final_value"],
            "profit": results["profit"],
            "return_pct": results["return_pct"],
            "metrics": results["metrics"]
        }
        
        with open(f"{path}_summary.json", "w") as f:
            json.dump(summary, f, indent=4)
        
        logger.info(f"Results saved to {path}")


class WalkForwardOptimizer:
    """
    Walk-forward optimization for trading strategies.
    """
    
    def __init__(self, 
                data: pd.DataFrame,
                train_window: int = 252,  # ~1 year of trading days
                test_window: int = 63,   # ~3 months of trading days
                step_size: int = 63,     # ~3 months of trading days
                initial_capital: float = 100000.0,
                commission_model: Callable = TransactionCostModel.percentage,
                slippage_model: Callable = TransactionCostModel.slippage_model,
                position_sizing: Callable = PositionSizer.fixed_size,
                price_col: str = "close",
                date_col: str = "timestamp"):
        """
        Initialize walk-forward optimizer.
        
        Args:
            data: DataFrame with market data
            train_window: Number of bars for training window
            test_window: Number of bars for test window
            step_size: Number of bars to step forward between windows
            initial_capital: Initial capital
            commission_model: Transaction cost model function
            slippage_model: Slippage model function
            position_sizing: Position sizing function
            price_col: Name of price column
            date_col: Name of date column
        """
        # Validate data
        required_cols = [price_col, "high", "low", "volume"]
        for col in required_cols:
            if col not in data.columns:
                logger.warning(f"Required column '{col}' not found in data")
        
        # Store parameters
        self.data = data.copy()
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.initial_capital = initial_capital
        self.commission_model = commission_model
        self.slippage_model = slippage_model
        self.position_sizing = position_sizing
        self.price_col = price_col
        self.date_col = date_col
        
        # Ensure data is sorted
        if date_col in self.data.columns:
            self.data = self.data.sort_values(date_col)
    
    def _split_data(self) -> List[Dict[str, pd.DataFrame]]:
        """
        Split data into train/test windows for walk-forward testing.
        
        Returns:
            List of dictionaries containing train and test DataFrames
        """
        windows = []
        data_len = len(self.data)
        
        # Calculate number of windows
        num_windows = (data_len - self.train_window - self.test_window) // self.step_size + 1
        
        for i in range(num_windows):
            # Calculate indices
            train_start = i * self.step_size
            train_end = train_start + self.train_window
            test_start = train_end
            test_end = test_start + self.test_window
            
            # Check if we have enough data
            if test_end > data_len:
                break
            
            # Create train/test split
            train_data = self.data.iloc[train_start:train_end].copy()
            test_data = self.data.iloc[test_start:test_end].copy()
            
            # Get date range info if available
            if self.date_col in self.data.columns:
                window_info = {
                    "train_start": train_data[self.date_col].min(),
                    "train_end": train_data[self.date_col].max(),
                    "test_start": test_data[self.date_col].min(),
                    "test_end": test_data[self.date_col].max()
                }
            else:
                window_info = {
                    "train_start": train_start,
                    "train_end": train_end - 1,
                    "test_start": test_start,
                    "test_end": test_end - 1
                }
            
            # Create window
            window = {
                "train": train_data,
                "test": test_data,
                "window_id": i,
                "info": window_info
            }
            
            windows.append(window)
        
        return windows
    
    def optimize(self, 
               strategy: Callable,
               param_grid: Dict[str, List[Any]],
               optimization_metric: str = "sharpe_ratio",
               max_optimization_time: Optional[int] = None,
               n_jobs: int = -1,
               random_state: int = 42) -> Dict[str, Any]:
        """
        Perform walk-forward optimization.
        
        Args:
            strategy: Strategy function that takes (context, data, params) and returns signals
            param_grid: Dictionary of parameter names and values to try
            optimization_metric: Metric to optimize
            max_optimization_time: Maximum time in seconds for optimization
            n_jobs: Number of parallel jobs
            random_state: Random state for reproducibility
        
        Returns:
            Dictionary of optimization results
        """
        # Split data into windows
        windows = self._split_data()
        
        if not windows:
            logger.error("No valid windows for walk-forward optimization")
            return {"error": "No valid windows"}
        
        # Generate parameter combinations
        import itertools
        param_keys = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        logger.info(f"Testing {len(param_combinations)} parameter combinations across {len(windows)} windows")
        
        # Define optimization function for a single window and parameter set
        def optimize_window(window, params_idx):
            # Create parameter dictionary
            params = {key: param_combinations[params_idx][i] for i, key in enumerate(param_keys)}
            
            # Create backtest engine for training
            engine = BacktestEngine(
                data=window["train"],
                initial_capital=self.initial_capital,
                commission_model=self.commission_model,
                slippage_model=self.slippage_model,
                position_sizing=self.position_sizing,
                price_col=self.price_col,
                date_col=self.date_col
            )
            
            # Run backtest on training data
            train_results = engine.run(strategy, strategy_params=params)
            
            # Check training performance
            train_metric = train_results["metrics"].get(optimization_metric, float("-inf"))
            
            # Return early if training performance is poor
            if train_metric < 0:
                return {
                    "window_id": window["window_id"],
                    "params": params,
                    "train_metric": train_metric,
                    "test_metric": float("-inf"),
                    "test_results": None
                }
            
            # Create backtest engine for testing
            engine = BacktestEngine(
                data=window["test"],
                initial_capital=self.initial_capital,
                commission_model=self.commission_model,
                slippage_model=self.slippage_model,
                position_sizing=self.position_sizing,
                price_col=self.price_col,
                date_col=self.date_col
            )
            
            # Run backtest on test data
            test_results = engine.run(strategy, strategy_params=params)
            
            # Get test metric
            test_metric = test_results["metrics"].get(optimization_metric, float("-inf"))
            
            return {
                "window_id": window["window_id"],
                "params": params,
                "train_metric": train_metric,
                "test_metric": test_metric,
                "test_results": test_results
            }
        
        # Initialize results container
        all_results = []
        
        # Configure parallel processing
        n_jobs = min(mp.cpu_count(), n_jobs) if n_jobs > 0 else mp.cpu_count()
        logger.info(f"Using {n_jobs} parallel processes")
        
        # Initialize timer
        start_time = datetime.now()
        
        # Run optimization
        try:
            # Create arguments list
            args_list = []
            for window_idx, window in enumerate(windows):
                for params_idx in range(len(param_combinations)):
                    args_list.append((window, params_idx))
            
            # Run optimizations
            with mp.Pool(processes=n_jobs) as pool:
                # Map arguments to optimization function
                for window, params_idx in args_list:
                    # Check time limit
                    if max_optimization_time is not None:
                        elapsed_time = (datetime.now() - start_time).total_seconds()
                        if elapsed_time > max_optimization_time:
                            logger.warning(f"Optimization stopped after {elapsed_time:.1f} seconds (time limit: {max_optimization_time} seconds)")
                            break
                    
                    # Run optimization
                    result = optimize_window(window, params_idx)
                    all_results.append(result)
        
        except KeyboardInterrupt:
            logger.warning("Optimization interrupted by user")
        
        # Calculate total optimization time
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
        # Find best parameters for each window
        best_params_by_window = []
        
        for window_id in range(len(windows)):
            # Get results for this window
            window_results = [r for r in all_results if r["window_id"] == window_id]
            
            if not window_results:
                continue
            
            # Find best parameters based on test metric
            best_result = max(window_results, key=lambda r: r["test_metric"])
            
            # Add window info
            best_result["info"] = windows[window_id]["info"]
            
            best_params_by_window.append(best_result)
        
        # Analyze parameter stability
        param_stats = {}
        for key in param_keys:
            values = [result["params"][key] for result in best_params_by_window]
            
            if isinstance(values[0], (int, float)):
                param_stats[key] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": min(values),
                    "max": max(values),
                    "values": values
                }
            else:
                # For non-numeric parameters
                from collections import Counter
                counts = Counter(values)
                param_stats[key] = {
                    "most_common": counts.most_common(1)[0][0],
                    "counts": dict(counts),
                    "values": values
                }
        
        # Determine robust parameters
        robust_params = {}
        
        for key in param_keys:
            stats = param_stats[key]
            
            if "mean" in stats:
                # For numeric parameters, use mean
                robust_params[key] = stats["mean"]
            else:
                # For non-numeric parameters, use most common
                robust_params[key] = stats["most_common"]
        
        # Calculate overall performance
        overall_train_metric = np.mean([r["train_metric"] for r in best_params_by_window])
        overall_test_metric = np.mean([r["test_metric"] for r in best_params_by_window])
        
        # Create final results
        optimization_results = {
            "windows": best_params_by_window,
            "param_stats": param_stats,
            "robust_params": robust_params,
            "overall_train_metric": overall_train_metric,
            "overall_test_metric": overall_test_metric,
            "optimization_metric": optimization_metric,
            "elapsed_time": elapsed_time
        }
        
        return optimization_results
    
    def run_robust_backtest(self, 
                         strategy: Callable,
                         robust_params: Dict[str, Any],
                         plot_results: bool = True,
                         save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run backtest with robust parameters.
        
        Args:
            strategy: Strategy function
            robust_params: Robust parameters from optimization
            plot_results: Whether to plot results
            save_path: Path to save results
        
        Returns:
            Dictionary of backtest results
        """
        # Create backtest engine
        engine = BacktestEngine(
            data=self.data,
            initial_capital=self.initial_capital,
            commission_model=self.commission_model,
            slippage_model=self.slippage_model,
            position_sizing=self.position_sizing,
            price_col=self.price_col,
            date_col=self.date_col
        )
        
        # Run backtest
        results = engine.run(strategy, strategy_params=robust_params)
        
        # Plot results if requested
        if plot_results:
            engine.plot_results(results, save_path)
        
        # Save results if path provided
        if save_path:
            engine.save_results(results, save_path)
        
        return results


class MultiStrategyBacktest:
    """
    Backtest multiple strategies and portfolio combinations.
    """
    
    def __init__(self, 
                data: pd.DataFrame,
                strategies: Dict[str, Tuple[Callable, Dict[str, Any]]],
                weights: Optional[Dict[str, float]] = None,
                initial_capital: float = 100000.0,
                commission_model: Callable = TransactionCostModel.percentage,
                slippage_model: Callable = TransactionCostModel.slippage_model,
                position_sizing: Callable = PositionSizer.fixed_size,
                price_col: str = "close",
                date_col: str = "timestamp"):
        """
        Initialize multi-strategy backtest.
        
        Args:
            data: DataFrame with market data
            strategies: Dictionary of strategy names and (strategy_function, params) tuples
            weights: Dictionary of strategy names and allocation weights
            initial_capital: Initial capital
            commission_model: Transaction cost model function
            slippage_model: Slippage model function
            position_sizing: Position sizing function
            price_col: Name of price column
            date_col: Name of date column
        """
        # Validate data
        required_cols = [price_col, "high", "low", "volume"]
        for col in required_cols:
            if col not in data.columns:
                logger.warning(f"Required column '{col}' not found in data")
        
        # Store parameters
        self.data = data.copy()
        self.strategies = strategies
        self.initial_capital = initial_capital
        self.commission_model = commission_model
        self.slippage_model = slippage_model
        self.position_sizing = position_sizing
        self.price_col = price_col
        self.date_col = date_col
        
        # Set weights
        if weights is None:
            # Equal weight allocation
            self.weights = {name: 1.0 / len(strategies) for name in strategies}
        else:
            # Normalize weights
            total_weight = sum(weights.values())
            self.weights = {name: weight / total_weight for name, weight in weights.items()}
        
        # Ensure data is sorted
        if date_col in self.data.columns:
            self.data = self.data.sort_values(date_col)
    
    def run(self, 
           start_date: Optional[Union[str, datetime]] = None,
           end_date: Optional[Union[str, datetime]] = None,
           warmup_period: int = 20) -> Dict[str, Any]:
        """
        Run backtest for all strategies and the combined portfolio.
        
        Args:
            start_date: Start date for backtest
            end_date: End date for backtest
            warmup_period: Number of bars for warmup
        
        Returns:
            Dictionary of backtest results
        """
        # Run individual strategy backtests
        strategy_results = {}
        
        for name, (strategy_fn, params) in self.strategies.items():
            logger.info(f"Running backtest for strategy: {name}")
            
            # Create backtest engine
            engine = BacktestEngine(
                data=self.data,
                initial_capital=self.initial_capital * self.weights[name],
                commission_model=self.commission_model,
                slippage_model=self.slippage_model,
                position_sizing=self.position_sizing,
                price_col=self.price_col,
                date_col=self.date_col
            )
            
            # Run backtest
            results = engine.run(strategy_fn, start_date, end_date, warmup_period, params)
            
            strategy_results[name] = results
        
        # Combine equity curves
        combined_equity = self._combine_equity_curves(strategy_results)
        
        # Calculate combined metrics
        combined_metrics = self._calculate_combined_metrics(combined_equity)
        
        # Format combined results
        combined_results = {
            "equity_curve": combined_equity,
            "metrics": combined_metrics,
            "strategy_results": strategy_results,
            "weights": self.weights,
            "initial_capital": self.initial_capital,
            "final_value": combined_equity["portfolio_value"].iloc[-1],
            "profit": combined_equity["portfolio_value"].iloc[-1] - self.initial_capital,
            "return_pct": (combined_equity["portfolio_value"].iloc[-1] / self.initial_capital - 1) * 100
        }
        
        return combined_results
    
    def _combine_equity_curves(self, strategy_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Combine equity curves from multiple strategies.
        
        Args:
            strategy_results: Dictionary of strategy names and results
        
        Returns:
            Combined equity curve DataFrame
        """
        # Get list of equity curves
        equity_curves = {name: results["equity_curve"] for name, results in strategy_results.items()}
        
        # Find common date range
        if self.date_col in equity_curves[list(equity_curves.keys())[0]].columns:
            # Combined based on dates
            common_dates = set(equity_curves[list(equity_curves.keys())[0]][self.date_col])
            
            for name, equity in equity_curves.items():
                common_dates &= set(equity[self.date_col])
            
            common_dates = sorted(common_dates)
            
            # Filter equity curves to common dates
            filtered_equity = {}
            for name, equity in equity_curves.items():
                filtered_equity[name] = equity[equity[self.date_col].isin(common_dates)]
            
            # Create combined equity curve
            combined = pd.DataFrame({self.date_col: common_dates})
            combined["portfolio_value"] = 0
            
            for name, equity in filtered_equity.items():
                equity = equity.sort_values(self.date_col)
                combined = combined.merge(
                    equity[[self.date_col, "portfolio_value"]].rename(
                        columns={"portfolio_value": f"{name}_value"}
                    ),
                    on=self.date_col
                )
                combined["portfolio_value"] += combined[f"{name}_value"]
        else:
            # Combine based on index
            min_length = min(len(equity) for equity in equity_curves.values())
            
            # Truncate equity curves to same length
            filtered_equity = {}
            for name, equity in equity_curves.items():
                filtered_equity[name] = equity.iloc[:min_length].reset_index(drop=True)
            
            # Create combined equity curve
            combined = pd.DataFrame({"portfolio_value": 0})
            
            for name, equity in filtered_equity.items():
                combined = combined.join(
                    equity[["portfolio_value"]].rename(
                        columns={"portfolio_value": f"{name}_value"}
                    )
                )
                combined["portfolio_value"] += combined[f"{name}_value"]
        
        # Calculate daily returns
        combined["daily_return"] = combined["portfolio_value"].pct_change()
        
        # Calculate cumulative returns
        combined["cum_return"] = (1 + combined["daily_return"]).cumprod() - 1
        
        # Calculate drawdown
        combined["high_watermark"] = combined["portfolio_value"].cummax()
        combined["drawdown"] = (combined["portfolio_value"] / combined["high_watermark"] - 1) * 100
        
        return combined
    
    def _calculate_combined_metrics(self, combined_equity: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate performance metrics for combined portfolio.
        
        Args:
            combined_equity: Combined equity curve DataFrame
        
        Returns:
            Dictionary of performance metrics
        """
        # Calculate daily returns
        combined_equity["daily_return"] = combined_equity["portfolio_value"].pct_change()
        
        # Calculate cumulative returns
        combined_equity["cum_return"] = (1 + combined_equity["daily_return"]).cumprod() - 1
        
        # Calculate drawdown
        combined_equity["high_watermark"] = combined_equity["portfolio_value"].cummax()
        combined_equity["drawdown"] = (combined_equity["portfolio_value"] / combined_equity["high_watermark"] - 1) * 100
        
        # Calculate annualized metrics
        if self.date_col in combined_equity.columns:
            # Calculate trading days in test period
            start_date = combined_equity[self.date_col].min()
            end_date = combined_equity[self.date_col].max()
            days = (end_date - start_date).days
            years = days / 365
            
            # Annualized return
            total_return = combined_equity["portfolio_value"].iloc[-1] / combined_equity["portfolio_value"].iloc[0] - 1
            ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
            
            # Annualized volatility
            daily_std = combined_equity["daily_return"].std()
            ann_volatility = daily_std * np.sqrt(252)
            
            # Sharpe ratio
            risk_free_rate = 0.03  # Assumed risk-free rate
            excess_return = ann_return - risk_free_rate
            sharpe_ratio = excess_return / ann_volatility if ann_volatility != 0 else 0
            
            # Sortino ratio
            downside_returns = combined_equity["daily_return"][combined_equity["daily_return"] < 0]
            downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
            ann_downside_std = downside_std * np.sqrt(252)
            sortino_ratio = excess_return / ann_downside_std if ann_downside_std != 0 else 0
        else:
            # Use index-based calculations if no date column
            total_return = combined_equity["portfolio_value"].iloc[-1] / combined_equity["portfolio_value"].iloc[0] - 1
            ann_return = total_return  # Cannot annualize without dates
            ann_volatility = combined_equity["daily_return"].std() * np.sqrt(252)
            
            risk_free_rate = 0.03
            excess_return = ann_return - risk_free_rate
            sharpe_ratio = excess_return / ann_volatility if ann_volatility != 0 else 0
            
            downside_returns = combined_equity["daily_return"][combined_equity["daily_return"] < 0]
            downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
            ann_downside_std = downside_std * np.sqrt(252)
            sortino_ratio = excess_return / ann_downside_std if ann_downside_std != 0 else 0
        
        # Calculate max drawdown
        max_drawdown = combined_equity["drawdown"].min()
        
        # Calculate Calmar ratio
        calmar_ratio = ann_return / abs(max_drawdown / 100) if max_drawdown != 0 else 0
        
        # Compile metrics
        metrics = {
            "total_return": total_return * 100,
            "annualized_return": ann_return * 100,
            "annualized_volatility": ann_volatility * 100,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "max_drawdown": max_drawdown
        }
        
        return metrics
    
    def optimize_weights(self, 
                       metric: str = "sharpe_ratio",
                       method: str = "grid_search",
                       n_samples: int = 100,
                       random_state: int = 42) -> Dict[str, Any]:
        """
        Optimize portfolio weights.
        
        Args:
            metric: Metric to optimize
            method: Optimization method ('grid_search' or 'random_search')
            n_samples: Number of weight combinations to try
            random_state: Random state for reproducibility
        
        Returns:
            Dictionary of optimization results
        """
        import numpy as np
        from scipy.optimize import minimize
        
        # Run individual strategy backtests
        strategy_results = {}
        
        for name, (strategy_fn, params) in self.strategies.items():
            logger.info(f"Running backtest for strategy: {name}")
            
            # Create backtest engine
            engine = BacktestEngine(
                data=self.data,
                initial_capital=self.initial_capital,
                commission_model=self.commission_model,
                slippage_model=self.slippage_model,
                position_sizing=self.position_sizing,
                price_col=self.price_col,
                date_col=self.date_col
            )
            
            # Run backtest
            results = engine.run(strategy_fn, strategy_params=params)
            
            strategy_results[name] = results
        
        # Extract equity curves
        equity_curves = {name: results["equity_curve"] for name, results in strategy_results.items()}
        
        # Find common date range
        if self.date_col in equity_curves[list(equity_curves.keys())[0]].columns:
            # Combined based on dates
            common_dates = set(equity_curves[list(equity_curves.keys())[0]][self.date_col])
            
            for name, equity in equity_curves.items():
                common_dates &= set(equity[self.date_col])
            
            common_dates = sorted(common_dates)
            
            # Filter equity curves to common dates
            filtered_equity = {}
            for name, equity in equity_curves.items():
                filtered_equity[name] = equity[equity[self.date_col].isin(common_dates)]
            
            # Extract daily returns
            returns = pd.DataFrame({self.date_col: common_dates})
            
            for name, equity in filtered_equity.items():
                equity = equity.sort_values(self.date_col)
                returns[name] = equity["daily_return"].values
        else:
            # Combine based on index
            min_length = min(len(equity) for equity in equity_curves.values())
            
            # Truncate equity curves to same length
            filtered_equity = {}
            for name, equity in equity_curves.items():
                filtered_equity[name] = equity.iloc[:min_length].reset_index(drop=True)
            
            # Extract daily returns
            returns = pd.DataFrame()
            
            for name, equity in filtered_equity.items():
                returns[name] = equity["daily_return"].values
        
        # Calculate returns matrix
        returns_matrix = returns.drop(columns=[self.date_col]) if self.date_col in returns.columns else returns
        returns_matrix = returns_matrix.dropna()
        
        # Define optimization function
        def portfolio_metric(weights, returns=returns_matrix, metric=metric):
            # Convert to numpy array
            weights = np.array(weights)
            
            # Normalize weights
            weights = weights / np.sum(weights)
            
            # Calculate portfolio returns
            portfolio_returns = returns.dot(weights)
            
            # Calculate metrics
            annual_return = portfolio_returns.mean() * 252
            annual_vol = portfolio_returns.std() * np.sqrt(252)
            sharpe = annual_return / annual_vol if annual_vol != 0 else 0
            
            # Calculate drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            peak = cumulative_returns.cummax()
            drawdown = (cumulative_returns / peak - 1)
            max_drawdown = drawdown.min()
            
            # Calculate sortino ratio
            downside_returns = portfolio_returns[portfolio_returns < 0]
            downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino = annual_return / downside_vol if downside_vol != 0 else 0
            
            # Calculate calmar ratio
            calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Return appropriate metric
            if metric == "sharpe_ratio":
                return -sharpe  # Negative because we want to maximize
            elif metric == "sortino_ratio":
                return -sortino
            elif metric == "calmar_ratio":
                return -calmar
            elif metric == "max_drawdown":
                return abs(max_drawdown)
            elif metric == "annual_return":
                return -annual_return
            elif metric == "annual_vol":
                return annual_vol
            else:
                return -sharpe
        
        # Number of strategies
        n_strategies = len(self.strategies)
        
        # Optimization
        if method == "grid_search" and n_strategies <= 3:
            # Grid search - only practical for 2-3 strategies
            results = []
            
            if n_strategies == 2:
                # Simplified grid for 2 strategies
                grid_points = 20
                weights_list = []
                
                for i in range(grid_points + 1):
                    w1 = i / grid_points
                    w2 = 1 - w1
                    weights_list.append([w1, w2])
            else:
                # Grid for 3 strategies
                grid_points = 10
                weights_list = []
                
                for i in range(grid_points + 1):
                    for j in range(grid_points + 1 - i):
                        w1 = i / grid_points
                        w2 = j / grid_points
                        w3 = 1 - w1 - w2
                        if w3 >= 0:
                            weights_list.append([w1, w2, w3])
            
            # Evaluate all weight combinations
            for weights in weights_list:
                metric_value = portfolio_metric(weights)
                
                results.append({
                    "weights": dict(zip(self.strategies.keys(), weights)),
                    "metric_value": -metric_value if metric not in ["max_drawdown", "annual_vol"] else metric_value
                })
            
            # Find best weights
            best_result = max(results, key=lambda x: x["metric_value"])
            
            optimal_weights = best_result["weights"]
            optimal_metric = best_result["metric_value"]
        
        elif method == "random_search":
            # Random search
            np.random.seed(random_state)
            best_metric = float("-inf") if metric not in ["max_drawdown", "annual_vol"] else float("inf")
            best_weights = None
            results = []
            
            for _ in range(n_samples):
                # Generate random weights
                weights = np.random.random(n_strategies)
                weights = weights / np.sum(weights)
                
                # Evaluate metric
                metric_value = portfolio_metric(weights)
                
                # Convert to positive for maximization metrics
                if metric not in ["max_drawdown", "annual_vol"]:
                    metric_value = -metric_value
                
                # Check if best so far
                if ((metric not in ["max_drawdown", "annual_vol"] and metric_value > best_metric) or
                    (metric in ["max_drawdown", "annual_vol"] and metric_value < best_metric)):
                    best_metric = metric_value
                    best_weights = dict(zip(self.strategies.keys(), weights))
                
                results.append({
                    "weights": dict(zip(self.strategies.keys(), weights)),
                    "metric_value": metric_value
                })
            
            optimal_weights = best_weights
            optimal_metric = best_metric
        
        else:
            # Scipy optimization
            bounds = [(0, 1) for _ in range(n_strategies)]
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            
            # Initial equal weights
            initial_weights = np.ones(n_strategies) / n_strategies
            
            # Optimize
            result = minimize(
                portfolio_metric,
                initial_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints
            )
            
            optimal_weights_array = result.x / np.sum(result.x)  # Normalize
            optimal_weights = dict(zip(self.strategies.keys(), optimal_weights_array))
            
            # Calculate optimal metric value
            metric_value = portfolio_metric(optimal_weights_array)
            optimal_metric = -metric_value if metric not in ["max_drawdown", "annual_vol"] else metric_value
        
        # Update weights
        self.weights = optimal_weights
        
        # Run backtest with optimal weights
        optimal_results = self.run()
        
        # Return optimization results
        optimization_results = {
            "optimal_weights": optimal_weights,
            "optimal_metric": optimal_metric,
            "optimal_results": optimal_results,
            "optimization_method": method,
            "optimization_metric": metric
        }
        
        return optimization_results
    
    def plot_results(self, results: Dict[str, Any], save_path: Optional[str] = None):
        """
        Plot backtest results.
        
        Args:
            results: Dictionary of backtest results
            save_path: Path to save the plots
        """
        # Set up matplotlib
        plt.style.use("seaborn-v0_8-darkgrid")
        plt.rcParams["figure.figsize"] = (12, 8)
        
        # Extract data
        combined_equity = results["equity_curve"]
        strategy_results = results["strategy_results"]
        
        # Create figure for equity curves
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]}, figsize=(12, 8))
        
        # Plot combined equity curve
        x_axis = combined_equity[self.date_col] if self.date_col in combined_equity.columns else combined_equity.index
        ax1.plot(x_axis, combined_equity["portfolio_value"], label="Combined Portfolio", linewidth=2, color="black")
        
        # Plot individual strategy equity curves
        for name, result in strategy_results.items():
            equity = result["equity_curve"]
            x_axis_strat = equity[self.date_col] if self.date_col in equity.columns else equity.index
            ax1.plot(x_axis_strat, equity["portfolio_value"], label=f"{name} ({self.weights[name]:.2f})", alpha=0.7)
        
        # Format equity curve plot
        ax1.set_title("Portfolio Equity Curves", fontsize=14)
        ax1.set_ylabel("Portfolio Value", fontsize=12)
        ax1.grid(True)
        ax1.legend()
        
        # Plot drawdown
        ax2.fill_between(x_axis, 0, combined_equity["drawdown"], 
                        color="red", alpha=0.5, label="Combined Drawdown")
        ax2.set_ylabel("Drawdown (%)", fontsize=12)
        ax2.set_xlabel("Date" if self.date_col in combined_equity.columns else "Bar", fontsize=12)
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            plt.savefig(f"{save_path}_equity_curve.png", dpi=300, bbox_inches="tight")
        else:
            plt.show()
        
        # Create second figure for correlation heatmap
        if len(strategy_results) > 1:
            # Extract daily returns
            returns_data = {}
            
            for name, result in strategy_results.items():
                equity = result["equity_curve"]
                returns_data[name] = equity["daily_return"].values
            
            # Create returns DataFrame
            returns_df = pd.DataFrame(returns_data)
            
            # Calculate correlation matrix
            corr_matrix = returns_df.corr()
            
            # Create correlation heatmap
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, ax=ax)
            ax.set_title("Strategy Returns Correlation Matrix", fontsize=14)
            
            plt.tight_layout()
            
            # Save or show the plot
            if save_path:
                plt.savefig(f"{save_path}_correlation.png", dpi=300, bbox_inches="tight")
            else:
                plt.show()
        
        # Create third figure for performance metrics
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Key metrics to display
        key_metrics = [
            "total_return", "annualized_return", "max_drawdown", "sharpe_ratio",
            "sortino_ratio", "calmar_ratio"
        ]
        
        # Collect metrics for all strategies and combined
        all_metrics = {}
        
        # Add combined metrics
        all_metrics["Combined"] = [results["metrics"].get(m, 0) for m in key_metrics]
        
        # Add individual strategy metrics
        for name, result in strategy_results.items():
            all_metrics[name] = [result["metrics"].get(m, 0) for m in key_metrics]
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame(all_metrics, index=key_metrics)
        
        # Format metric names
        metrics_df.index = [m.replace("_", " ").title() for m in key_metrics]
        
        # Plot metrics
        metrics_df.plot(kind="bar", ax=ax)
        ax.set_title("Performance Metrics Comparison", fontsize=14)
        ax.set_ylabel("Value", fontsize=12)
        ax.legend(title="Strategy")
        
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            plt.savefig(f"{save_path}_metrics.png", dpi=300, bbox_inches="tight")
        else:
            plt.show()