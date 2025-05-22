"""
Reinforcement learning models for trading (xLSTM with PPO).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
import os
import json
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime
import copy
import random
from collections import deque
import gym
from gym import spaces

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class xLSTMPolicyNetwork(nn.Module):
    """
    Extended LSTM (xLSTM) policy network for reinforcement learning.
    
    Features a bidirectional LSTM with skip connections and attention mechanism.
    """
    
    def __init__(self, 
                input_dim: int,
                lstm_units: int = 128,
                lstm_layers: int = 2,
                policy_hidden_units: List[int] = [64, 32],
                action_dim: int = 3,  # 0: Hold, 1: Buy, 2: Sell
                dropout: float = 0.1):
        """
        Initialize xLSTM policy network.
        
        Args:
            input_dim: Number of input features
            lstm_units: Number of LSTM units
            lstm_layers: Number of LSTM layers
            policy_hidden_units: List of hidden units for policy head
            action_dim: Number of possible actions
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.lstm_units = lstm_units
        self.lstm_layers = lstm_layers
        self.policy_hidden_units = policy_hidden_units
        self.action_dim = action_dim
        
        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_units,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Skip connection
        self.skip_connection = nn.Linear(input_dim, lstm_units * 2)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(lstm_units * 2, lstm_units),
            nn.Tanh(),
            nn.Linear(lstm_units, 1)
        )
        
        # Policy head
        policy_layers = []
        in_features = lstm_units * 2
        
        for units in policy_hidden_units:
            policy_layers.append(nn.Linear(in_features, units))
            policy_layers.append(nn.ReLU())
            policy_layers.append(nn.Dropout(dropout))
            in_features = units
        
        policy_layers.append(nn.Linear(in_features, action_dim))
        self.policy_head = nn.Sequential(*policy_layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        
        Returns:
            Tuple of (action_probs, lstm_output)
                - action_probs: Action probabilities of shape (batch_size, action_dim)
                - lstm_output: LSTM output of shape (batch_size, lstm_units * 2)
        """
        batch_size, seq_len, _ = x.size()
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, lstm_units * 2)
        
        # Apply skip connection
        skip_out = self.skip_connection(x)  # (batch_size, seq_len, lstm_units * 2)
        lstm_out = lstm_out + skip_out  # Residual connection
        
        # Apply attention
        attention_scores = self.attention(lstm_out)  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, seq_len, 1)
        
        # Apply attention weights
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)  # (batch_size, lstm_units * 2)
        
        # Pass through policy head
        action_logits = self.policy_head(context_vector)  # (batch_size, action_dim)
        action_probs = F.softmax(action_logits, dim=1)  # (batch_size, action_dim)
        
        return action_probs, context_vector
    
    def _initialize_weights(self):
        """
        Initialize weights for better convergence.
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # LSTM weights
                    nn.init.orthogonal_(param)
                else:
                    # Linear layer weights
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)


class xLSTMValueNetwork(nn.Module):
    """
    Extended LSTM (xLSTM) value network for reinforcement learning.
    """
    
    def __init__(self, 
                lstm_units: int = 128,
                value_hidden_units: List[int] = [64, 32],
                dropout: float = 0.1):
        """
        Initialize xLSTM value network.
        
        Args:
            lstm_units: Number of LSTM units (from policy network)
            value_hidden_units: List of hidden units for value head
            dropout: Dropout rate
        """
        super().__init__()
        
        # Value head
        value_layers = []
        in_features = lstm_units * 2  # Bidirectional LSTM
        
        for units in value_hidden_units:
            value_layers.append(nn.Linear(in_features, units))
            value_layers.append(nn.ReLU())
            value_layers.append(nn.Dropout(dropout))
            in_features = units
        
        value_layers.append(nn.Linear(in_features, 1))
        self.value_head = nn.Sequential(*value_layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            lstm_output: LSTM output of shape (batch_size, lstm_units * 2)
        
        Returns:
            Value prediction of shape (batch_size, 1)
        """
        return self.value_head(lstm_output)  # (batch_size, 1)
    
    def _initialize_weights(self):
        """
        Initialize weights for better convergence.
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                # Linear layer weights
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)


class PPOMemory:
    """
    Memory buffer for PPO algorithm.
    """
    
    def __init__(self, batch_size: int = 64):
        """
        Initialize PPO memory.
        
        Args:
            batch_size: Batch size for updates
        """
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size
    
    def store(self, 
             state: np.ndarray, 
             action: int, 
             probs: float, 
             val: float, 
             reward: float, 
             done: bool):
        """
        Store transition.
        
        Args:
            state: State observation
            action: Action taken
            probs: Action probabilities
            val: Value prediction
            reward: Reward received
            done: Done flag
        """
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def clear(self):
        """
        Clear memory.
        """
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
    
    def generate_batches(self) -> List[Tuple]:
        """
        Generate batches for training.
        
        Returns:
            List of tuples (states, actions, probs, vals, rewards, dones, batches)
        """
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        return self.states, self.actions, self.probs, self.vals, self.rewards, self.dones, batches


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent with xLSTM networks.
    """
    
    def __init__(self, 
                input_dim: int,
                lstm_units: int = 128,
                lstm_layers: int = 2,
                policy_hidden_units: List[int] = [64, 32],
                value_hidden_units: List[int] = [64, 32],
                action_dim: int = 3,  # 0: Hold, 1: Buy, 2: Sell
                dropout: float = 0.1,
                gamma: float = 0.99,
                lambda_gae: float = 0.95,
                policy_clip: float = 0.2,
                batch_size: int = 64,
                n_epochs: int = 10,
                lr: float = 0.0003,
                device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize PPO agent.
        
        Args:
            input_dim: Number of input features
            lstm_units: Number of LSTM units
            lstm_layers: Number of LSTM layers
            policy_hidden_units: List of hidden units for policy head
            value_hidden_units: List of hidden units for value head
            action_dim: Number of possible actions
            dropout: Dropout rate
            gamma: Discount factor
            lambda_gae: GAE parameter
            policy_clip: PPO clipping parameter
            batch_size: Batch size for updates
            n_epochs: Number of epochs per update
            lr: Learning rate
            device: Device to use (cuda or cpu)
        """
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.device = device
        
        # Create policy and value networks
        self.policy = xLSTMPolicyNetwork(
            input_dim=input_dim,
            lstm_units=lstm_units,
            lstm_layers=lstm_layers,
            policy_hidden_units=policy_hidden_units,
            action_dim=action_dim,
            dropout=dropout
        ).to(device)
        
        self.value = xLSTMValueNetwork(
            lstm_units=lstm_units,
            value_hidden_units=value_hidden_units,
            dropout=dropout
        ).to(device)
        
        # Create optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr)
        
        # Create memory buffer
        self.memory = PPOMemory(batch_size)
    
    def choose_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """
        Choose an action based on current state.
        
        Args:
            state: State observation
        
        Returns:
            Tuple of (action, action_prob, value)
        """
        # Convert state to tensor
        state_tensor = torch.tensor(np.array([state]), dtype=torch.float32).to(self.device)
        
        # Get action probabilities and value
        with torch.no_grad():
            action_probs, lstm_out = self.policy(state_tensor)
            value = self.value(lstm_out)
        
        # Sample action from distribution
        action_probs_np = action_probs.cpu().numpy()[0]
        action = np.random.choice(len(action_probs_np), p=action_probs_np)
        
        # Return action, probability, and value
        return action, action_probs_np[action], value.cpu().numpy()[0, 0]
    
    def store_transition(self, 
                       state: np.ndarray, 
                       action: int, 
                       prob: float, 
                       val: float, 
                       reward: float, 
                       done: bool):
        """
        Store transition in memory.
        
        Args:
            state: State observation
            action: Action taken
            prob: Action probability
            val: Value prediction
            reward: Reward received
            done: Done flag
        """
        self.memory.store(state, action, prob, val, reward, done)
    
    def learn(self) -> Dict[str, float]:
        """
        Update policy and value networks.
        
        Returns:
            Dictionary of training metrics
        """
        # Get data from memory
        states, actions, old_probs, values, rewards, dones, batches = self.memory.generate_batches()
        
        # Convert to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        old_probs = np.array(old_probs)
        values = np.array(values)
        rewards = np.array(rewards)
        dones = np.array(dones)
        
        # Compute advantages
        advantages = np.zeros(len(rewards), dtype=np.float32)
        
        for t in range(len(rewards) - 1):
            discount = 1
            a_t = 0
            
            for k in range(t, len(rewards) - 1):
                a_t += discount * (rewards[k] + self.gamma * values[k + 1] * (1 - dones[k]) - values[k])
                discount *= self.gamma * self.lambda_gae
                
                if dones[k]:
                    break
            
            advantages[t] = a_t
        
        # Compute returns
        returns = advantages + values
        
        # Training loop
        actor_losses = []
        critic_losses = []
        total_losses = []
        
        for _ in range(self.n_epochs):
            # Process each batch
            for batch in batches:
                # Convert batch data to tensors
                batch_states = torch.tensor(states[batch], dtype=torch.float32).to(self.device)
                batch_actions = torch.tensor(actions[batch], dtype=torch.long).to(self.device)
                batch_old_probs = torch.tensor(old_probs[batch], dtype=torch.float32).to(self.device)
                batch_returns = torch.tensor(returns[batch], dtype=torch.float32).to(self.device)
                batch_advantages = torch.tensor(advantages[batch], dtype=torch.float32).to(self.device)
                
                # Get action probabilities and values
                action_probs, lstm_out = self.policy(batch_states)
                values = self.value(lstm_out)
                
                # Extract probabilities for taken actions
                dist = torch.distributions.Categorical(action_probs)
                new_probs = dist.log_prob(batch_actions)
                
                # Compute entropy (for exploration)
                entropy = dist.entropy().mean()
                
                # Compute actor loss
                prob_ratio = torch.exp(new_probs - torch.log(batch_old_probs + 1e-10))
                weighted_probs = batch_advantages * prob_ratio
                clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * batch_advantages
                actor_loss = -torch.min(weighted_probs, clipped_probs).mean() - 0.01 * entropy
                
                # Compute critic loss
                critic_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # Compute total loss
                total_loss = actor_loss + 0.5 * critic_loss
                
                # Update policy and value networks
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=0.5)
                self.policy_optimizer.step()
                self.value_optimizer.step()
                
                # Store losses
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                total_losses.append(total_loss.item())
        
        # Clear memory
        self.memory.clear()
        
        # Return metrics
        return {
            "actor_loss": np.mean(actor_losses),
            "critic_loss": np.mean(critic_losses),
            "total_loss": np.mean(total_losses)
        }
    
    def save(self, filepath: str) -> None:
        """
        Save agent.
        
        Args:
            filepath: Path to save agent
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save policy and value networks
        torch.save(self.policy.state_dict(), f"{filepath}_policy.pt")
        torch.save(self.value.state_dict(), f"{filepath}_value.pt")
        
        # Save agent parameters
        params = {
            "input_dim": self.policy.input_dim,
            "lstm_units": self.policy.lstm_units,
            "lstm_layers": self.policy.lstm_layers,
            "policy_hidden_units": self.policy.policy_hidden_units,
            "action_dim": self.policy.action_dim,
            "gamma": self.gamma,
            "lambda_gae": self.lambda_gae,
            "policy_clip": self.policy_clip,
            "n_epochs": self.n_epochs
        }
        
        with open(f"{filepath}_params.json", "w") as f:
            json.dump(params, f)
    
    def load(self, filepath: str) -> None:
        """
        Load agent.
        
        Args:
            filepath: Path to load agent from
        """
        # Load parameters
        with open(f"{filepath}_params.json", "r") as f:
            params = json.load(f)
        
        # Update agent parameters
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Load policy and value networks
        self.policy.load_state_dict(torch.load(f"{filepath}_policy.pt"))
        self.value.load_state_dict(torch.load(f"{filepath}_value.pt"))


class TradingEnvironment(gym.Env):
    """
    Trading environment for reinforcement learning.
    """
    
    def __init__(self,
                data: pd.DataFrame,
                window_size: int = 20,
                initial_balance: float = 10000.0,
                commission: float = 0.001,
                reward_scaling: float = 0.01,
                features: Optional[List[str]] = None,
                price_col: str = "close",
                debug: bool = False):
        """
        Initialize trading environment.
        
        Args:
            data: DataFrame with market data
            window_size: Size of observation window
            initial_balance: Initial account balance
            commission: Trading commission
            reward_scaling: Scaling factor for rewards
            features: List of feature columns to use (if None, use all except price_col)
            price_col: Name of price column
            debug: Whether to print debug information
        """
        super().__init__()
        
        # Store parameters
        self.data = data.copy()
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.commission = commission
        self.reward_scaling = reward_scaling
        self.price_col = price_col
        self.debug = debug
        
        # Extract price series
        self.prices = self.data[price_col].values
        
        # Determine features
        if features is None:
            self.features = [col for col in self.data.columns if col != price_col]
        else:
            self.features = features
        
        # Extract feature data
        self.feature_data = self.data[self.features].values
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        
        # Observation space includes:
        # - Feature data for window_size time steps
        # - Current position (0: no position, 1: long)
        # - Current balance
        self.observation_space = spaces.Dict({
            "features": spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(self.window_size, len(self.features)), 
                dtype=np.float32
            ),
            "position": spaces.Discrete(2),
            "balance": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        })
        
        # Initialize state
        self.reset()
    
    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset environment to initial state.
        
        Returns:
            Initial observation
        """
        # Reset position
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares_held = 0
        self.position = 0  # 0: no position, 1: long
        self.done = False
        self.trades = []
        
        # Get initial observation
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take (0: Hold, 1: Buy, 2: Sell)
        
        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Execute action
        reward, info = self._take_action(action)
        
        # Update step
        self.current_step += 1
        
        # Check if episode is done
        if self.current_step >= len(self.prices):
            self.done = True
        
        # Get new observation
        obs = self._get_observation()
        
        return obs, reward, self.done, info
    
    def _take_action(self, action: int) -> Tuple[float, Dict]:
        """
        Execute trading action.
        
        Args:
            action: Action to take (0: Hold, 1: Buy, 2: Sell)
        
        Returns:
            Tuple of (reward, info)
        """
        # Get current price
        current_price = self.prices[self.current_step]
        
        # Initialize reward and info
        reward = 0
        trade_reward = 0
        trade_info = {}
        
        # Execute action
        if action == 0:  # Hold
            # Small negative reward to encourage action
            reward = -0.01 * self.reward_scaling
            
        elif action == 1:  # Buy
            if self.position == 0:  # Only buy if not already long
                # Calculate maximum shares that can be bought
                max_shares = int(self.balance / (current_price * (1 + self.commission)))
                
                if max_shares > 0:
                    # Execute buy order
                    self.shares_held = max_shares
                    cost = self.shares_held * current_price * (1 + self.commission)
                    self.balance -= cost
                    self.position = 1
                    
                    # Record trade
                    trade_info = {
                        "type": "buy",
                        "step": self.current_step,
                        "price": current_price,
                        "shares": self.shares_held,
                        "cost": cost
                    }
                    self.trades.append(trade_info)
                    
                    if self.debug:
                        print(f"Buy: {self.shares_held} shares at {current_price} for {cost}")
                else:
                    # Not enough balance to buy
                    reward = -0.1 * self.reward_scaling
            else:
                # Already in position, penalize unnecessary action
                reward = -0.1 * self.reward_scaling
        
        elif action == 2:  # Sell
            if self.position == 1:  # Only sell if holding shares
                # Execute sell order
                revenue = self.shares_held * current_price * (1 - self.commission)
                self.balance += revenue
                
                # Calculate profit/loss
                cost_basis = self.trades[-1]["cost"]
                profit = revenue - cost_basis
                profit_pct = profit / cost_basis
                
                # Adjust reward based on profit/loss
                trade_reward = profit_pct * self.reward_scaling
                
                # Record trade
                trade_info = {
                    "type": "sell",
                    "step": self.current_step,
                    "price": current_price,
                    "shares": self.shares_held,
                    "revenue": revenue,
                    "profit": profit,
                    "profit_pct": profit_pct
                }
                self.trades.append(trade_info)
                
                if self.debug:
                    print(f"Sell: {self.shares_held} shares at {current_price} for {revenue}, profit: {profit} ({profit_pct:.2%})")
                
                # Reset position
                self.shares_held = 0
                self.position = 0
            else:
                # No shares to sell, penalize unnecessary action
                reward = -0.1 * self.reward_scaling
        
        # Calculate total portfolio value
        portfolio_value = self.balance + self.shares_held * current_price
        
        # Additional reward for portfolio growth
        if self.current_step > self.window_size:
            # Calculate market return (buy-and-hold strategy)
            market_return = self.prices[self.current_step] / self.prices[self.window_size] - 1
            
            # Calculate portfolio return
            portfolio_return = portfolio_value / self.initial_balance - 1
            
            # Reward outperforming the market
            market_reward = (portfolio_return - market_return) * self.reward_scaling
            reward += market_reward
        
        # Add trade reward
        reward += trade_reward
        
        # Return reward and info
        info = {
            "balance": self.balance,
            "shares_held": self.shares_held,
            "portfolio_value": portfolio_value,
            "current_price": current_price,
            "position": self.position,
            "trade_info": trade_info
        }
        
        return reward, info
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Get current observation.
        
        Returns:
            Observation dictionary
        """
        # Get feature window
        feature_window = self.feature_data[self.current_step - self.window_size:self.current_step]
        
        # Create observation
        obs = {
            "features": feature_window.astype(np.float32),
            "position": np.array(self.position, dtype=np.int64),
            "balance": np.array([self.balance], dtype=np.float32)
        }
        
        return obs
    
    def render(self, mode: str = "human") -> None:
        """
        Render environment.
        
        Args:
            mode: Rendering mode
        """
        # Get current price and portfolio value
        current_price = self.prices[self.current_step]
        portfolio_value = self.balance + self.shares_held * current_price
        
        print(f"Step: {self.current_step}")
        print(f"Price: {current_price:.2f}")
        print(f"Balance: {self.balance:.2f}")
        print(f"Shares held: {self.shares_held}")
        print(f"Portfolio value: {portfolio_value:.2f}")
        print(f"Position: {'long' if self.position == 1 else 'none'}")
        print("-" * 40)
    
    def get_portfolio_history(self) -> pd.DataFrame:
        """
        Get history of portfolio performance.
        
        Returns:
            DataFrame with portfolio history
        """
        # Initialize history
        history = []
        
        # Calculate portfolio value at each step
        for step in range(self.window_size, len(self.prices)):
            # Get price at step
            price = self.prices[step]
            
            # Find relevant trades up to this step
            buys = [t for t in self.trades if t["type"] == "buy" and t["step"] <= step]
            sells = [t for t in self.trades if t["type"] == "sell" and t["step"] <= step]
            
            # Calculate shares held
            if buys and (not sells or buys[-1]["step"] > sells[-1]["step"]):
                # Last trade was a buy
                shares_held = buys[-1]["shares"]
                last_buy_cost = buys[-1]["cost"]
            else:
                # Last trade was a sell or no trades
                shares_held = 0
                last_buy_cost = 0
            
            # Calculate balance
            if not self.trades:
                # No trades yet
                balance = self.initial_balance
            else:
                # Get balance after last trade
                last_trade = max(self.trades, key=lambda t: t["step"])
                if last_trade["step"] <= step:
                    if last_trade["type"] == "buy":
                        balance = self.initial_balance - sum(t["cost"] for t in buys) + sum(t["revenue"] for t in sells)
                    else:  # sell
                        balance = self.initial_balance - sum(t["cost"] for t in buys) + sum(t["revenue"] for t in sells)
                else:
                    # Last trade is after this step
                    balance = self.initial_balance - sum(t["cost"] for t in buys if t["step"] <= step) + sum(t["revenue"] for t in sells if t["step"] <= step)
            
            # Calculate portfolio value
            portfolio_value = balance + shares_held * price
            
            # Calculate profit/loss
            if shares_held > 0:
                unrealized_profit = shares_held * price - last_buy_cost
                unrealized_profit_pct = unrealized_profit / last_buy_cost if last_buy_cost > 0 else 0
            else:
                unrealized_profit = 0
                unrealized_profit_pct = 0
            
            # Add to history
            history.append({
                "step": step,
                "price": price,
                "balance": balance,
                "shares_held": shares_held,
                "portfolio_value": portfolio_value,
                "unrealized_profit": unrealized_profit,
                "unrealized_profit_pct": unrealized_profit_pct
            })
        
        # Convert to DataFrame
        return pd.DataFrame(history)


def train_rl_model(data: pd.DataFrame,
                  window_size: int = 20,
                  initial_balance: float = 10000.0,
                  commission: float = 0.001,
                  reward_scaling: float = 0.1,
                  features: Optional[List[str]] = None,
                  price_col: str = "close",
                  lstm_units: int = 64,
                  lstm_layers: int = 2,
                  policy_hidden_units: List[int] = [64, 32],
                  value_hidden_units: List[int] = [64, 32],
                  batch_size: int = 64,
                  n_epochs: int = 10,
                  learning_rate: float = 0.0001,
                  max_episodes: int = 100,
                  gamma: float = 0.99,
                  lambda_gae: float = 0.95,
                  save_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Train a reinforcement learning model for trading.
    
    Args:
        data: DataFrame with market data
        window_size: Size of observation window
        initial_balance: Initial account balance
        commission: Trading commission
        reward_scaling: Scaling factor for rewards
        features: List of feature columns to use (if None, use all except price_col)
        price_col: Name of price column
        lstm_units: Number of LSTM units
        lstm_layers: Number of LSTM layers
        policy_hidden_units: List of hidden units for policy head
        value_hidden_units: List of hidden units for value head
        batch_size: Batch size for updates
        n_epochs: Number of epochs per update
        learning_rate: Learning rate
        max_episodes: Maximum number of episodes
        gamma: Discount factor
        lambda_gae: GAE parameter
        save_path: Path to save model
    
    Returns:
        Dictionary of training results
    """
    # Determine features
    if features is None:
        features = [col for col in data.columns if col != price_col]
    
    # Create environment
    env = TradingEnvironment(
        data=data,
        window_size=window_size,
        initial_balance=initial_balance,
        commission=commission,
        reward_scaling=reward_scaling,
        features=features,
        price_col=price_col
    )
    
    # Determine input dimension
    input_dim = len(features)
    
    # Create agent
    agent = PPOAgent(
        input_dim=input_dim,
        lstm_units=lstm_units,
        lstm_layers=lstm_layers,
        policy_hidden_units=policy_hidden_units,
        value_hidden_units=value_hidden_units,
        action_dim=3,  # 0: Hold, 1: Buy, 2: Sell
        gamma=gamma,
        lambda_gae=lambda_gae,
        batch_size=batch_size,
        n_epochs=n_epochs,
        lr=learning_rate
    )
    
    # Training variables
    best_reward = -np.inf
    episode_rewards = []
    episode_profits = []
    episode_lengths = []
    
    # Training loop
    for episode in range(max_episodes):
        # Reset environment
        state = env.reset()
        done = False
        total_reward = 0
        n_steps = 0
        
        # Episode loop
        while not done:
            # Choose action
            action, prob, val = agent.choose_action(state)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, prob, val, reward, done)
            
            # Update state and counters
            state = next_state
            total_reward += reward
            n_steps += 1
        
        # Learn from episode
        metrics = agent.learn()
        
        # Get portfolio history
        history = env.get_portfolio_history()
        
        # Calculate final portfolio value and profit
        final_value = history.iloc[-1]["portfolio_value"]
        profit = final_value - initial_balance
        profit_pct = profit / initial_balance
        
        # Store episode results
        episode_rewards.append(total_reward)
        episode_profits.append(profit_pct)
        episode_lengths.append(n_steps)
        
        # Print progress
        print(f"Episode {episode + 1}/{max_episodes} | "
              f"Reward: {total_reward:.2f} | "
              f"Profit: {profit:.2f} ({profit_pct:.2%}) | "
              f"Steps: {n_steps} | "
              f"Actor Loss: {metrics['actor_loss']:.4f} | "
              f"Critic Loss: {metrics['critic_loss']:.4f}")
        
        # Save best model
        if save_path is not None and profit_pct > best_reward:
            best_reward = profit_pct
            agent.save(f"{save_path}_best")
            print(f"Saved best model with profit: {profit_pct:.2%}")
    
    # Save final model
    if save_path is not None:
        agent.save(save_path)
        print(f"Saved final model")
    
    # Return results
    results = {
        "agent": agent,
        "env": env,
        "episode_rewards": episode_rewards,
        "episode_profits": episode_profits,
        "episode_lengths": episode_lengths,
        "best_reward": best_reward,
        "config": {
            "window_size": window_size,
            "initial_balance": initial_balance,
            "commission": commission,
            "reward_scaling": reward_scaling,
            "lstm_units": lstm_units,
            "lstm_layers": lstm_layers,
            "policy_hidden_units": policy_hidden_units,
            "value_hidden_units": value_hidden_units,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "learning_rate": learning_rate,
            "max_episodes": max_episodes,
            "gamma": gamma,
            "lambda_gae": lambda_gae
        }
    }
    
    return results


def evaluate_rl_model(agent: PPOAgent,
                     test_data: pd.DataFrame,
                     window_size: int = 20,
                     initial_balance: float = 10000.0,
                     commission: float = 0.001,
                     features: Optional[List[str]] = None,
                     price_col: str = "close") -> Dict[str, Any]:
    """
    Evaluate a trained reinforcement learning model.
    
    Args:
        agent: Trained PPO agent
        test_data: DataFrame with test market data
        window_size: Size of observation window
        initial_balance: Initial account balance
        commission: Trading commission
        features: List of feature columns to use (if None, use all except price_col)
        price_col: Name of price column
    
    Returns:
        Dictionary of evaluation results
    """
    # Create environment
    env = TradingEnvironment(
        data=test_data,
        window_size=window_size,
        initial_balance=initial_balance,
        commission=commission,
        features=features,
        price_col=price_col,
        debug=True
    )
    
    # Reset environment
    state = env.reset()
    done = False
    
    # Episode loop
    while not done:
        # Choose action
        action, _, _ = agent.choose_action(state)
        
        # Take action
        state, _, done, _ = env.step(action)
    
    # Get portfolio history
    history = env.get_portfolio_history()
    
    # Calculate final portfolio value and profit
    final_value = history.iloc[-1]["portfolio_value"]
    profit = final_value - initial_balance
    profit_pct = profit / initial_balance
    
    # Calculate buy-and-hold baseline
    first_price = test_data[price_col].iloc[window_size]
    last_price = test_data[price_col].iloc[-1]
    baseline_profit_pct = last_price / first_price - 1
    
    # Calculate Sharpe ratio
    daily_returns = history["portfolio_value"].pct_change().dropna()
    sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)  # Annualized
    
    # Calculate drawdown
    peak = history["portfolio_value"].cummax()
    drawdown = (history["portfolio_value"] - peak) / peak
    max_drawdown = drawdown.min()
    
    # Calculate trade statistics
    trades = env.trades
    buy_trades = [t for t in trades if t["type"] == "buy"]
    sell_trades = [t for t in trades if t["type"] == "sell"]
    
    n_trades = len(sell_trades)
    
    if n_trades > 0:
        profitable_trades = [t for t in sell_trades if t["profit"] > 0]
        win_rate = len(profitable_trades) / n_trades
        
        avg_profit = np.mean([t["profit_pct"] for t in sell_trades])
        avg_win = np.mean([t["profit_pct"] for t in profitable_trades]) if profitable_trades else 0
        avg_loss = np.mean([t["profit_pct"] for t in sell_trades if t["profit"] <= 0]) if n_trades > len(profitable_trades) else 0
        
        profit_factor = -np.sum([t["profit"] for t in profitable_trades]) / np.sum([t["profit"] for t in sell_trades if t["profit"] <= 0]) if n_trades > len(profitable_trades) and np.sum([t["profit"] for t in sell_trades if t["profit"] <= 0]) != 0 else np.inf
    else:
        win_rate = 0
        avg_profit = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
    
    # Print results
    print(f"Final portfolio value: {final_value:.2f}")
    print(f"Profit: {profit:.2f} ({profit_pct:.2%})")
    print(f"Buy-and-hold profit: {baseline_profit_pct:.2%}")
    print(f"Outperformance: {profit_pct - baseline_profit_pct:.2%}")
    print(f"Sharpe ratio: {sharpe_ratio:.2f}")
    print(f"Max drawdown: {max_drawdown:.2%}")
    print(f"Number of trades: {n_trades}")
    print(f"Win rate: {win_rate:.2%}")
    print(f"Average profit per trade: {avg_profit:.2%}")
    print(f"Average win: {avg_win:.2%}")
    print(f"Average loss: {avg_loss:.2%}")
    print(f"Profit factor: {profit_factor:.2f}")
    
    # Return results
    results = {
        "history": history,
        "trades": trades,
        "final_value": final_value,
        "profit": profit,
        "profit_pct": profit_pct,
        "baseline_profit_pct": baseline_profit_pct,
        "outperformance": profit_pct - baseline_profit_pct,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "avg_profit": avg_profit,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor
    }
    
    return results


def run_rl_experiment(train_data: pd.DataFrame,
                    test_data: pd.DataFrame,
                    window_size: int = 20,
                    initial_balance: float = 10000.0,
                    commission: float = 0.001,
                    features: Optional[List[str]] = None,
                    price_col: str = "close",
                    lstm_units: int = 64,
                    lstm_layers: int = 2,
                    max_episodes: int = 100,
                    save_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Run a complete reinforcement learning experiment.
    
    Args:
        train_data: DataFrame with training market data
        test_data: DataFrame with test market data
        window_size: Size of observation window
        initial_balance: Initial account balance
        commission: Trading commission
        features: List of feature columns to use (if None, use all except price_col)
        price_col: Name of price column
        lstm_units: Number of LSTM units
        lstm_layers: Number of LSTM layers
        max_episodes: Maximum number of episodes
        save_path: Path to save model
    
    Returns:
        Dictionary of experiment results
    """
    # Train model
    train_results = train_rl_model(
        data=train_data,
        window_size=window_size,
        initial_balance=initial_balance,
        commission=commission,
        features=features,
        price_col=price_col,
        lstm_units=lstm_units,
        lstm_layers=lstm_layers,
        max_episodes=max_episodes,
        save_path=save_path
    )
    
    # Evaluate on test data
    test_results = evaluate_rl_model(
        agent=train_results["agent"],
        test_data=test_data,
        window_size=window_size,
        initial_balance=initial_balance,
        commission=commission,
        features=features,
        price_col=price_col
    )
    
    # Combine results
    results = {
        "train_results": train_results,
        "test_results": test_results,
        "config": train_results["config"]
    }
    
    return results