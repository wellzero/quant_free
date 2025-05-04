import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.performance import calculate_returns_metrics, plot_equity_curve
from utils.risk import calculate_risk_metrics
from utils.optimization import risk_parity_weights

def risk_parity_portfolio(prices_dict, lookback_period=252, rebalance_frequency=21, 
                         risk_budget=None, target_volatility=0.10, max_leverage=1.0):
    """
    Implements risk parity portfolio strategy (Strategy 101 in '151 Trading Strategies')
    
    Parameters:
    prices_dict (dict): Dictionary of DataFrames with 'Close' prices for each asset
    lookback_period (int): Lookback period for covariance estimation
    rebalance_frequency (int): Frequency of rebalancing in days
    risk_budget (list, optional): Target risk contribution for each asset (equal by default)
    target_volatility (float): Target annualized volatility
    max_leverage (float): Maximum allowed leverage
    
    Returns:
    dict: Strategy results
    """
    # Validate inputs
    if not isinstance(prices_dict, dict):
        raise ValueError("prices_dict must be a dictionary of DataFrames")
    
    # Extract close prices for all assets
    close_prices = {}
    for asset, df in prices_dict.items():
        if 'Close' not in df.columns:
            raise ValueError(f"DataFrame for {asset} must contain a 'Close' column")
        close_prices[asset] = df['Close']
    
    # Combine close prices into a single DataFrame
    prices = pd.DataFrame(close_prices)
    
    # Calculate returns for all assets
    returns = prices.pct_change().dropna()
    
    # Initialize portfolio weights DataFrame
    weights = pd.DataFrame(0, index=returns.index, columns=returns.columns)
    
    # For each rebalancing date
    for i in range(lookback_period, len(returns), rebalance_frequency):
        if i + rebalance_frequency > len(returns):
            end_idx = len(returns)
        else:
            end_idx = i + rebalance_frequency
        
        # Calculate risk parity weights using lookback period
        asset_returns = returns.iloc[i-lookback_period:i]
        rp_weights = risk_parity_weights(asset_returns, risk_budget)
        
        # Scale weights to target volatility
        portfolio_vol = np.sqrt(np.dot(rp_weights.T, np.dot(asset_returns.cov(), rp_weights))) * np.sqrt(252)
        vol_scalar = target_volatility / portfolio_vol
        
        # Apply maximum leverage constraint
        vol_scalar = min(vol_scalar, max_leverage)
        scaled_weights = rp_weights * vol_scalar
        
        # Set weights for the rebalancing period
        for date_idx in range(i, end_idx):
            if date_idx >= len(weights):
                break
            weights.iloc[date_idx] = scaled_weights
    
    # Shift weights by 1 day to avoid look-ahead bias
    weights = weights.shift(1).fillna(0)
    
    # Calculate strategy returns
    strategy_returns = (weights * returns).sum(axis=1)
    
    # Calculate cumulative returns
    cumulative_returns = (1 + strategy_returns).cumprod()
    
    # Calculate performance metrics
    performance = calculate_returns_metrics(strategy_returns.dropna())
    risk = calculate_risk_metrics(strategy_returns.dropna())
    
    return {
        'weights': weights,
        'returns': strategy_returns,
        'cumulative_returns': cumulative_returns,
        'performance': performance,
        'risk': risk
    }

def risk_parity_with_momentum(prices_dict, lookback_period=252, momentum_period=63, 
                             rebalance_frequency=21, target_volatility=0.10, 
                             momentum_filter=True, min_assets=2):
    """
    Implements risk parity portfolio with momentum filter
    
    Parameters:
    prices_dict (dict): Dictionary of DataFrames with 'Close' prices for each asset
    lookback_period (int): Lookback period for covariance estimation
    momentum_period (int): Lookback period for momentum calculation
    rebalance_frequency (int): Frequency of rebalancing in days
    target_volatility (float): Target annualized volatility
    momentum_filter (bool): Whether to filter assets by positive momentum
    min_assets (int): Minimum number of assets to include
    
    Returns:
    dict: Strategy results
    """
    # Validate inputs
    if not isinstance(prices_dict, dict):
        raise ValueError("prices_dict must be a dictionary of DataFrames")
    
    # Extract close prices for all assets
    close_prices = {}
    for asset, df in prices_dict.items():
        if 'Close' not in df.columns:
            raise ValueError(f"DataFrame for {asset} must contain a 'Close' column")
        close_prices[asset] = df['Close']
    
    # Combine close prices into a single DataFrame
    prices = pd.DataFrame(close_prices)
    
    # Calculate returns for all assets
    returns = prices.pct_change().dropna()
    
    # Initialize portfolio weights DataFrame
    weights = pd.DataFrame(0, index=returns.index, columns=returns.columns)
    
    # For each rebalancing date
    for i in range(lookback_period, len(returns), rebalance_frequency):
        if i + rebalance_frequency > len(returns):
            end_idx = len(returns)
        else:
            end_idx = i + rebalance_frequency
        
        # Calculate momentum for each asset
        momentum_values = {}
        for asset in returns.columns:
            # Calculate momentum as cumulative return over momentum period
            asset_returns = returns[asset].iloc[i-momentum_period:i]
            momentum = (1 + asset_returns).prod() - 1
            momentum_values[asset] = momentum
        
        # Filter assets by momentum if enabled
        selected_assets = returns.columns.tolist()
        if momentum_filter:
            # Select assets with positive momentum
            positive_momentum_assets = [asset for asset, mom in momentum_values.items() if mom > 0]
            
            # Ensure minimum number of assets
            if len(positive_momentum_assets) >= min_assets:
                selected_assets = positive_momentum_assets
            else:
                # If not enough assets have positive momentum, select top assets by momentum
                sorted_assets = sorted(momentum_values.items(), key=lambda x: x[1], reverse=True)
                selected_assets = [asset for asset, _ in sorted_assets[:min_assets]]
        
        # Calculate risk parity weights using lookback period for selected assets
        asset_returns = returns[selected_assets].iloc[i-lookback_period:i]
        
        # Skip if not enough data or assets
        if len(asset_returns) < 2 or len(selected_assets) < 2:
            continue
            
        rp_weights = risk_parity_weights(asset_returns)
        
        # Scale weights to target volatility
        portfolio_vol = np.sqrt(np.dot(rp_weights.T, np.dot(asset_returns.cov(), rp_weights))) * np.sqrt(252)
        vol_scalar = target_volatility / portfolio_vol
        scaled_weights = rp_weights * vol_scalar
        
        # Create full weights vector with zeros for non-selected assets
        full_weights = pd.Series(0, index=returns.columns)
        for asset in selected_assets:
            full_weights[asset] = scaled_weights[asset]
        
        # Set weights for the rebalancing period
        for date_idx in range(i, end_idx):
            if date_idx >= len(weights):
                break
            weights.iloc[date_idx] = full_weights
    
    # Shift weights by 1 day to avoid look-ahead bias
    weights = weights.shift(1).fillna(0)
    
    # Calculate strategy returns
    strategy_returns = (weights * returns).sum(axis=1)
    
    # Calculate cumulative returns
    cumulative_returns = (1 + strategy_returns).cumprod()
    
    # Calculate performance metrics
    performance = calculate_returns_metrics(strategy_returns.dropna())
    risk = calculate_risk_metrics(strategy_returns.dropna())
    
    return {
        'weights': weights,
        'returns': strategy_returns,
        'cumulative_returns': cumulative_returns,
        'performance': performance,
        'risk': risk
    }

def risk_parity_with_regime_switching(prices_dict, lookback_period=252, vol_lookback=63, 
                                     rebalance_frequency=21, target_volatility=0.10,
                                     high_vol_threshold=0.15, low_vol_threshold=0.10,
                                     high_vol_target=0.05, low_vol_target=0.15):
    """
    Implements risk parity portfolio with regime switching based on volatility
    
    Parameters:
    prices_dict (dict): Dictionary of DataFrames with 'Close' prices for each asset
    lookback_period (int): Lookback period for covariance estimation
    vol_lookback (int): Lookback period for volatility calculation
    rebalance_frequency (int): Frequency of rebalancing in days
    target_volatility (float): Base target annualized volatility
    high_vol_threshold (float): Threshold for high volatility regime
    low_vol_threshold (float): Threshold for low volatility regime
    high_vol_target (float): Target volatility in high volatility regime
    low_vol_target (float): Target volatility in low volatility regime
    
    Returns:
    dict: Strategy results
    """
    # Validate inputs
    if not isinstance(prices_dict, dict):
        raise ValueError("prices_dict must be a dictionary of DataFrames")
    
    # Extract close prices for all assets
    close_prices = {}
    for asset, df in prices_dict.items():
        if 'Close' not in df.columns:
            raise ValueError(f"DataFrame for {asset} must contain a 'Close' column")
        close_prices[asset] = df['Close']
    
    # Combine close prices into a single DataFrame
    prices = pd.DataFrame(close_prices)
    
    # Calculate returns for all assets
    returns = prices.pct_change().dropna()
    
    # Calculate market proxy (equal-weighted portfolio)
    market_returns = returns.mean(axis=1)
    
    # Calculate rolling volatility of market
    market_vol = market_returns.rolling(window=vol_lookback).std() * np.sqrt(252)
    
    # Initialize portfolio weights DataFrame
    weights = pd.DataFrame(0, index=returns.index, columns=returns.columns)
    
    # For each rebalancing date
    for i in range(lookback_period, len(returns), rebalance_frequency):
        if i + rebalance_frequency > len(returns):
            end_idx = len(returns)
        else:
            end_idx = i + rebalance_frequency
        
        # Determine current volatility regime
        current_vol = market_vol.iloc[i-1]
        
        # Set target volatility based on regime
        if current_vol > high_vol_threshold:
            # High volatility regime - reduce risk
            current_target_vol = high_vol_target
        elif current_vol < low_vol_threshold:
            # Low volatility regime - increase risk
            current_target_vol = low_vol_target
        else:
            # Normal regime - use base target
            current_target_vol = target_volatility
        
        # Calculate risk parity weights using lookback period
        asset_returns = returns.iloc[i-lookback_period:i]
        rp_weights = risk_parity_weights(asset_returns)
        
        # Scale weights to target volatility
        portfolio_vol = np.sqrt(np.dot(rp_weights.T, np.dot(asset_returns.cov(), rp_weights))) * np.sqrt(252)
        vol_scalar = current_target_vol / portfolio_vol
        scaled_weights = rp_weights * vol_scalar
        
        # Set weights for the rebalancing period
        for date_idx in range(i, end_idx):
            if date_idx >= len(weights):
                break
            weights.iloc[date_idx] = scaled_weights
    
    # Shift weights by 1 day to avoid look-ahead bias
    weights = weights.shift(1).fillna(0)
    
    # Calculate strategy returns
    strategy_returns = (weights * returns).sum(axis=1)
    
    # Calculate cumulative returns
    cumulative_returns = (1 + strategy_returns).cumprod()
    
    # Calculate performance metrics
    performance = calculate_returns_metrics(strategy_returns.dropna())
    risk = calculate_risk_metrics(strategy_returns.dropna())
    
    return {
        'weights': weights,
        'returns': strategy_returns,
        'cumulative_returns': cumulative_returns,
        'market_vol': market_vol,
        'performance': performance,
        'risk': risk
    }

def plot_risk_parity_portfolio(results, title="Risk Parity Portfolio Performance"):
    """
    Plot risk parity portfolio performance
    
    Parameters:
    results (dict): Results from a risk parity portfolio function
    title (str): Plot title
    
    Returns:
    matplotlib.figure.Figure: The figure object
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 2]})
    
    # Plot cumulative returns
    ax1.plot(results['cumulative_returns'], label='Strategy', linewidth=1.5)
    ax1.set_title(title)
    ax1.set_ylabel('Cumulative Return')
    ax1.legend()
    ax1.grid(True)
    
    # Plot asset weights over time
    weights = results['weights']
    
    # Plot stacked area chart of weights
    ax2.stackplot(weights.index, weights.T, labels=weights.columns, alpha=0.7)
    ax2.set_ylabel('Portfolio Weight')
    ax2.set_xlabel('Date')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

def plot_regime_switching_portfolio(results, title="Regime Switching Portfolio Performance"):
    """
    Plot regime switching portfolio performance
    
    Parameters:
    results (dict): Results from a regime switching portfolio function
    title (str): Plot title
    
    Returns:
    matplotlib.figure.Figure: The figure object
    """
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [3, 1, 2]})
    
    # Plot cumulative returns
    ax1.plot(results['cumulative_returns'], label='Strategy', linewidth=1.5)
    ax1.set_title(title)
    ax1.set_ylabel('Cumulative Return')
    ax1.legend()
    ax1.grid(True)
    
    # Plot market volatility
    ax2.plot(results['market_vol'], color='red', label='Market Volatility')
    ax2.axhline(y=0.15, color='r', linestyle='--', label='High Vol Threshold')
    ax2.axhline(y=0.10, color='g', linestyle='--', label='Low Vol Threshold')
    ax2.set_ylabel('Annualized Volatility')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True)
    
    # Plot asset weights over time
    weights = results['weights']
    
    # Plot stacked area chart of weights
    ax3.stackplot(weights.index, weights.T, labels=weights.columns, alpha=0.7)
    ax3.set_ylabel('Portfolio Weight')
    ax3.set_xlabel('Date')
    ax3.legend(loc='upper left', fontsize=8)
    ax3.grid(True)
    
    plt.tight_layout()
    return fig