import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.performance import calculate_returns_metrics, plot_equity_curve
from utils.risk import calculate_risk_metrics, calculate_position_sizing

def time_series_momentum(prices_df, lookback_period=252, holding_period=63, volatility_lookback=63, 
                         volatility_scaling=True, target_volatility=0.15):
    """
    Implements time series momentum strategy (Strategy 1 in '151 Trading Strategies')
    
    Parameters:
    prices_df (DataFrame): DataFrame with 'Close' prices
    lookback_period (int): Lookback period for momentum calculation
    holding_period (int): Holding period for positions
    volatility_lookback (int): Lookback period for volatility calculation
    volatility_scaling (bool): Whether to scale positions by volatility
    target_volatility (float): Target annualized volatility when scaling
    
    Returns:
    DataFrame: Strategy results
    """
    # Validate inputs
    if not isinstance(prices_df, pd.DataFrame):
        raise ValueError("prices_df must be a pandas DataFrame")
    
    if 'Close' not in prices_df.columns:
        raise ValueError("prices_df must contain a 'Close' column")
    
    # Calculate returns
    returns = prices_df['Close'].pct_change().dropna()
    
    # Calculate momentum signal (past returns over lookback period)
    momentum_signal = returns.rolling(window=lookback_period).apply(
        lambda x: (1 + x).prod() - 1, raw=True
    ).dropna()
    
    # Generate positions: 1 for positive momentum, -1 for negative momentum
    positions = np.sign(momentum_signal)
    
    # If volatility scaling is enabled, adjust position sizes
    if volatility_scaling:
        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=volatility_lookback).std() * np.sqrt(252)
        
        # Scale positions by volatility
        vol_scalar = target_volatility / rolling_vol
        positions = positions * vol_scalar
    
    # Shift positions to implement them on the next day
    positions = positions.shift(1).dropna()
    
    # Calculate strategy returns
    strategy_returns = positions * returns.loc[positions.index]
    
    # Create results DataFrame
    results = pd.DataFrame({
        'returns': returns.loc[positions.index],
        'momentum_signal': momentum_signal.loc[positions.index],
        'position': positions,
        'strategy_returns': strategy_returns
    })
    
    # Calculate cumulative returns
    results['cumulative_returns'] = (1 + results['returns']).cumprod()
    results['strategy_cumulative_returns'] = (1 + results['strategy_returns']).cumprod()
    
    return results

def cross_sectional_momentum(prices_dict, lookback_period=252, holding_period=63, 
                            n_top=10, n_bottom=10, long_short=True):
    """
    Implements cross-sectional momentum strategy (Strategy 2 in '151 Trading Strategies')
    
    Parameters:
    prices_dict (dict): Dictionary of DataFrames with 'Close' prices for each asset
    lookback_period (int): Lookback period for momentum calculation
    holding_period (int): Holding period for positions
    n_top (int): Number of top performers to go long
    n_bottom (int): Number of bottom performers to go short
    long_short (bool): Whether to implement long-short (True) or long-only (False)
    
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
    for i in range(lookback_period, len(returns), holding_period):
        if i + holding_period > len(returns):
            end_idx = len(returns)
        else:
            end_idx = i + holding_period
        
        # Calculate momentum for each asset
        momentum_values = {}
        for asset in returns.columns:
            # Calculate momentum as cumulative return over lookback period
            asset_returns = returns[asset].iloc[i-lookback_period:i]
            momentum = (1 + asset_returns).prod() - 1
            momentum_values[asset] = momentum
        
        # Sort assets by momentum
        sorted_assets = sorted(momentum_values.items(), key=lambda x: x[1], reverse=True)
        
        # Select top and bottom assets
        top_assets = [asset for asset, _ in sorted_assets[:n_top]]
        bottom_assets = [asset for asset, _ in sorted_assets[-n_bottom:]] if long_short else []
        
        # Set weights for the holding period
        for date_idx in range(i, end_idx):
            if date_idx >= len(weights):
                break
                
            # Long top assets
            for asset in top_assets:
                weights.iloc[date_idx, weights.columns.get_loc(asset)] = 1.0 / len(top_assets)
            
            # Short bottom assets if long-short
            if long_short:
                for asset in bottom_assets:
                    weights.iloc[date_idx, weights.columns.get_loc(asset)] = -1.0 / len(bottom_assets)
    
    # Calculate strategy returns
    strategy_returns = (weights * returns).sum(axis=1)
    
    # Calculate cumulative returns
    cumulative_returns = (1 + strategy_returns).cumprod()
    
    # Calculate performance metrics
    performance = calculate_returns_metrics(strategy_returns)
    risk = calculate_risk_metrics(strategy_returns)
    
    return {
        'weights': weights,
        'returns': strategy_returns,
        'cumulative_returns': cumulative_returns,
        'performance': performance,
        'risk': risk
    }

def momentum_factor(prices_df, lookback_periods=[1, 3, 6, 12], weights=None):
    """
    Implements momentum factor model (Strategy 3 in '151 Trading Strategies')
    
    Parameters:
    prices_df (DataFrame): DataFrame with 'Close' prices
    lookback_periods (list): List of lookback periods in months
    weights (list): Weights for each lookback period (default: equal weights)
    
    Returns:
    DataFrame: Momentum factor scores
    """
    # Validate inputs
    if not isinstance(prices_df, pd.DataFrame):
        raise ValueError("prices_df must be a pandas DataFrame")
    
    if 'Close' not in prices_df.columns:
        raise ValueError("prices_df must contain a 'Close' column")
    
    # If weights not provided, use equal weights
    if weights is None:
        weights = [1.0 / len(lookback_periods)] * len(lookback_periods)
    
    if len(weights) != len(lookback_periods):
        raise ValueError("Length of weights must match length of lookback_periods")
    
    # Calculate returns
    returns = prices_df['Close'].pct_change().dropna()
    
    # Assume daily data, convert months to days
    lookback_days = [period * 21 for period in lookback_periods]  # Approx. 21 trading days per month
    
    # Calculate momentum for each lookback period
    momentum_scores = pd.DataFrame(index=returns.index)
    
    for i, period in enumerate(lookback_days):
        # Skip if period is longer than available data
        if period >= len(returns):
            continue
            
        # Calculate momentum as rolling return over the period
        period_momentum = returns.rolling(window=period).apply(
            lambda x: (1 + x).prod() - 1, raw=True
        ).dropna()
        
        # Add to momentum scores with weight
        momentum_scores[f'momentum_{lookback_periods[i]}m'] = period_momentum * weights[i]
    
    # Calculate combined momentum score
    momentum_scores['momentum_factor'] = momentum_scores.sum(axis=1)
    
    return momentum_scores

def dual_momentum(prices_dict, lookback_period=252, risk_free_rate=0.0):
    """
    Implements dual momentum strategy (absolute and relative momentum)
    
    Parameters:
    prices_dict (dict): Dictionary of DataFrames with 'Close' prices for each asset
    lookback_period (int): Lookback period for momentum calculation
    risk_free_rate (float): Annualized risk-free rate
    
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
    
    # Convert annual risk-free rate to daily
    daily_rf = (1 + risk_free_rate) ** (1/252) - 1
    
    # Initialize portfolio weights DataFrame
    weights = pd.DataFrame(0, index=returns.index, columns=returns.columns)
    
    # For each day after the lookback period
    for i in range(lookback_period, len(returns)):
        # Calculate absolute momentum (vs. risk-free rate)
        absolute_momentum = {}
        for asset in returns.columns:
            # Calculate momentum as cumulative return over lookback period
            asset_returns = returns[asset].iloc[i-lookback_period:i]
            momentum = (1 + asset_returns).prod() - 1
            absolute_momentum[asset] = momentum > (1 + daily_rf) ** lookback_period - 1
        
        # Calculate relative momentum (vs. other assets)
        momentum_values = {asset: (1 + returns[asset].iloc[i-lookback_period:i]).prod() - 1 
                          for asset in returns.columns}
        
        # Find asset with highest momentum
        best_asset = max(momentum_values.items(), key=lambda x: x[1])[0]
        
        # Invest in the best asset only if it has positive absolute momentum
        if absolute_momentum[best_asset]:
            weights.iloc[i, weights.columns.get_loc(best_asset)] = 1.0
    
    # Shift weights by 1 day to avoid look-ahead bias
    weights = weights.shift(1).fillna(0)
    
    # Calculate strategy returns
    strategy_returns = (weights * returns).sum(axis=1)
    
    # Calculate cumulative returns
    cumulative_returns = (1 + strategy_returns).cumprod()
    
    # Calculate performance metrics
    performance = calculate_returns_metrics(strategy_returns)
    risk = calculate_risk_metrics(strategy_returns)
    
    return {
        'weights': weights,
        'returns': strategy_returns,
        'cumulative_returns': cumulative_returns,
        'performance': performance,
        'risk': risk
    }

def momentum_with_volatility_control(prices_df, lookback_period=252, vol_lookback=63, 
                                    target_volatility=0.10, max_leverage=2.0):
    """
    Implements momentum strategy with volatility targeting
    
    Parameters:
    prices_df (DataFrame): DataFrame with 'Close' prices
    lookback_period (int): Lookback period for momentum calculation
    vol_lookback (int): Lookback period for volatility calculation
    target_volatility (float): Target annualized volatility
    max_leverage (float): Maximum allowed leverage
    
    Returns:
    DataFrame: Strategy results
    """
    # Validate inputs
    if not isinstance(prices_df, pd.DataFrame):
        raise ValueError("prices_df must be a pandas DataFrame")
    
    if 'Close' not in prices_df.columns:
        raise ValueError("prices_df must contain a 'Close' column")
    
    # Calculate returns
    returns = prices_df['Close'].pct_change().dropna()
    
    # Calculate momentum signal (past returns over lookback period)
    momentum_signal = returns.rolling(window=lookback_period).apply(
        lambda x: (1 + x).prod() - 1, raw=True
    ).dropna()
    
    # Generate base positions: 1 for positive momentum, -1 for negative momentum
    base_positions = np.sign(momentum_signal)
    
    # Calculate rolling volatility (annualized)
    rolling_vol = returns.rolling(window=vol_lookback).std() * np.sqrt(252)
    
    # Calculate position scaling factor based on target volatility
    vol_scalar = target_volatility / rolling_vol
    
    # Apply maximum leverage constraint
    vol_scalar = vol_scalar.clip(upper=max_leverage)
    
    # Calculate final positions
    positions = base_positions * vol_scalar
    
    # Shift positions to implement them on the next day
    positions = positions.shift(1).dropna()
    
    # Calculate strategy returns
    strategy_returns = positions * returns.loc[positions.index]
    
    # Create results DataFrame
    results = pd.DataFrame({
        'returns': returns.loc[positions.index],
        'momentum_signal': momentum_signal.loc[positions.index],
        'volatility': rolling_vol.loc[positions.index],
        'position': positions,
        'strategy_returns': strategy_returns
    })
    
    # Calculate cumulative returns
    results['cumulative_returns'] = (1 + results['returns']).cumprod()
    results['strategy_cumulative_returns'] = (1 + results['strategy_returns']).cumprod()
    
    return results

def plot_momentum_strategy(strategy_results, title="Momentum Strategy Performance"):
    """
    Plot momentum strategy performance
    
    Parameters:
    strategy_results (DataFrame): Results from a momentum strategy function
    title (str): Plot title
    
    Returns:
    matplotlib.figure.Figure: The figure object
    """
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Plot cumulative returns
    ax1.plot(strategy_results['cumulative_returns'], label='Buy & Hold', alpha=0.7)
    ax1.plot(strategy_results['strategy_cumulative_returns'], label='Strategy', linewidth=1.5)
    ax1.set_title(title)
    ax1.set_ylabel('Cumulative Return')
    ax1.legend()
    ax1.grid(True)
    
    # Plot momentum signal
    ax2.plot(strategy_results['momentum_signal'], color='purple')
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_ylabel('Momentum Signal')
    ax2.grid(True)
    
    # Plot positions
    ax3.plot(strategy_results['position'], color='green')
    ax3.set_ylabel('Position Size')
    ax3.set_xlabel('Date')
    ax3.grid(True)
    
    plt.tight_layout()
    return fig