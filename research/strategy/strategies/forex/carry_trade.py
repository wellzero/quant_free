import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.performance import calculate_returns_metrics, plot_equity_curve
from utils.risk import calculate_risk_metrics, calculate_var

def carry_trade_strategy(exchange_rates_df, interest_rates_df, n_pairs=3, rebalance_frequency=21,
                        volatility_lookback=63, volatility_scaling=True, target_volatility=0.10):
    """
    Implements currency carry trade strategy (Strategy 59 in '151 Trading Strategies')
    
    Parameters:
    exchange_rates_df (DataFrame): DataFrame with exchange rates (vs USD) for different currencies
    interest_rates_df (DataFrame): DataFrame with interest rates for different currencies
    n_pairs (int): Number of currency pairs to include in the portfolio
    rebalance_frequency (int): Frequency of rebalancing in days
    volatility_lookback (int): Lookback period for volatility calculation
    volatility_scaling (bool): Whether to scale positions by volatility
    target_volatility (float): Target annualized volatility when scaling
    
    Returns:
    dict: Strategy results
    """
    # Validate inputs
    if not isinstance(exchange_rates_df, pd.DataFrame):
        raise ValueError("exchange_rates_df must be a pandas DataFrame")
    
    if not isinstance(interest_rates_df, pd.DataFrame):
        raise ValueError("interest_rates_df must be a pandas DataFrame")
    
    # Ensure both DataFrames have the same currencies
    common_currencies = list(set(exchange_rates_df.columns).intersection(set(interest_rates_df.columns)))
    if len(common_currencies) < 2:
        raise ValueError("At least two common currencies required in both DataFrames")
    
    # Filter DataFrames to include only common currencies
    exchange_rates = exchange_rates_df[common_currencies]
    interest_rates = interest_rates_df[common_currencies]
    
    # Calculate daily returns for exchange rates
    fx_returns = exchange_rates.pct_change().dropna()
    
    # Initialize portfolio weights DataFrame
    weights = pd.DataFrame(0, index=fx_returns.index, columns=fx_returns.columns)
    
    # Initialize results dictionary
    results = {
        'portfolio_returns': pd.Series(0, index=fx_returns.index),
        'cumulative_returns': pd.Series(1, index=fx_returns.index),
        'positions': pd.DataFrame(0, index=fx_returns.index, columns=fx_returns.columns),
        'interest_differentials': pd.DataFrame(0, index=fx_returns.index, columns=fx_returns.columns)
    }
    
    # For each rebalancing date
    rebalance_dates = []
    for i in range(0, len(fx_returns), rebalance_frequency):
        if i + volatility_lookback >= len(fx_returns):
            break
            
        current_date = fx_returns.index[i]
        rebalance_dates.append(current_date)
        
        # Get current interest rates
        current_rates = interest_rates.loc[current_date]
        
        # Calculate interest rate differentials vs USD (assuming USD is included)
        if 'USD' in current_rates:
            usd_rate = current_rates['USD']
            interest_diff = current_rates - usd_rate
        else:
            # If USD not included, use average rate as reference
            avg_rate = current_rates.mean()
            interest_diff = current_rates - avg_rate
        
        # Sort currencies by interest rate differential
        sorted_currencies = interest_diff.sort_values(ascending=False)
        
        # Select top n_pairs high-yielding currencies to go long
        high_yield = sorted_currencies.index[:n_pairs]
        
        # Select bottom n_pairs low-yielding currencies to go short
        low_yield = sorted_currencies.index[-n_pairs:]
        
        # Calculate equal weights for long and short positions
        long_weight = 1.0 / n_pairs
        short_weight = -1.0 / n_pairs
        
        # Assign weights
        for currency in high_yield:
            weights.loc[current_date:, currency] = long_weight
        
        for currency in low_yield:
            weights.loc[current_date:, currency] = short_weight
            
        # Store interest differentials for this period
        for currency in common_currencies:
            results['interest_differentials'].loc[current_date:, currency] = interest_diff[currency]
    
    # Apply volatility scaling if enabled
    if volatility_scaling:
        # Calculate rolling volatility of portfolio returns
        portfolio_returns = (weights.shift(1) * fx_returns).sum(axis=1)
        rolling_vol = portfolio_returns.rolling(window=volatility_lookback).std() * np.sqrt(252)
        
        # Scale weights by volatility
        vol_scalar = target_volatility / rolling_vol
        vol_scalar = vol_scalar.clip(upper=3.0)  # Cap leverage at 3x
        
        # Apply scaling to weights
        for date in weights.index:
            if date in rolling_vol.index and not np.isnan(rolling_vol[date]):
                weights.loc[date] = weights.loc[date] * vol_scalar[date]
    
    # Calculate strategy returns
    # FX return component
    fx_return_component = (weights.shift(1) * fx_returns).sum(axis=1)
    
    # Interest rate component (carry)
    carry_component = pd.Series(0, index=fx_returns.index)
    for date in fx_returns.index:
        if date in results['interest_differentials'].index:
            # Daily interest rate differential (annualized rates / 252 trading days)
            daily_carry = (weights.loc[date] * results['interest_differentials'].loc[date] / 252).sum()
            carry_component.loc[date] = daily_carry
    
    # Total returns = FX return + Carry
    results['portfolio_returns'] = fx_return_component + carry_component
    
    # Calculate cumulative returns
    results['cumulative_returns'] = (1 + results['portfolio_returns']).cumprod()
    
    # Store positions
    results['positions'] = weights
    
    return results

def plot_carry_trade_strategy(results, title="Currency Carry Trade Strategy Performance"):
    """
    Plot carry trade strategy results
    
    Parameters:
    results (dict): Strategy results from carry_trade_strategy
    title (str): Plot title
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot positions over time (stacked area chart)
    positions = results['positions']
    dates = positions.index
    
    # Separate long and short positions for better visualization
    long_pos = positions.copy()
    short_pos = positions.copy()
    
    long_pos[long_pos < 0] = 0
    short_pos[short_pos > 0] = 0
    
    # Plot long positions
    axes[0].stackplot(dates, long_pos.T.values, labels=long_pos.columns, alpha=0.7)
    
    # Plot short positions
    axes[0].stackplot(dates, short_pos.T.values, labels=short_pos.columns, alpha=0.7)
    
    axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[0].set_ylabel('Position Size')
    axes[0].set_title('Currency Positions Over Time')
    axes[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # Plot cumulative returns
    axes[1].plot(results['cumulative_returns'].index, results['cumulative_returns'], 
                label='Strategy Returns', color='green')
    axes[1].set_ylabel('Cumulative Returns')
    axes[1].set_title('Strategy Performance')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print performance metrics
    print("\nPerformance Metrics:")
    metrics = calculate_returns_metrics(results['portfolio_returns'].dropna())
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Print risk metrics
    print("\nRisk Metrics:")
    risk_metrics = calculate_risk_metrics(results['portfolio_returns'].dropna())
    for key, value in risk_metrics.items():
        print(f"{key}: {value:.4f}")

def interest_rate_differential_strategy(exchange_rates_df, interest_rates_df, lookback_period=63, 
                                      z_score_threshold=1.0, holding_period=21, n_currencies=5):
    """
    Implements interest rate differential strategy (Strategy 64 in '151 Trading Strategies')
    
    Parameters:
    exchange_rates_df (DataFrame): DataFrame with exchange rates (vs USD) for different currencies
    interest_rates_df (DataFrame): DataFrame with interest rates for different currencies
    lookback_period (int): Lookback period for z-score calculation
    z_score_threshold (float): Z-score threshold for trade entry
    holding_period (int): Maximum holding period for positions
    n_currencies (int): Number of currencies to include in the portfolio
    
    Returns:
    dict: Strategy results
    """
    # Validate inputs
    if not isinstance(exchange_rates_df, pd.DataFrame):
        raise ValueError("exchange_rates_df must be a pandas DataFrame")
    
    if not isinstance(interest_rates_df, pd.DataFrame):
        raise ValueError("interest_rates_df must be a pandas DataFrame")
    
    # Ensure both DataFrames have the same currencies
    common_currencies = list(set(exchange_rates_df.columns).intersection(set(interest_rates_df.columns)))
    if len(common_currencies) < n_currencies:
        raise ValueError(f"At least {n_currencies} common currencies required in both DataFrames")
    
    # Filter DataFrames to include only common currencies
    exchange_rates = exchange_rates_df[common_currencies]
    interest_rates = interest_rates_df[common_currencies]
    
    # Calculate daily returns for exchange rates
    fx_returns = exchange_rates.pct_change().dropna()
    
    # Calculate interest rate differentials
    if 'USD' in interest_rates.columns:
        # Calculate differentials against USD
        interest_diff = interest_rates.sub(interest_rates['USD'], axis=0)
    else:
        # Calculate differentials against average rate
        interest_diff = interest_rates.sub(interest_rates.mean(axis=1), axis=0)
    
    # Calculate z-scores of interest rate differentials
    z_scores = pd.DataFrame(0, index=interest_diff.index, columns=interest_diff.columns)
    
    for currency in interest_diff.columns:
        rolling_mean = interest_diff[currency].rolling(window=lookback_period).mean()
        rolling_std = interest_diff[currency].rolling(window=lookback_period).std()
        z_scores[currency] = (interest_diff[currency] - rolling_mean) / rolling_std
    
    # Initialize positions DataFrame
    positions = pd.DataFrame(0, index=fx_returns.index, columns=fx_returns.columns)
    
    # Generate signals based on z-scores
    for i, date in enumerate(positions.index):
        if i < lookback_period:
            continue
            
        # Get current z-scores
        current_z = z_scores.loc[date]
        
        # Sort currencies by z-score
        sorted_z = current_z.sort_values(ascending=False)
        
        # Select top currencies with highest z-scores to go short
        # (high z-score means interest rate is high relative to history, expect mean reversion)
        top_currencies = sorted_z.index[:n_currencies]
        
        # Select bottom currencies with lowest z-scores to go long
        # (low z-score means interest rate is low relative to history, expect mean reversion)
        bottom_currencies = sorted_z.index[-n_currencies:]
        
        # Assign equal weights
        for currency in top_currencies:
            if current_z[currency] > z_score_threshold:
                positions.loc[date, currency] = -1.0 / n_currencies
        
        for currency in bottom_currencies:
            if current_z[currency] < -z_score_threshold:
                positions.loc[date, currency] = 1.0 / n_currencies
    
    # Apply holding period logic
    for currency in positions.columns:
        entry_signals = positions[currency].ne(0) & positions[currency].shift(1).eq(0)
        entry_dates = positions.index[entry_signals]
        
        for entry_date in entry_dates:
            entry_idx = positions.index.get_loc(entry_date)
            exit_idx = min(entry_idx + holding_period, len(positions) - 1)
            
            # Maintain position for holding period
            position_value = positions.loc[entry_date, currency]
            positions.loc[positions.index[entry_idx:exit_idx], currency] = position_value
    
    # Calculate strategy returns
    # FX return component
    fx_return_component = (positions.shift(1) * fx_returns).sum(axis=1)
    
    # Interest rate component (carry)
    carry_component = pd.Series(0, index=fx_returns.index)
    for date in fx_returns.index:
        if date in interest_diff.index:
            # Daily interest rate differential (annualized rates / 252 trading days)
            daily_carry = (positions.loc[date] * interest_diff.loc[date] / 252).sum()
            carry_component.loc[date] = daily_carry
    
    # Total returns = FX return + Carry
    portfolio_returns = fx_return_component + carry_component
    
    # Calculate cumulative returns
    cumulative_returns = (1 + portfolio_returns).cumprod()
    
    # Create results dictionary
    results = {
        'portfolio_returns': portfolio_returns,
        'cumulative_returns': cumulative_returns,
        'positions': positions,
        'z_scores': z_scores,
        'interest_differentials': interest_diff
    }
    
    return results