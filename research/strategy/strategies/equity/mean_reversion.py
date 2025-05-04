import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.performance import calculate_returns_metrics, plot_equity_curve
from utils.risk import calculate_risk_metrics, calculate_position_sizing

def bollinger_band_strategy(prices_df, window=20, num_std=2.0, holding_period=1, 
                           partial_exit=False, stop_loss_std=3.0):
    """
    Implements Bollinger Band mean reversion strategy
    
    Parameters:
    prices_df (DataFrame): DataFrame with 'Close' prices
    window (int): Lookback window for moving average calculation
    num_std (float): Number of standard deviations for bands
    holding_period (int): Maximum holding period for positions
    partial_exit (bool): Whether to exit positions partially when price reverts to mean
    stop_loss_std (float): Stop loss as multiple of standard deviation
    
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
    
    # Calculate Bollinger Bands
    rolling_mean = prices_df['Close'].rolling(window=window).mean()
    rolling_std = prices_df['Close'].rolling(window=window).std()
    
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    
    # Generate signals
    signals = pd.Series(0, index=prices_df.index)
    
    # Short signal when price crosses above upper band
    signals[prices_df['Close'] > upper_band] = -1
    
    # Long signal when price crosses below lower band
    signals[prices_df['Close'] < lower_band] = 1
    
    # Initialize position and holding period counters
    position = 0
    entry_price = 0
    days_held = 0
    positions = pd.Series(0, index=prices_df.index)
    
    # Process signals to generate positions
    for i in range(1, len(signals)):
        current_date = signals.index[i]
        prev_date = signals.index[i-1]
        
        # If in a position, increment holding period counter
        if position != 0:
            days_held += 1
        
        # Check for exit conditions
        if position != 0:
            # Exit if maximum holding period reached
            if days_held >= holding_period:
                position = 0
                days_held = 0
            
            # Partial exit if price reverts to mean and partial_exit is enabled
            elif partial_exit and ((position > 0 and prices_df.loc[current_date, 'Close'] >= rolling_mean[current_date]) or 
                                  (position < 0 and prices_df.loc[current_date, 'Close'] <= rolling_mean[current_date])):
                position = position / 2  # Reduce position size by half
            
            # Stop loss if price moves too far against position
            elif ((position > 0 and prices_df.loc[current_date, 'Close'] < entry_price - stop_loss_std * rolling_std[current_date]) or 
                  (position < 0 and prices_df.loc[current_date, 'Close'] > entry_price + stop_loss_std * rolling_std[current_date])):
                position = 0
                days_held = 0
        
        # Enter new position if not already in one
        if position == 0 and signals[current_date] != 0:
            position = signals[current_date]
            entry_price = prices_df.loc[current_date, 'Close']
            days_held = 0
        
        positions[current_date] = position
    
    # Calculate strategy returns
    strategy_returns = positions.shift(1) * returns
    
    # Create results DataFrame
    results = pd.DataFrame({
        'returns': returns,
        'position': positions,
        'upper_band': upper_band,
        'lower_band': lower_band,
        'rolling_mean': rolling_mean,
        'strategy_returns': strategy_returns
    })
    
    # Calculate cumulative returns
    results['cumulative_returns'] = (1 + results['returns']).cumprod()
    results['strategy_cumulative_returns'] = (1 + results['strategy_returns']).cumprod()
    
    return results

def rsi_mean_reversion(prices_df, rsi_period=14, overbought=70, oversold=30, 
                      holding_period=5, stop_loss_pct=0.05):
    """
    Implements RSI-based mean reversion strategy
    
    Parameters:
    prices_df (DataFrame): DataFrame with 'Close' prices
    rsi_period (int): Period for RSI calculation
    overbought (float): RSI threshold for overbought condition
    oversold (float): RSI threshold for oversold condition
    holding_period (int): Maximum holding period for positions
    stop_loss_pct (float): Stop loss percentage
    
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
    
    # Calculate RSI
    delta = prices_df['Close'].diff().dropna()
    gains = delta.copy()
    losses = delta.copy()
    
    gains[gains < 0] = 0
    losses[losses > 0] = 0
    losses = abs(losses)
    
    avg_gain = gains.rolling(window=rsi_period).mean()
    avg_loss = losses.rolling(window=rsi_period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Generate signals
    signals = pd.Series(0, index=prices_df.index)
    
    # Short signal when RSI is overbought
    signals[rsi > overbought] = -1
    
    # Long signal when RSI is oversold
    signals[rsi < oversold] = 1
    
    # Initialize position and holding period counters
    position = 0
    entry_price = 0
    days_held = 0
    positions = pd.Series(0, index=prices_df.index)
    
    # Process signals to generate positions
    for i in range(1, len(signals)):
        current_date = signals.index[i]
        prev_date = signals.index[i-1]
        
        # If in a position, increment holding period counter
        if position != 0:
            days_held += 1
        
        # Check for exit conditions
        if position != 0:
            # Exit if maximum holding period reached
            if days_held >= holding_period:
                position = 0
                days_held = 0
            
            # Stop loss if price moves too far against position
            elif ((position > 0 and prices_df.loc[current_date, 'Close'] < entry_price * (1 - stop_loss_pct)) or 
                  (position < 0 and prices_df.loc[current_date, 'Close'] > entry_price * (1 + stop_loss_pct))):
                position = 0
                days_held = 0
            
            # Exit long if RSI moves above 50
            elif position > 0 and rsi[current_date] > 50:
                position = 0
                days_held = 0
            
            # Exit short if RSI moves below 50
            elif position < 0 and rsi[current_date] < 50:
                position = 0
                days_held = 0
        
        # Enter new position if not already in one
        if position == 0 and signals[current_date] != 0:
            position = signals[current_date]
            entry_price = prices_df.loc[current_date, 'Close']
            days_held = 0
        
        positions[current_date] = position
    
    # Calculate strategy returns
    strategy_returns = positions.shift(1) * returns
    
    # Create results DataFrame
    results = pd.DataFrame({
        'returns': returns,
        'position': positions,
        'rsi': rsi,
        'strategy_returns': strategy_returns
    })
    
    # Calculate cumulative returns
    results['cumulative_returns'] = (1 + results['returns']).cumprod()
    results['strategy_cumulative_returns'] = (1 + results['strategy_returns']).cumprod()
    
    return results

def mean_reversion_with_hurst(prices_df, lookback_period=100, hurst_threshold=0.4, 
                             window=20, num_std=2.0, holding_period=5):
    """
    Implements mean reversion strategy with Hurst exponent filter
    
    Parameters:
    prices_df (DataFrame): DataFrame with 'Close' prices
    lookback_period (int): Lookback period for Hurst exponent calculation
    hurst_threshold (float): Threshold below which mean reversion is considered strong
    window (int): Lookback window for moving average calculation
    num_std (float): Number of standard deviations for bands
    holding_period (int): Maximum holding period for positions
    
    Returns:
    DataFrame: Strategy results
    """
    # Import Hurst exponent calculation from existing module
    sys.path.append('/home/quant_volumn/quant_free/research/strategy')
    from hurst_exponent import hurst_rs
    
    # Validate inputs
    if not isinstance(prices_df, pd.DataFrame):
        raise ValueError("prices_df must be a pandas DataFrame")
    
    if 'Close' not in prices_df.columns:
        raise ValueError("prices_df must contain a 'Close' column")
    
    # Calculate returns
    returns = prices_df['Close'].pct_change().dropna()
    
    # Calculate Bollinger Bands
    rolling_mean = prices_df['Close'].rolling(window=window).mean()
    rolling_std = prices_df['Close'].rolling(window=window).std()
    
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    
    # Calculate Hurst exponent over rolling window
    hurst_values = pd.Series(index=prices_df.index)
    
    for i in range(lookback_period, len(prices_df)):
        price_window = prices_df['Close'].iloc[i-lookback_period:i].values
        h = hurst_rs(price_window)
        hurst_values.iloc[i] = h
    
    # Generate signals
    signals = pd.Series(0, index=prices_df.index)
    
    # Only generate signals when Hurst exponent indicates mean reversion
    for i in range(lookback_period, len(prices_df)):
        current_date = prices_df.index[i]
        
        # Check if Hurst exponent indicates mean reversion
        if hurst_values.iloc[i] < hurst_threshold:
            # Short signal when price crosses above upper band
            if prices_df['Close'].iloc[i] > upper_band.iloc[i]:
                signals.iloc[i] = -1
            
            # Long signal when price crosses below lower band
            elif prices_df['Close'].iloc[i] < lower_band.iloc[i]:
                signals.iloc[i] = 1
    
    # Initialize position and holding period counters
    position = 0
    days_held = 0
    positions = pd.Series(0, index=prices_df.index)
    
    # Process signals to generate positions
    for i in range(1, len(signals)):
        current_date = signals.index[i]
        
        # If in a position, increment holding period counter
        if position != 0:
            days_held += 1
        
        # Exit if maximum holding period reached
        if position != 0 and days_held >= holding_period:
            position = 0
            days_held = 0
        
        # Enter new position if not already in one
        if position == 0 and signals.iloc[i] != 0:
            position = signals.iloc[i]
            days_held = 0
        
        positions.iloc[i] = position
    
    # Calculate strategy returns
    strategy_returns = positions.shift(1) * returns
    
    # Create results DataFrame
    results = pd.DataFrame({
        'returns': returns,
        'position': positions,
        'upper_band': upper_band,
        'lower_band': lower_band,
        'rolling_mean': rolling_mean,
        'hurst': hurst_values,
        'strategy_returns': strategy_returns
    })
    
    # Calculate cumulative returns
    results['cumulative_returns'] = (1 + results['returns']).cumprod()
    results['strategy_cumulative_returns'] = (1 + results['strategy_returns']).cumprod()
    
    return results

def statistical_arbitrage(prices_dict, lookback_period=252, z_threshold=2.0, 
                         half_life=20, holding_period=20):
    """
    Implements statistical arbitrage (pairs trading) strategy
    
    Parameters:
    prices_dict (dict): Dictionary of DataFrames with 'Close' prices for each asset
    lookback_period (int): Lookback period for cointegration and spread calculation
    z_threshold (float): Z-score threshold for entry/exit
    half_life (int): Half-life for mean reversion estimation
    holding_period (int): Maximum holding period for positions
    
    Returns:
    dict: Strategy results
    """
    from scipy import stats
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import coint
    
    # Validate inputs
    if not isinstance(prices_dict, dict) or len(prices_dict) < 2:
        raise ValueError("prices_dict must be a dictionary with at least two assets")
    
    # Extract close prices for all assets
    close_prices = {}
    for asset, df in prices_dict.items():
        if 'Close' not in df.columns:
            raise ValueError(f"DataFrame for {asset} must contain a 'Close' column")
        close_prices[asset] = df['Close']
    
    # Combine close prices into a single DataFrame
    prices = pd.DataFrame(close_prices)
    
    # Find the most cointegrated pair
    assets = list(prices.columns)
    best_pvalue = 1.0
    best_pair = (assets[0], assets[1])
    
    for i in range(len(assets)):
        for j in range(i+1, len(assets)):
            asset1 = assets[i]
            asset2 = assets[j]
            
            # Test for cointegration
            _, pvalue, _ = coint(prices[asset1], prices[asset2])
            
            if pvalue < best_pvalue:
                best_pvalue = pvalue
                best_pair = (asset1, asset2)
    
    # Extract the best pair
    asset1, asset2 = best_pair
    
    # Calculate hedge ratio using OLS regression
    model = sm.OLS(prices[asset1], prices[asset2]).fit()
    hedge_ratio = model.params[0]
    
    # Calculate spread
    spread = prices[asset1] - hedge_ratio * prices[asset2]
    
    # Calculate z-score of spread
    spread_mean = spread.rolling(window=lookback_period).mean()
    spread_std = spread.rolling(window=lookback_period).std()
    z_score = (spread - spread_mean) / spread_std
    
    # Generate signals
    signals = pd.Series(0, index=prices.index)
    
    # Short spread (short asset1, long asset2) when z-score is high
    signals[z_score > z_threshold] = -1
    
    # Long spread (long asset1, short asset2) when z-score is low
    signals[z_score < -z_threshold] = 1
    
    # Initialize position and holding period counters
    position = 0
    days_held = 0
    positions = pd.Series(0, index=prices.index)
    
    # Process signals to generate positions
    for i in range(lookback_period, len(signals)):
        current_date = signals.index[i]
        
        # If in a position, increment holding period counter
        if position != 0:
            days_held += 1
        
        # Check for exit conditions
        if position != 0:
            # Exit if maximum holding period reached
            if days_held >= holding_period:
                position = 0
                days_held = 0
            
            # Exit long spread if z-score crosses above 0
            elif position > 0 and z_score.iloc[i] > 0:
                position = 0
                days_held = 0
            
            # Exit short spread if z-score crosses below 0
            elif position < 0 and z_score.iloc[i] < 0:
                position = 0
                days_held = 0
        
        # Enter new position if not already in one
        if position == 0 and signals.iloc[i] != 0:
            position = signals.iloc[i]
            days_held = 0
        
        positions.iloc[i] = position
    
    # Calculate asset returns
    returns1 = prices[asset1].pct_change()
    returns2 = prices[asset2].pct_change()
    
    # Calculate strategy returns
    # When position = 1: long asset1, short asset2
    # When position = -1: short asset1, long asset2
    strategy_returns = positions.shift(1) * (returns1 - hedge_ratio * returns2)
    
    # Calculate cumulative returns
    cumulative_returns = (1 + strategy_returns).cumprod()
    
    # Calculate performance metrics
    performance = calculate_returns_metrics(strategy_returns.dropna())
    risk = calculate_risk_metrics(strategy_returns.dropna())
    
    return {
        'pair': best_pair,
        'hedge_ratio': hedge_ratio,
        'spread': spread,
        'z_score': z_score,
        'positions': positions,
        'returns': strategy_returns,
        'cumulative_returns': cumulative_returns,
        'performance': performance,
        'risk': risk
    }

def plot_mean_reversion_strategy(results, title="Mean Reversion Strategy Performance"):
    """
    Plot mean reversion strategy performance
    
    Parameters:
    results (DataFrame): Results from a mean reversion strategy function
    title (str): Plot title
    
    Returns:
    matplotlib.figure.Figure: The figure object
    """
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Plot cumulative returns
    ax1.plot(results['cumulative_returns'], label='Buy & Hold', alpha=0.7)
    ax1.plot(results['strategy_cumulative_returns'], label='Strategy', linewidth=1.5)
    ax1.set_title(title)
    ax1.set_ylabel('Cumulative Return')
    ax1.legend()
    ax1.grid(True)
    
    # Plot price and bands if available
    if all(band in results.columns for band in ['upper_band', 'lower_band', 'rolling_mean']):
        ax2.plot(results.index, results['Close'] if 'Close' in results.columns else results.index.to_series(), label='Price', alpha=0.7)
        ax2.plot(results.index, results['upper_band'], 'r--', label='Upper Band')
        ax2.plot(results.index, results['lower_band'], 'g--', label='Lower Band')
        ax2.plot(results.index, results['rolling_mean'], 'b-', label='Mean', alpha=0.7)
        ax2.set_ylabel('Price')
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True)
    elif 'rsi' in results.columns:
        ax2.plot(results.index, results['rsi'], 'purple')
        ax2.axhline(y=70, color='r', linestyle='--')
        ax2.axhline(y=30, color='g', linestyle='--')
        ax2.axhline(y=50, color='k', linestyle='-', alpha=0.3)
        ax2.set_ylabel('RSI')
        ax2.grid(True)
    elif 'z_score' in results.columns:
        ax2.plot(results.index, results['z_score'], 'purple')
        ax2.axhline(y=2.0, color='r', linestyle='--')
        ax2.axhline(y=-2.0, color='g', linestyle='--')
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.set_ylabel('Z-Score')
        ax2.grid(True)
    
    # Plot positions
    ax3.plot(results.index, results['position'], color='green')
    ax3.set_ylabel('Position')
    ax3.set_xlabel('Date')
    ax3.grid(True)
    
    plt.tight_layout()
    return fig