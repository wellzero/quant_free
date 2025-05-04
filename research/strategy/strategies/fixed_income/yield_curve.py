import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.performance import calculate_returns_metrics, plot_equity_curve
from utils.risk import calculate_risk_metrics, calculate_var

def yield_curve_steepener(short_yield, long_yield, lookback_period=60, z_score_threshold=1.5,
                         holding_period=20, stop_loss_z=3.0, take_profit_z=0.5):
    """
    Implements yield curve steepener/flattener strategy (Strategy 30 in '151 Trading Strategies')
    
    Parameters:
    short_yield (Series): Series of short-term yields (e.g., 2-year)
    long_yield (Series): Series of long-term yields (e.g., 10-year)
    lookback_period (int): Lookback period for z-score calculation
    z_score_threshold (float): Z-score threshold for trade entry
    holding_period (int): Maximum holding period for positions
    stop_loss_z (float): Stop loss threshold in terms of z-score
    take_profit_z (float): Take profit threshold in terms of z-score
    
    Returns:
    DataFrame: Strategy results
    """
    # Validate inputs
    if not isinstance(short_yield, pd.Series):
        raise ValueError("short_yield must be a pandas Series")
    
    if not isinstance(long_yield, pd.Series):
        raise ValueError("long_yield must be a pandas Series")
    
    # Calculate yield spread (long - short)
    spread = long_yield - short_yield
    
    # Calculate z-score of spread
    spread_mean = spread.rolling(window=lookback_period).mean()
    spread_std = spread.rolling(window=lookback_period).std()
    z_score = (spread - spread_mean) / spread_std
    
    # Initialize positions DataFrame
    positions = pd.DataFrame(0, index=spread.index, columns=['position'])
    
    # Generate signals
    # Steepener trade: short when spread is tight (negative z-score)
    # Flattener trade: long when spread is wide (positive z-score)
    positions.loc[z_score < -z_score_threshold, 'position'] = 1  # Steepener
    positions.loc[z_score > z_score_threshold, 'position'] = -1  # Flattener
    
    # Apply holding period logic and stop loss/take profit
    current_position = 0
    entry_z = 0
    entry_date = None
    days_held = 0
    
    for i, date in enumerate(positions.index):
        if i < lookback_period:
            positions.loc[date, 'position'] = 0
            continue
            
        # Check if we need to close existing position
        if current_position != 0:
            days_held += 1
            current_z = z_score.loc[date]
            
            # Stop loss
            if (current_position == 1 and current_z < entry_z - stop_loss_z) or \
               (current_position == -1 and current_z > entry_z + stop_loss_z):
                positions.loc[date, 'position'] = 0
                current_position = 0
                days_held = 0
                continue
                
            # Take profit
            if (current_position == 1 and current_z > entry_z + take_profit_z) or \
               (current_position == -1 and current_z < entry_z - take_profit_z):
                positions.loc[date, 'position'] = 0
                current_position = 0
                days_held = 0
                continue
                
            # Max holding period reached
            if days_held >= holding_period:
                positions.loc[date, 'position'] = 0
                current_position = 0
                days_held = 0
                continue
                
            # Maintain current position
            positions.loc[date, 'position'] = current_position
        
        # Check for new entry signal
        elif positions.loc[date, 'position'] != 0:
            current_position = positions.loc[date, 'position']
            entry_z = z_score.loc[date]
            entry_date = date
            days_held = 0
    
    # Calculate strategy returns
    # For simplicity, we assume equal notional exposure to both legs
    # In practice, DV01 matching would be used for proper risk balancing
    short_returns = -short_yield.pct_change()  # Negative because bond prices move inversely to yields
    long_returns = -long_yield.pct_change()
    
    # Strategy returns: long the spread (long long-term, short short-term) or vice versa
    strategy_returns = positions.shift(1)['position'] * (long_returns - short_returns)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'spread': spread,
        'z_score': z_score,
        'position': positions['position'],
        'returns': strategy_returns
    })
    
    # Calculate cumulative returns
    results['cumulative_returns'] = (1 + results['returns']).cumprod()
    
    return results

def butterfly_trade(short_yield, mid_yield, long_yield, lookback_period=60, z_score_threshold=2.0,
                   holding_period=20, stop_loss_z=4.0, take_profit_z=1.0):
    """
    Implements butterfly trade strategy (Strategy 31 in '151 Trading Strategies')
    
    Parameters:
    short_yield (Series): Series of short-term yields (e.g., 2-year)
    mid_yield (Series): Series of mid-term yields (e.g., 5-year)
    long_yield (Series): Series of long-term yields (e.g., 10-year)
    lookback_period (int): Lookback period for z-score calculation
    z_score_threshold (float): Z-score threshold for trade entry
    holding_period (int): Maximum holding period for positions
    stop_loss_z (float): Stop loss threshold in terms of z-score
    take_profit_z (float): Take profit threshold in terms of z-score
    
    Returns:
    DataFrame: Strategy results
    """
    # Validate inputs
    if not isinstance(short_yield, pd.Series):
        raise ValueError("short_yield must be a pandas Series")
    
    if not isinstance(mid_yield, pd.Series):
        raise ValueError("mid_yield must be a pandas Series")
        
    if not isinstance(long_yield, pd.Series):
        raise ValueError("long_yield must be a pandas Series")
    
    # Calculate butterfly spread: 2*mid - (short + long)
    butterfly = 2 * mid_yield - (short_yield + long_yield)
    
    # Calculate z-score of butterfly
    butterfly_mean = butterfly.rolling(window=lookback_period).mean()
    butterfly_std = butterfly.rolling(window=lookback_period).std()
    z_score = (butterfly - butterfly_mean) / butterfly_std
    
    # Initialize positions DataFrame
    positions = pd.DataFrame(0, index=butterfly.index, columns=['position'])
    
    # Generate signals
    # Long butterfly: when butterfly is tight (negative z-score)
    # Short butterfly: when butterfly is wide (positive z-score)
    positions.loc[z_score < -z_score_threshold, 'position'] = 1  # Long butterfly
    positions.loc[z_score > z_score_threshold, 'position'] = -1  # Short butterfly
    
    # Apply holding period logic and stop loss/take profit
    # Similar implementation as yield_curve_steepener
    current_position = 0
    entry_z = 0
    entry_date = None
    days_held = 0
    
    for i, date in enumerate(positions.index):
        if i < lookback_period:
            positions.loc[date, 'position'] = 0
            continue
            
        # Check if we need to close existing position
        if current_position != 0:
            days_held += 1
            current_z = z_score.loc[date]
            
            # Stop loss
            if (current_position == 1 and current_z < entry_z - stop_loss_z) or \
               (current_position == -1 and current_z > entry_z + stop_loss_z):
                positions.loc[date, 'position'] = 0
                current_position = 0
                days_held = 0
                continue
                
            # Take profit
            if (current_position == 1 and current_z > entry_z + take_profit_z) or \
               (current_position == -1 and current_z < entry_z - take_profit_z):
                positions.loc[date, 'position'] = 0
                current_position = 0
                days_held = 0
                continue
                
            # Max holding period reached
            if days_held >= holding_period:
                positions.loc[date, 'position'] = 0
                current_position = 0
                days_held = 0
                continue
                
            # Maintain current position
            positions.loc[date, 'position'] = current_position
        
        # Check for new entry signal
        elif positions.loc[date, 'position'] != 0:
            current_position = positions.loc[date, 'position']
            entry_z = z_score.loc[date]
            entry_date = date
            days_held = 0
    
    # Calculate strategy returns
    # For a long butterfly: long 2x mid-term, short short-term and long-term
    # For a short butterfly: short 2x mid-term, long short-term and long-term
    short_returns = -short_yield.pct_change()
    mid_returns = -mid_yield.pct_change()
    long_returns = -long_yield.pct_change()
    
    # Strategy returns
    strategy_returns = positions.shift(1)['position'] * (2 * mid_returns - (short_returns + long_returns))
    
    # Create results DataFrame
    results = pd.DataFrame({
        'butterfly': butterfly,
        'z_score': z_score,
        'position': positions['position'],
        'returns': strategy_returns
    })
    
    # Calculate cumulative returns
    results['cumulative_returns'] = (1 + results['returns']).cumprod()
    
    return results

def plot_yield_curve_strategy(results, title="Yield Curve Strategy Performance"):
    """
    Plot yield curve strategy results
    
    Parameters:
    results (DataFrame): Strategy results from yield_curve_steepener or butterfly_trade
    title (str): Plot title
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Plot spread and z-score
    if 'spread' in results.columns:
        axes[0].plot(results.index, results['spread'], label='Yield Spread')
        axes[0].set_ylabel('Spread (%)')
        axes[0].set_title('Yield Spread')
        axes[0].legend()
        
        axes[1].plot(results.index, results['z_score'], label='Z-Score', color='orange')
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1].axhline(y=1.5, color='red', linestyle='--', alpha=0.3)
        axes[1].axhline(y=-1.5, color='green', linestyle='--', alpha=0.3)
        axes[1].set_ylabel('Z-Score')
        axes[1].set_title('Spread Z-Score')
        axes[1].legend()
    elif 'butterfly' in results.columns:
        axes[0].plot(results.index, results['butterfly'], label='Butterfly Spread')
        axes[0].set_ylabel('Spread (%)')
        axes[0].set_title('Butterfly Spread')
        axes[0].legend()
        
        axes[1].plot(results.index, results['z_score'], label='Z-Score', color='orange')
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1].axhline(y=2.0, color='red', linestyle='--', alpha=0.3)
        axes[1].axhline(y=-2.0, color='green', linestyle='--', alpha=0.3)
        axes[1].set_ylabel('Z-Score')
        axes[1].set_title('Butterfly Z-Score')
        axes[1].legend()
    
    # Plot cumulative returns
    axes[2].plot(results.index, results['cumulative_returns'], label='Strategy Returns', color='green')
    axes[2].set_ylabel('Cumulative Returns')
    axes[2].set_title('Strategy Performance')
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print performance metrics
    print("\nPerformance Metrics:")
    metrics = calculate_returns_metrics(results['returns'].dropna())
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Print risk metrics
    print("\nRisk Metrics:")
    risk_metrics = calculate_risk_metrics(results['returns'].dropna())
    for key, value in risk_metrics.items():
        print(f"{key}: {value:.4f}")