import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.performance import calculate_returns_metrics, plot_equity_curve
from utils.risk import calculate_risk_metrics, calculate_position_sizing

def commodity_trend_following(prices_df, lookback_periods=[20, 60, 120], volatility_lookback=63, 
                             volatility_scaling=True, target_volatility=0.15):
    """
    Implements commodity trend following strategy (Strategy 88 in '151 Trading Strategies')
    
    Parameters:
    prices_df (DataFrame): DataFrame with 'Close' prices for commodity futures
    lookback_periods (list): List of lookback periods for moving average calculation
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
    
    # Calculate moving averages for each lookback period
    mas = {}
    for period in lookback_periods:
        mas[period] = prices_df['Close'].rolling(window=period).mean()
    
    # Generate signals based on price relative to moving averages
    signals = pd.DataFrame(0, index=prices_df.index, columns=['signal'])
    
    for period in lookback_periods:
        # Long when price is above MA, short when below
        signals.loc[prices_df['Close'] > mas[period], 'signal'] += 1
        signals.loc[prices_df['Close'] < mas[period], 'signal'] -= 1
    
    # Normalize signals to range [-1, 1]
    signals['signal'] = signals['signal'] / len(lookback_periods)
    
    # If volatility scaling is enabled, adjust position sizes
    if volatility_scaling:
        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=volatility_lookback).std() * np.sqrt(252)
        
        # Scale positions by volatility
        vol_scalar = target_volatility / rolling_vol
        vol_scalar = vol_scalar.clip(upper=3.0)  # Cap leverage at 3x
        
        # Apply scaling to signals
        positions = signals['signal'] * vol_scalar
    else:
        positions = signals['signal']
    
    # Calculate strategy returns
    strategy_returns = positions.shift(1) * returns
    
    # Create results DataFrame
    results = pd.DataFrame({
        'close': prices_df['Close'],
        'signal': signals['signal'],
        'position': positions,
        'returns': strategy_returns
    })
    
    # Add moving averages to results
    for period in lookback_periods:
        results[f'ma_{period}'] = mas[period]
    
    # Calculate cumulative returns
    results['cumulative_returns'] = (1 + results['returns']).cumprod()
    
    return results

def commodity_term_structure(front_contract, back_contract, lookback_period=60, z_score_threshold=1.5,
                           holding_period=20, stop_loss_z=3.0):
    """
    Implements commodity term structure strategy (Strategy 93 in '151 Trading Strategies')
    
    Parameters:
    front_contract (Series): Series of front-month contract prices
    back_contract (Series): Series of back-month contract prices
    lookback_period (int): Lookback period for z-score calculation
    z_score_threshold (float): Z-score threshold for trade entry
    holding_period (int): Maximum holding period for positions
    stop_loss_z (float): Stop loss threshold in terms of z-score
    
    Returns:
    DataFrame: Strategy results
    """
    # Validate inputs
    if not isinstance(front_contract, pd.Series):
        raise ValueError("front_contract must be a pandas Series")
    
    if not isinstance(back_contract, pd.Series):
        raise ValueError("back_contract must be a pandas Series")
    
    # Calculate price ratio (back/front)
    # When ratio is high, market is in contango (upward sloping curve)
    # When ratio is low, market is in backwardation (downward sloping curve)
    price_ratio = back_contract / front_contract
    
    # Calculate z-score of ratio
    ratio_mean = price_ratio.rolling(window=lookback_period).mean()
    ratio_std = price_ratio.rolling(window=lookback_period).std()
    z_score = (price_ratio - ratio_mean) / ratio_std
    
    # Initialize positions DataFrame
    positions = pd.DataFrame(0, index=price_ratio.index, columns=['front', 'back'])
    
    # Generate signals
    # When z-score is high (contango is extreme), go long front-month and short back-month
    # When z-score is low (backwardation is extreme), go short front-month and long back-month
    for i, date in enumerate(positions.index):
        if i < lookback_period:
            continue
            
        current_z = z_score.loc[date]
        
        if current_z > z_score_threshold:
            # Extreme contango - expect mean reversion
            positions.loc[date, 'front'] = 1  # Long front-month
            positions.loc[date, 'back'] = -1  # Short back-month
        elif current_z < -z_score_threshold:
            # Extreme backwardation - expect mean reversion
            positions.loc[date, 'front'] = -1  # Short front-month
            positions.loc[date, 'back'] = 1   # Long back-month
    
    # Apply holding period logic and stop loss
    current_position_front = 0
    current_position_back = 0
    entry_z = 0
    entry_date = None
    days_held = 0
    
    for i, date in enumerate(positions.index):
        if i < lookback_period:
            positions.loc[date, 'front'] = 0
            positions.loc[date, 'back'] = 0
            continue
            
        # Check if we need to close existing position
        if current_position_front != 0:
            days_held += 1
            current_z = z_score.loc[date]
            
            # Stop loss
            if (current_position_front == 1 and current_z < entry_z - stop_loss_z) or \
               (current_position_front == -1 and current_z > entry_z + stop_loss_z):
                positions.loc[date, 'front'] = 0
                positions.loc[date, 'back'] = 0
                current_position_front = 0
                current_position_back = 0
                days_held = 0
                continue
                
            # Max holding period reached
            if days_held >= holding_period:
                positions.loc[date, 'front'] = 0
                positions.loc[date, 'back'] = 0
                current_position_front = 0
                current_position_back = 0
                days_held = 0
                continue
                
            # Maintain current position
            positions.loc[date, 'front'] = current_position_front
            positions.loc[date, 'back'] = current_position_back
        
        # Check for new entry signal
        elif positions.loc[date, 'front'] != 0:
            current_position_front = positions.loc[date, 'front']
            current_position_back = positions.loc[date, 'back']
            entry_z = z_score.loc[date]
            entry_date = date
            days_held = 0
    
    # Calculate returns for front and back contracts
    front_returns = front_contract.pct_change().dropna()
    back_returns = back_contract.pct_change().dropna()
    
    # Calculate strategy returns
    strategy_returns = positions.shift(1)['front'] * front_returns + positions.shift(1)['back'] * back_returns
    
    # Create results DataFrame
    results = pd.DataFrame({
        'price_ratio': price_ratio,
        'z_score': z_score,
        'position_front': positions['front'],
        'position_back': positions['back'],
        'returns': strategy_returns
    })
    
    # Calculate cumulative returns
    results['cumulative_returns'] = (1 + results['returns']).cumprod()
    
    return results

def plot_commodity_strategy(results, title="Commodity Strategy Performance", strategy_type="trend"):
    """
    Plot commodity strategy results
    
    Parameters:
    results (DataFrame): Strategy results from commodity_trend_following or commodity_term_structure
    title (str): Plot title
    strategy_type (str): Type of strategy ('trend' or 'term_structure')
    """
    if strategy_type == "trend":
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        
        # Plot price and moving averages
        axes[0].plot(results.index, results['close'], label='Price', color='black')
        
        ma_columns = [col for col in results.columns if col.startswith('ma_')]
        for col in ma_columns:
            period = col.split('_')[1]
            axes[0].plot(results.index, results[col], label=f'MA {period}')
            
        axes[0].set_ylabel('Price')
        axes[0].set_title('Price and Moving Averages')
        axes[0].legend()
        
        # Plot positions
        axes[1].plot(results.index, results['position'], label='Position', color='orange')
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1].set_ylabel('Position Size')
        axes[1].set_title('Strategy Positions')
        axes[1].legend()
        
        # Plot cumulative returns
        axes[2].plot(results.index, results['cumulative_returns'], label='Strategy Returns', color='green')
        axes[2].set_ylabel('Cumulative Returns')
        axes[2].set_title('Strategy Performance')
        axes[2].legend()
        
    elif strategy_type == "term_structure":
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        
        # Plot price ratio and z-score
        axes[0].plot(results.index, results['price_ratio'], label='Price Ratio (Back/Front)')
        axes[0].set_ylabel('Ratio')
        axes[0].set_title('Term Structure Ratio')
        axes[0].legend()
        
        axes[1].plot(results.index, results['z_score'], label='Z-Score', color='orange')
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1].axhline(y=1.5, color='red', linestyle='--', alpha=0.3)
        axes[1].axhline(y=-1.5, color='green', linestyle='--', alpha=0.3)
        axes[1].set_ylabel('Z-Score')
        axes[1].set_title('Term Structure Z-Score')
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