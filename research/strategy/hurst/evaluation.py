import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def evaluate_strategy(strategy_df):
    """
    Evaluate the performance of a trading strategy
    
    Parameters:
    strategy_df (DataFrame): DataFrame with strategy results
    
    Returns:
    dict: Dictionary with performance metrics
    """
    # Filter out rows with no position
    trading_days = strategy_df[strategy_df['position'] != 0]
    
    # Calculate returns
    strategy_returns = strategy_df['strategy_returns'].dropna()
    
    # Calculate performance metrics
    total_return = (1 + strategy_returns).prod() - 1
    
    # Calculate annualized return
    days = len(strategy_returns)
    annual_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
    
    # Calculate Sharpe ratio - FIX: Handle case where std is zero
    if len(strategy_returns) > 0 and strategy_returns.std() > 0:
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
    else:
        sharpe_ratio = 0
    
    # Calculate max drawdown
    cum_returns = (1 + strategy_returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max - 1)
    
    # Filter out inf and nan values
    drawdown = drawdown.replace([np.inf, -np.inf], np.nan).dropna()
    
    max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
    
    # Calculate win rate
    trades = strategy_df[strategy_df['position'] != strategy_df['position'].shift(1)]
    wins = trades[trades['strategy_returns'] > 0]
    win_rate = len(wins) / len(trades) if len(trades) > 0 else 0
    
    # Return metrics as dictionary
    return {
        'total_return': total_return * 100,
        'annual_return': annual_return * 100,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown * 100,
        'win_rate': win_rate * 100,
        'num_trades': len(trades)
    }

def save_strategy_results_to_csv(strategy_df, csv_path):
    """
    Save strategy results to CSV
    
    Parameters:
    strategy_df (DataFrame): DataFrame with strategy results
    csv_path (str): Path to save CSV results
    
    Returns:
    None
    """
    # Save to CSV
    strategy_df.to_csv(csv_path)
    print(f"Strategy results saved to {csv_path}")


def evaluate_strategy_by_type(strategy_df):
    """
    Evaluate strategy performance separately for mean reversion and trend following
    
    Parameters:
    strategy_df (DataFrame): Strategy results DataFrame
    
    Returns:
    dict: Performance metrics by strategy type
    """
    results = {}
    
    # Overall performance
    results['overall'] = evaluate_strategy(strategy_df)
    
    # Performance by strategy type
    for strategy_type in ['mean_reversion', 'trend_following']:
        # Filter trades by strategy type
        type_mask = strategy_df['strategy_type'] == strategy_type
        if type_mask.sum() == 0:
            continue
            
        # Create a copy with only this strategy type's trades
        type_df = strategy_df.copy()
        type_df.loc[~type_mask, 'position'] = 0
        
        # Recalculate returns for this strategy type
        type_df['strategy_returns'] = type_df['position'].shift(1) * type_df['returns']
        type_df['strategy_returns'] = type_df['strategy_returns'].fillna(0)
        type_df['cum_strategy_returns'] = (1 + type_df['strategy_returns']).cumprod()
        
        # Evaluate this strategy type
        results[strategy_type] = evaluate_strategy(type_df)
    
    return results