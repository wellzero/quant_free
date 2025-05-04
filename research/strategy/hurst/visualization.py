import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .core import hurst_rs, hurst_dfa, hurst_aggvar, hurst_higuchi

def plot_hurst_methods_comparison(time_series, max_lag=20):
    """
    Plot comparison of different Hurst exponent estimation methods
    
    Parameters:
    time_series (array): Time series data
    max_lag (int): Maximum lag to consider
    """
    # Calculate Hurst exponents using different methods
    h_rs = hurst_rs(time_series, max_lag)
    h_dfa = hurst_dfa(time_series)
    h_aggvar = hurst_aggvar(time_series, max_lag)
    h_higuchi = hurst_higuchi(time_series, max_lag)
    
    # Convert NumPy arrays to scalar values if needed
    h_rs = float(h_rs) if hasattr(h_rs, 'item') else h_rs
    h_dfa = float(h_dfa) if hasattr(h_dfa, 'item') else h_dfa
    h_aggvar = float(h_aggvar) if hasattr(h_aggvar, 'item') else h_aggvar
    h_higuchi = float(h_higuchi) if hasattr(h_higuchi, 'item') else h_higuchi
    
    # Create a bar plot
    methods = ['R/S Analysis', 'DFA', 'Aggregated Variance', 'Higuchi']
    values = [h_rs, h_dfa, h_aggvar, h_higuchi]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, values)
    
    # Add a horizontal line at H=0.5 (random walk)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random Walk (H=0.5)')
    
    # Add labels and title
    plt.ylabel('Hurst Exponent')
    plt.title('Comparison of Hurst Exponent Estimation Methods')
    
    # Add the values on top of the bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, 
                 bar.get_height() + 0.01, 
                 f'{value:.3f}', 
                 ha='center')
    
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    return plt.gcf()

def plot_rolling_hurst(time_series, dates=None, window_size=100, step=10, method='rs', max_lag=20):
    """
    Plot rolling Hurst exponent
    
    Parameters:
    time_series (array): Time series data
    dates (array): Dates corresponding to the time series
    window_size (int): Size of the rolling window
    step (int): Step size for the rolling window
    method (str): Method to use ('rs', 'dfa', 'aggvar', 'higuchi')
    max_lag (int): Maximum lag to consider
    """
    from .core import rolling_hurst
    
    indices, hurst_values = rolling_hurst(time_series, window_size, step, method, max_lag)
    
    plt.figure(figsize=(12, 6))
    
    if dates is not None:
        plot_dates = [dates[i] for i in indices]
        plt.plot(plot_dates, hurst_values)
        plt.xlabel('Date')
    else:
        plt.plot(indices, hurst_values)
        plt.xlabel('Index')
    
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random Walk (H=0.5)')
    plt.axhline(y=0.45, color='g', linestyle='--', label='Mean Reversion Threshold')
    plt.axhline(y=0.55, color='b', linestyle='--', label='Trending Threshold')
    
    plt.ylabel('Hurst Exponent')
    plt.title(f'Rolling Hurst Exponent ({method.upper()} method, window={window_size})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    return plt.gcf()

def plot_strategy_results(strategy_df, title="Hurst Trading Strategy Results"):
    """
    Plot strategy results
    
    Parameters:
    strategy_df (DataFrame): Strategy results DataFrame
    title (str): Plot title
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})
    
    # Plot price and positions
    ax1 = axes[0]
    ax1.plot(strategy_df.index, strategy_df['price'], label='Price', color='blue')
    
    # Add buy/sell markers
    buy_signals = strategy_df[strategy_df['position'].diff() > 0].index
    sell_signals = strategy_df[strategy_df['position'].diff() < 0].index
    
    ax1.scatter(buy_signals, strategy_df.loc[buy_signals, 'price'], marker='^', color='green', s=100, label='Buy')
    ax1.scatter(sell_signals, strategy_df.loc[sell_signals, 'price'], marker='v', color='red', s=100, label='Sell')
    
    ax1.set_ylabel('Price')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True)
    
    # Plot Hurst exponent
    ax2 = axes[1]
    ax2.plot(strategy_df.index, strategy_df['hurst'], color='purple', label='Hurst Exponent')
    ax2.axhline(y=0.5, color='r', linestyle='--', label='Random Walk (H=0.5)')
    ax2.axhline(y=0.45, color='g', linestyle='--', label='Mean Reversion Threshold')
    ax2.axhline(y=0.55, color='b', linestyle='--', label='Trending Threshold')
    ax2.set_ylabel('Hurst Exponent')
    ax2.legend()
    ax2.grid(True)
    
    # Plot cumulative returns
    ax3 = axes[2]
    if 'cum_returns' in strategy_df.columns and 'cum_strategy_returns' in strategy_df.columns:
        ax3.plot(strategy_df.index, strategy_df['cum_returns'], label='Buy & Hold', color='blue')
        ax3.plot(strategy_df.index, strategy_df['cum_strategy_returns'], label='Strategy', color='green')
    else:
        # Calculate cumulative returns if not already in DataFrame
        strategy_df['cum_returns'] = (1 + strategy_df['returns']).cumprod()
        strategy_df['cum_strategy_returns'] = (1 + strategy_df['strategy_returns']).cumprod()
        ax3.plot(strategy_df.index, strategy_df['cum_returns'], label='Buy & Hold', color='blue')
        ax3.plot(strategy_df.index, strategy_df['cum_strategy_returns'], label='Strategy', color='green')
    
    ax3.set_ylabel('Cumulative Returns')
    ax3.set_xlabel('Date')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    return fig