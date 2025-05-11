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
    # Increase grid line visibility
    plt.grid(axis='both', linestyle='-', alpha=0.9, which='major', linewidth=0.8, color='gray')
    plt.grid(axis='both', linestyle='--', alpha=0.5, which='minor', linewidth=0.5, color='darkgray')
    plt.minorticks_on()
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
    # Increase grid line visibility
    plt.grid(True, which='major', linestyle='-', alpha=0.9, linewidth=0.8, color='gray')
    plt.grid(True, which='minor', linestyle='--', alpha=0.5, linewidth=0.5, color='darkgray')
    plt.minorticks_on()
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
    # Increase grid line visibility
    ax1.grid(True, which='major', linestyle='-', alpha=0.9, linewidth=0.8, color='gray')
    ax1.grid(True, which='minor', linestyle='--', alpha=0.5, linewidth=0.5, color='darkgray')
    ax1.minorticks_on()
    
    # Plot Hurst exponent
    ax2 = axes[1]
    ax2.plot(strategy_df.index, strategy_df['hurst'], color='purple', label='Hurst Exponent')
    ax2.axhline(y=0.5, color='r', linestyle='--', label='Random Walk (H=0.5)')
    ax2.axhline(y=0.45, color='g', linestyle='--', label='Mean Reversion Threshold')
    ax2.axhline(y=0.55, color='b', linestyle='--', label='Trending Threshold')
    ax2.set_ylabel('Hurst Exponent')
    ax2.legend()
    # Increase grid line visibility
    ax2.grid(True, which='major', linestyle='-', alpha=0.9, linewidth=0.8, color='gray')
    ax2.grid(True, which='minor', linestyle='--', alpha=0.5, linewidth=0.5, color='darkgray')
    ax2.minorticks_on()
    
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
    # Increase grid line visibility
    ax3.grid(True, which='major', linestyle='-', alpha=0.9, linewidth=0.8, color='gray')
    ax3.grid(True, which='minor', linestyle='--', alpha=0.5, linewidth=0.5, color='darkgray')
    ax3.minorticks_on()
    
    plt.tight_layout()
    return fig

def plot_strategy_types_performance(strategy_df, title="Strategy Performance by Type"):
    """
    Plot the performance of different strategy types
    
    Parameters:
    strategy_df (DataFrame): Strategy results DataFrame
    title (str): Plot title
    
    Returns:
    matplotlib.figure.Figure: Figure object
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Plot price and positions
    ax1.plot(strategy_df.index, strategy_df['price'], color='black', alpha=0.5)
    
    # Plot positions by strategy type
    mean_rev_mask = strategy_df['strategy_type'] == 'mean_reversion'
    trend_mask = strategy_df['strategy_type'] == 'trend_following'
    
    # Plot mean reversion positions
    if mean_rev_mask.sum() > 0:
        mr_long = strategy_df[mean_rev_mask & (strategy_df['position'] > 0)]
        mr_short = strategy_df[mean_rev_mask & (strategy_df['position'] < 0)]
        
        if len(mr_long) > 0:
            ax1.scatter(mr_long.index, mr_long['price'], marker='^', color='green', s=100, label='MR Long')
        if len(mr_short) > 0:
            ax1.scatter(mr_short.index, mr_short['price'], marker='v', color='red', s=100, label='MR Short')
    
    # Plot trend following positions
    if trend_mask.sum() > 0:
        tf_long = strategy_df[trend_mask & (strategy_df['position'] > 0)]
        tf_short = strategy_df[trend_mask & (strategy_df['position'] < 0)]
        
        if len(tf_long) > 0:
            ax1.scatter(tf_long.index, tf_long['price'], marker='^', color='blue', s=100, label='TF Long')
        if len(tf_short) > 0:
            ax1.scatter(tf_short.index, tf_short['price'], marker='v', color='purple', s=100, label='TF Short')
    
    ax1.set_title(title)
    ax1.set_ylabel('Price')
    ax1.legend()
    # Increase grid line visibility
    ax1.grid(True, which='major', linestyle='-', alpha=0.9, linewidth=0.8, color='gray')
    ax1.grid(True, which='minor', linestyle='--', alpha=0.5, linewidth=0.5, color='darkgray')
    ax1.minorticks_on()
    
    # Plot Hurst exponent and thresholds
    ax2.plot(strategy_df.index, strategy_df['hurst'], color='blue', label='Hurst Exponent')
    
    if 'adaptive_mr_threshold' in strategy_df.columns:
        ax2.plot(strategy_df.index, strategy_df['adaptive_mr_threshold'], color='green', linestyle='--', 
                 label='Mean Reversion Threshold')
        ax2.plot(strategy_df.index, strategy_df['adaptive_tr_threshold'], color='red', linestyle='--',
                 label='Trend Threshold')
    else:
        # Plot fixed thresholds if adaptive thresholds are not used
        mr_threshold = 0.38  # Default value
        tr_threshold = 0.54  # Default value
        ax2.axhline(y=mr_threshold, color='green', linestyle='--', label='Mean Reversion Threshold')
        ax2.axhline(y=tr_threshold, color='red', linestyle='--', label='Trend Threshold')
    
    ax2.set_ylabel('Hurst Exponent')
    ax2.legend()
    # Increase grid line visibility
    ax2.grid(True, which='major', linestyle='-', alpha=0.9, linewidth=0.8, color='gray')
    ax2.grid(True, which='minor', linestyle='--', alpha=0.5, linewidth=0.5, color='darkgray')
    ax2.minorticks_on()
    
    # Plot cumulative returns
    ax3.plot(strategy_df.index, strategy_df['cum_market_returns'], color='gray', label='Market')
    ax3.plot(strategy_df.index, strategy_df['cum_strategy_returns'], color='blue', label='Strategy')
    
    # Calculate and plot returns by strategy type
    if mean_rev_mask.sum() > 0:
        mr_df = strategy_df.copy()
        mr_df.loc[~mean_rev_mask, 'position'] = 0
        mr_df['mr_returns'] = mr_df['position'].shift(1) * mr_df['returns']
        mr_df['mr_returns'] = mr_df['mr_returns'].fillna(0)
        mr_df['cum_mr_returns'] = (1 + mr_df['mr_returns']).cumprod()
        ax3.plot(mr_df.index, mr_df['cum_mr_returns'], color='green', label='Mean Reversion')
    
    if trend_mask.sum() > 0:
        tf_df = strategy_df.copy()
        tf_df.loc[~trend_mask, 'position'] = 0
        tf_df['tf_returns'] = tf_df['position'].shift(1) * tf_df['returns']
        tf_df['tf_returns'] = tf_df['tf_returns'].fillna(0)
        tf_df['cum_tf_returns'] = (1 + tf_df['tf_returns']).cumprod()
        ax3.plot(tf_df.index, tf_df['cum_tf_returns'], color='red', label='Trend Following')
    
    ax3.set_ylabel('Cumulative Returns')
    ax3.legend()
    # Increase grid line visibility
    ax3.grid(True, which='major', linestyle='-', alpha=0.9, linewidth=0.8, color='gray')
    ax3.grid(True, which='minor', linestyle='--', alpha=0.5, linewidth=0.5, color='darkgray')
    ax3.minorticks_on()
    
    # Format x-axis
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    
    plt.tight_layout()
    return fig