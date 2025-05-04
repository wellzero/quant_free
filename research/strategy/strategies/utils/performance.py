import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def calculate_returns_metrics(returns):
    """
    Calculate common performance metrics for a returns series
    
    Parameters:
    returns (Series): Series of returns (not cumulative)
    
    Returns:
    dict: Dictionary of performance metrics
    """
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    
    # Annualization factor (assuming daily returns)
    ann_factor = 252
    
    # Calculate metrics
    total_return = (1 + returns).prod() - 1
    ann_return = (1 + returns).prod() ** (ann_factor / len(returns)) - 1
    ann_volatility = returns.std() * np.sqrt(ann_factor)
    sharpe_ratio = ann_return / ann_volatility if ann_volatility > 0 else 0
    
    # Drawdown analysis
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max) - 1
    max_drawdown = drawdown.min()
    
    # Calculate Calmar ratio
    calmar_ratio = abs(ann_return / max_drawdown) if max_drawdown < 0 else np.nan
    
    # Calculate Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(ann_factor)
    sortino_ratio = ann_return / downside_deviation if len(downside_returns) > 0 and downside_deviation > 0 else np.nan
    
    # Win rate
    win_rate = len(returns[returns > 0]) / len(returns)
    
    # Profit factor
    profit_factor = abs(returns[returns > 0].sum() / returns[returns < 0].sum()) if len(returns[returns < 0]) > 0 and returns[returns < 0].sum() != 0 else np.nan
    
    return {
        'total_return': total_return,
        'ann_return': ann_return,
        'ann_volatility': ann_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'sortino_ratio': sortino_ratio,
        'win_rate': win_rate,
        'profit_factor': profit_factor
    }

def plot_equity_curve(returns, benchmark_returns=None, title="Equity Curve"):
    """
    Plot equity curve and drawdowns
    
    Parameters:
    returns (Series): Series of returns (not cumulative)
    benchmark_returns (Series, optional): Series of benchmark returns for comparison
    title (str): Title for the plot
    
    Returns:
    matplotlib.figure.Figure: The figure object
    """
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    
    # Calculate cumulative returns
    cum_returns = (1 + returns).cumprod()
    
    # Calculate drawdowns
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max) - 1
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot equity curve
    ax1.plot(cum_returns.index, cum_returns, label='Strategy')
    
    # Add benchmark if provided
    if benchmark_returns is not None:
        if not isinstance(benchmark_returns, pd.Series):
            benchmark_returns = pd.Series(benchmark_returns)
        cum_benchmark = (1 + benchmark_returns).cumprod()
        ax1.plot(cum_benchmark.index, cum_benchmark, label='Benchmark', alpha=0.7)
    
    ax1.set_title(title)
    ax1.set_ylabel('Cumulative Return')
    ax1.legend()
    ax1.grid(True)
    
    # Plot drawdowns
    ax2.fill_between(drawdown.index, 0, drawdown, color='red', alpha=0.3)
    ax2.set_ylabel('Drawdown')
    ax2.set_xlabel('Date')
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

def plot_monthly_returns_heatmap(returns):
    """
    Plot monthly returns as a heatmap
    
    Parameters:
    returns (Series): Series of returns (not cumulative) with DatetimeIndex
    
    Returns:
    matplotlib.figure.Figure: The figure object
    """
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    
    # Ensure we have a DatetimeIndex
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("Returns must have a DatetimeIndex")
    
    # Resample to monthly returns
    monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    
    # Create a pivot table with years as rows and months as columns
    monthly_returns.index = monthly_returns.index.to_period('M')
    heatmap_data = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).first().unstack()
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = plt.cm.RdYlGn  # Red for negative, green for positive
    
    # Create the heatmap
    im = ax.imshow(heatmap_data, cmap=cmap)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Returns', rotation=-90, va="bottom")
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(heatmap_data.columns)))
    ax.set_yticks(np.arange(len(heatmap_data.index)))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.set_yticklabels(heatmap_data.index)
    
    # Rotate the tick labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations with the values
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            value = heatmap_data.iloc[i, j]
            if not np.isnan(value):
                text_color = 'black' if abs(value) < 0.1 else 'white'
                ax.text(j, i, f'{value:.1%}', ha="center", va="center", color=text_color)
    
    ax.set_title("Monthly Returns Heatmap")
    fig.tight_layout()
    
    return fig

def calculate_rolling_metrics(returns, window=252):
    """
    Calculate rolling performance metrics
    
    Parameters:
    returns (Series): Series of returns (not cumulative)
    window (int): Rolling window size
    
    Returns:
    DataFrame: DataFrame with rolling metrics
    """
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    
    # Annualization factor (assuming daily returns)
    ann_factor = 252
    
    # Calculate rolling metrics
    rolling_return = returns.rolling(window=window).apply(lambda x: (1 + x).prod() - 1) * (ann_factor / window)
    rolling_vol = returns.rolling(window=window).std() * np.sqrt(ann_factor)
    rolling_sharpe = rolling_return / rolling_vol
    
    # Calculate rolling max drawdown
    rolling_max_drawdown = pd.Series(index=returns.index)
    
    for i in range(window, len(returns) + 1):
        window_returns = returns.iloc[i-window:i]
        cum_returns = (1 + window_returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns / running_max) - 1
        rolling_max_drawdown.iloc[i-1] = drawdown.min()
    
    # Combine metrics
    rolling_metrics = pd.DataFrame({
        'Rolling Return (Ann.)': rolling_return,
        'Rolling Volatility (Ann.)': rolling_vol,
        'Rolling Sharpe': rolling_sharpe,
        'Rolling Max Drawdown': rolling_max_drawdown
    })
    
    return rolling_metrics