import numpy as np
import pandas as pd
from scipy import stats

def calculate_var(returns, confidence_level=0.95, method='historical'):
    """
    Calculate Value at Risk (VaR)
    
    Parameters:
    returns (Series): Series of returns (not cumulative)
    confidence_level (float): Confidence level (e.g., 0.95 for 95%)
    method (str): Method to use ('historical', 'parametric', or 'monte_carlo')
    
    Returns:
    float: Value at Risk
    """
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    
    if method == 'historical':
        # Historical VaR
        return -np.percentile(returns, 100 * (1 - confidence_level))
    
    elif method == 'parametric':
        # Parametric VaR (assuming normal distribution)
        z_score = stats.norm.ppf(1 - confidence_level)
        return -(returns.mean() + z_score * returns.std())
    
    elif method == 'monte_carlo':
        # Monte Carlo VaR
        # Fit a normal distribution to the returns
        mu, sigma = returns.mean(), returns.std()
        
        # Generate random samples
        n_samples = 10000
        simulated_returns = np.random.normal(mu, sigma, n_samples)
        
        # Calculate VaR from simulated returns
        return -np.percentile(simulated_returns, 100 * (1 - confidence_level))
    
    else:
        raise ValueError("Method must be one of 'historical', 'parametric', or 'monte_carlo'")

def calculate_cvar(returns, confidence_level=0.95):
    """
    Calculate Conditional Value at Risk (CVaR) / Expected Shortfall
    
    Parameters:
    returns (Series): Series of returns (not cumulative)
    confidence_level (float): Confidence level (e.g., 0.95 for 95%)
    
    Returns:
    float: Conditional Value at Risk
    """
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    
    # Calculate VaR
    var = calculate_var(returns, confidence_level, 'historical')
    
    # Calculate CVaR as the mean of returns beyond VaR
    return -returns[returns <= -var].mean()

def calculate_drawdowns(returns):
    """
    Calculate drawdown series and related metrics
    
    Parameters:
    returns (Series): Series of returns (not cumulative)
    
    Returns:
    dict: Dictionary with drawdown series and metrics
    """
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    
    # Calculate cumulative returns
    cum_returns = (1 + returns).cumprod()
    
    # Calculate drawdowns
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max) - 1
    
    # Calculate max drawdown
    max_drawdown = drawdown.min()
    
    # Calculate drawdown duration
    is_in_drawdown = drawdown < 0
    
    # Find the start of each drawdown period
    drawdown_start = is_in_drawdown & ~is_in_drawdown.shift(1).fillna(False)
    drawdown_start_indices = drawdown_start[drawdown_start].index
    
    # Find the end of each drawdown period
    drawdown_end = ~is_in_drawdown & is_in_drawdown.shift(1).fillna(False)
    drawdown_end_indices = drawdown_end[drawdown_end].index
    
    # If we're still in a drawdown, add the last date as an end
    if len(drawdown_start_indices) > len(drawdown_end_indices):
        drawdown_end_indices = drawdown_end_indices.append(pd.Index([returns.index[-1]]))
    
    # Calculate drawdown durations
    drawdown_durations = []
    max_duration = 0
    
    for i in range(min(len(drawdown_start_indices), len(drawdown_end_indices))):
        start = drawdown_start_indices[i]
        end = drawdown_end_indices[i]
        duration = len(returns.loc[start:end])
        drawdown_durations.append(duration)
        max_duration = max(max_duration, duration)
    
    return {
        'drawdown_series': drawdown,
        'max_drawdown': max_drawdown,
        'max_drawdown_duration': max_duration,
        'drawdown_durations': drawdown_durations,
        'avg_drawdown_duration': np.mean(drawdown_durations) if drawdown_durations else 0
    }

def calculate_risk_metrics(returns):
    """
    Calculate comprehensive risk metrics
    
    Parameters:
    returns (Series): Series of returns (not cumulative)
    
    Returns:
    dict: Dictionary of risk metrics
    """
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    
    # Annualization factor (assuming daily returns)
    ann_factor = 252
    
    # Basic metrics
    volatility = returns.std() * np.sqrt(ann_factor)
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(ann_factor) if len(downside_returns) > 0 else 0
    
    # Drawdown analysis
    drawdown_info = calculate_drawdowns(returns)
    
    # VaR and CVaR
    var_95 = calculate_var(returns, 0.95, 'historical')
    var_99 = calculate_var(returns, 0.99, 'historical')
    cvar_95 = calculate_cvar(returns, 0.95)
    
    # Skewness and kurtosis
    skewness = returns.skew()
    kurtosis = returns.kurtosis()
    
    # Calculate beta if benchmark is provided
    beta = np.nan
    
    # Combine metrics
    risk_metrics = {
        'volatility': volatility,
        'downside_deviation': downside_deviation,
        'max_drawdown': drawdown_info['max_drawdown'],
        'max_drawdown_duration': drawdown_info['max_drawdown_duration'],
        'avg_drawdown_duration': drawdown_info['avg_drawdown_duration'],
        'var_95': var_95,
        'var_99': var_99,
        'cvar_95': cvar_95,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'beta': beta
    }
    
    return risk_metrics

def calculate_position_sizing(price, volatility, risk_per_trade=0.01, account_size=100000, stop_loss_pct=None):
    """
    Calculate position size based on volatility and risk parameters
    
    Parameters:
    price (float): Current price of the asset
    volatility (float): Volatility of the asset (e.g., ATR or standard deviation)
    risk_per_trade (float): Percentage of account to risk per trade (e.g., 0.01 for 1%)
    account_size (float): Total account size
    stop_loss_pct (float, optional): Stop loss percentage, if None will use 2 * volatility
    
    Returns:
    dict: Dictionary with position sizing information
    """
    # Calculate risk amount in currency
    risk_amount = account_size * risk_per_trade
    
    # Calculate stop loss distance
    if stop_loss_pct is None:
        # Use 2 times the volatility as a default stop loss
        stop_loss_distance = 2 * volatility
    else:
        stop_loss_distance = price * stop_loss_pct
    
    # Calculate position size
    position_size = risk_amount / stop_loss_distance
    position_value = position_size * price
    
    # Calculate number of shares/contracts
    shares = position_value / price
    
    return {
        'risk_amount': risk_amount,
        'stop_loss_distance': stop_loss_distance,
        'position_size': position_size,
        'position_value': position_value,
        'shares': shares
    }

def calculate_kelly_criterion(win_rate, win_loss_ratio):
    """
    Calculate Kelly Criterion for optimal position sizing
    
    Parameters:
    win_rate (float): Probability of winning (between 0 and 1)
    win_loss_ratio (float): Ratio of average win to average loss
    
    Returns:
    float: Kelly percentage (between 0 and 1)
    """
    # Kelly formula: f* = (p * b - (1 - p)) / b
    # where p is win rate, b is win/loss ratio
    
    kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
    
    # Limit to reasonable bounds
    kelly = max(0, min(kelly, 1))
    
    return kelly