import numpy as np
from scipy import stats
import pandas as pd

def hurst_rs(time_series, max_lag=20):
    """
    Calculate Hurst exponent using R/S analysis
    
    Parameters:
    time_series (array): Time series data
    max_lag (int): Maximum lag to consider
    
    Returns:
    float: Hurst exponent
    """
    # Implementation of R/S analysis
    # Convert to numpy array if it's not already
    time_series = np.array(time_series)
    
    # Calculate the array of the variances of the lagged differences
    lags = range(2, max_lag)
    
    # Calculate the rescaled range for each lag
    rs_values = []
    for lag in lags:
        # Split time series into chunks of size 'lag'
        chunks = len(time_series) // lag
        if chunks < 1:
            break
            
        # Ignore the remainder
        values = time_series[:chunks * lag].reshape((chunks, lag))
        
        # Calculate the mean of each chunk
        means = np.mean(values, axis=1)
        
        # Calculate the standard deviation of each chunk
        stds = np.std(values, axis=1)
        stds = np.where(stds == 0, 1, stds)  # Avoid division by zero
        
        # Calculate the range of cumulative sum within each chunk
        deviations = values - means.reshape(-1, 1)
        cumsum = np.cumsum(deviations, axis=1)
        ranges = np.max(cumsum, axis=1) - np.min(cumsum, axis=1)
        
        # Calculate R/S value for this lag
        rs = np.mean(ranges / stds)
        rs_values.append(rs)
    
    if len(rs_values) < 2:
        return np.nan
    
    # Fit a line to log-log plot and get the slope
    hurst = np.polyfit(np.log(lags[:len(rs_values)]), np.log(rs_values), 1)[0]
    
    return hurst

def hurst_dfa(time_series, min_boxes=4, max_boxes=None):
    """
    Calculate Hurst exponent using Detrended Fluctuation Analysis
    
    Parameters:
    time_series (array): Time series data
    min_boxes (int): Minimum number of boxes
    max_boxes (int): Maximum number of boxes (default: len(time_series) // 4)
    
    Returns:
    float: Hurst exponent
    """
    # Convert to numpy array if it's not already
    time_series = np.array(time_series)
    
    # Calculate the cumulative sum of deviations from the mean
    y = np.cumsum(time_series - np.mean(time_series))
    
    # Set the maximum number of boxes if not provided
    if max_boxes is None:
        max_boxes = len(time_series) // 4
    
    # Create a range of box sizes
    box_sizes = np.unique(np.logspace(np.log10(min_boxes), np.log10(max_boxes), 20).astype(int))
    
    # Calculate the fluctuation for each box size
    fluctuations = []
    for box_size in box_sizes:
        # Skip if box_size is too large
        if box_size >= len(time_series) // 2:
            continue
            
        # Number of boxes
        n_boxes = len(time_series) // box_size
        
        # Truncate the series to fit the boxes
        y_trunc = y[:n_boxes * box_size]
        
        # Reshape the series into boxes
        y_reshaped = y_trunc.reshape((n_boxes, box_size))
        
        # Calculate local trends for each box
        x = np.arange(box_size)
        trends = np.array([np.polyval(np.polyfit(x, y_box, 1), x) for y_box in y_reshaped])
        
        # Calculate the fluctuation as the root mean square deviation
        fluctuation = np.sqrt(np.mean((y_reshaped - trends) ** 2))
        fluctuations.append(fluctuation)
    
    if len(fluctuations) < 2:
        return np.nan
    
    # Fit a line to log-log plot and get the slope
    hurst = np.polyfit(np.log(box_sizes[:len(fluctuations)]), np.log(fluctuations), 1)[0]
    
    return hurst

def hurst_aggvar(time_series, max_k=20):
    """
    Calculate Hurst exponent using Aggregated Variance method
    
    Parameters:
    time_series (array): Time series data
    max_k (int): Maximum aggregation level to consider
    
    Returns:
    float: Hurst exponent
    """
    # Implementation of Aggregated Variance method
    # Convert to numpy array if it's not already
    time_series = np.array(time_series)
    
    # Calculate the variance of the original series
    var_t = np.var(time_series)
    
    # Calculate the variance at different aggregation levels
    ks = []
    variances = []
    
    for k in range(1, max_k + 1):
        # Skip if k is too large
        if len(time_series) // k < 2:
            break
            
        # Truncate the series to be divisible by k
        trunc = len(time_series) - (len(time_series) % k)
        time_agg = time_series[:trunc].reshape(-1, k).mean(axis=1)
        
        # Calculate the variance of the aggregated series
        var_k = np.var(time_agg)
        
        # Normalize by k
        var_k_norm = var_k * (k ** 2)
        
        ks.append(k)
        variances.append(var_k)
    
    if len(variances) < 2:
        return np.nan
        
    ks = ks[:len(variances)]
    hurst = 1 - np.polyfit(np.log(ks), np.log(variances), 1)[0]/2
    return hurst

def hurst_higuchi(time_series, max_k=20):
    """
    Calculate Hurst exponent using Higuchi's method
    
    Parameters:
    time_series (array): Time series data
    max_k (int): Maximum lag to consider
    
    Returns:
    float: Hurst exponent
    """
    # Implementation of Higuchi's method
    # This is a placeholder - you would need to implement the actual algorithm
    return 0.5  # Default value

def rolling_hurst(time_series, window_size=100, step=10, method='rs', max_lag=20):
    """
    Calculate rolling Hurst exponent
    
    Parameters:
    time_series (array): Time series data
    window_size (int): Size of the rolling window
    step (int): Step size for the rolling window
    method (str): Method to use ('rs', 'dfa', 'aggvar', 'higuchi')
    max_lag (int): Maximum lag to consider
    
    Returns:
    tuple: (dates, hurst_values)
    """
    # Select the method
    if method == 'rs':
        hurst_func = lambda x: hurst_rs(x, max_lag)
    elif method == 'dfa':
        hurst_func = lambda x: hurst_dfa(x)
    elif method == 'aggvar':
        hurst_func = lambda x: hurst_aggvar(x, max_lag)  # Pass max_lag as max_k
    elif method == 'higuchi':
        hurst_func = lambda x: hurst_higuchi(x, max_lag)
    else:
        raise ValueError("Method must be one of 'rs', 'dfa', 'aggvar', 'higuchi'")
    
    # Calculate rolling Hurst exponent
    indices = []
    hurst_values = []
    
    for i in range(window_size, len(time_series), step):
        window = time_series[i-window_size:i]
        h = hurst_func(window)
        indices.append(i)
        hurst_values.append(h)
    
    return indices, hurst_values