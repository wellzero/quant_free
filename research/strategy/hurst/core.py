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

def hurst_dfa(time_series, min_boxes=10, max_boxes=None):
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
    n = len(time_series)
    
    # Input validation
    if n < 2 * min_boxes:
        return np.nan
    
    # Calculate the cumulative sum of deviations from the mean
    y = np.cumsum(time_series - np.mean(time_series))
    
    # Set the maximum number of boxes if not provided
    if max_boxes is None:
        max_boxes = n // 4
    
    # Ensure max_boxes is not too large
    max_boxes = min(max_boxes, n // 4)
    
    # Create a range of box sizes - use more points for better fit
    # and ensure they're evenly distributed on log scale
    box_sizes = np.unique(
        np.logspace(
            np.log10(min_boxes), 
            np.log10(max_boxes),
            30  # More points for better fit
        ).astype(int)
    )
    
    # Filter out box sizes that are too large
    box_sizes = box_sizes[box_sizes < n // 2]
    
    # Pre-allocate array for fluctuations
    fluctuations = np.zeros(len(box_sizes))
    
    # Calculate the fluctuation for each box size
    for i, box_size in enumerate(box_sizes):
        # Number of boxes
        n_boxes = n // box_size
        
        if n_boxes < 1:
            continue
            
        # Truncate the series to fit the boxes
        y_trunc = y[:n_boxes * box_size]
        
        # Reshape the series into boxes
        y_reshaped = y_trunc.reshape((n_boxes, box_size))
        
        # Create x values for polynomial fitting (only once)
        x = np.arange(box_size)
        
        # Vectorized calculation of local trends
        # This is more efficient than a loop and avoids broadcasting issues
        trends = np.zeros_like(y_reshaped)
        
        for j in range(n_boxes):
            # Calculate trend for each box individually to avoid broadcasting issues
            coeffs = np.polyfit(x, y_reshaped[j], 1)
            trends[j] = coeffs[0] * x + coeffs[1]
        
        # Calculate the fluctuation as the root mean square deviation
        # Use np.mean for better numerical stability
        fluctuation = np.sqrt(np.mean(np.square(y_reshaped - trends)))
        fluctuations[i] = fluctuation
    
    # Remove any zero values that might have been skipped
    valid_indices = fluctuations > 0
    box_sizes = box_sizes[valid_indices]
    fluctuations = fluctuations[valid_indices]
    
    if len(fluctuations) < 2:
        return np.nan
    
    # Fit a line to log-log plot and get the slope
    # Use weighted fit to reduce the impact of outliers
    log_box_sizes = np.log(box_sizes)
    log_fluctuations = np.log(fluctuations)
    
    # Use linear regression with weights proportional to box size
    # This gives more importance to larger scales which are more reliable
    weights = np.sqrt(box_sizes)
    hurst = np.polyfit(log_box_sizes, log_fluctuations, 1, w=weights)[0]
    
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
    # Fit a line to log-log plot of variances vs aggregation levels
    slope = np.polyfit(np.log(ks), np.log(variances), 1)[0]
    
    # Corrected Hurst exponent calculation
    # H = (slope + 2) / 2
    hurst = (slope + 2.0) / 2.0
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
    # Convert to numpy array if it's not already
    time_series = np.array(time_series)
    n = len(time_series)
    
    # Calculate L(k) for different k values
    k_values = []
    L_values = []
    
    for k in range(1, max_k + 1):
        if k >= n:
            continue
            
        L = 0
        valid_subseries = 0
        for m in range(k):
            subseries = time_series[m::k]
            if len(subseries) < 2:  # Skip if insufficient points
                continue
                
            # Calculate normalized length
            diffs = np.diff(subseries)
            if len(diffs) > 0:  # Only process if differences exist
                L += (np.sum(np.abs(diffs)) * (n - 1)) / (k * len(subseries) * k)
                valid_subseries += 1
                
        if valid_subseries == 0:  # Skip if no valid subseries
            continue
            
        L /= k
        k_values.append(k)
        L_values.append(L)
    
    if len(k_values) < 2:
        return np.nan
        
    # Fit a line to log-log plot and get the slope
    hurst = np.polyfit(np.log(k_values), np.log(L_values), 1)[0]
    
    return hurst

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