import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import logging

def calculate_atr(prices_df, period=14):
    """
    Calculate Average True Range (ATR) with improved error handling
    
    Parameters:
    prices_df (DataFrame): DataFrame containing 'High', 'Low', 'Close' columns
    period (int): Period for ATR calculation

    Returns:
    Series: ATR values aligned with the input DataFrame index
    """
    try:
        if not all(col in prices_df.columns for col in ['High', 'Low', 'Close']):
            raise ValueError("Input DataFrame must contain 'High', 'Low', 'Close' columns for ATR calculation.")

        high = prices_df['High']
        low = prices_df['Low']
        close = prices_df['Close']

        # Calculate True Range components
        tr1 = abs(high - low)
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
    
        # Combine components safely with explicit column names
        tr_components = pd.concat([tr1, tr2, tr3], axis=1, keys=['tr1', 'tr2', 'tr3'])
        tr_components = tr_components.dropna()
        
        if tr_components.empty:
            return pd.Series(index=prices_df.index)
            
        true_range = tr_components.max(axis=1)

        # Calculate ATR using Exponential Moving Average
        atr = true_range.ewm(alpha=1/period, adjust=False).mean()
        return atr
        
    except Exception as e:
        logging.error(f"Error calculating ATR: {str(e)}")
        return pd.Series(index=prices_df.index)

def hurst_rs(time_series, max_lag=20):
    """
    Calculate Hurst exponent using Rescaled Range (R/S) Analysis
    
    Parameters:
    time_series (array): Time series data
    max_lag (int): Maximum lag to consider
    
    Returns:
    float: Hurst exponent
    """
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
    Calculate Hurst exponent using Detrended Fluctuation Analysis (DFA)
    
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
    max_k (int): Maximum aggregation level
    
    Returns:
    float: Hurst exponent
    """
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

def hurst_higuchi(time_series, k_max=20):
    """
    Calculate Hurst exponent using Higuchi's fractal dimension method
    
    Parameters:
    time_series (array): Time series data
    k_max (int): Maximum lag
    
    Returns:
    float: Hurst exponent (related to fractal dimension)
    """
    n = len(time_series)
    L = []
    
    for k in range(1, k_max+1):
        Lk = 0
        for m in range(k):
            if m + k >= n: 
                continue
                
            # Calculate normalized length
            Lmk = np.sum(np.abs(time_series[m::k][1:] - time_series[m::k][:-1]))
            Lmk = Lmk * (n - 1) / (k * ((n - m - 1) // k))
            Lk += Lmk
            
        L.append(Lk/k)
    
    if len(L) < 2:
        return np.nan
        
    x = np.log(np.arange(1, k_max+1)[:len(L)])
    y = np.log(L)
    hurst = np.polyfit(x, y, 1)[0]
    return 2 - hurst  # Convert fractal dimension to Hurst exponent
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

def interpret_hurst(h):
    """
    Interpret the Hurst exponent value
    
    Parameters:
    h (float): Hurst exponent
    
    Returns:
    str: Interpretation of the Hurst exponent
    """
    if h < 0.4:
        return "Strong mean reversion (anti-persistent)"
    elif h < 0.45:
        return "Mean reversion (anti-persistent)"
    elif h < 0.55:
        return "Random walk (no memory)"
    elif h < 0.7:
        return "Trending (persistent)"
    else:
        return "Strong trending (persistent)"

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
        hurst_func = lambda x: hurst_aggvar(x, max_lag)
    elif method == 'higuchi':
        hurst_func = lambda x: hurst_higuchi(x, max_lag)
    else:
        raise ValueError("Method must be one of 'rs', 'dfa', 'aggvar', 'higuchi'")
    
    # Calculate rolling Hurst exponent
    hurst_values = []
    indices = []
    
    for i in range(window_size, len(time_series), step):
        window = time_series[i-window_size:i]
        h = hurst_func(window)
        hurst_values.append(h)
        indices.append(i)
    
    return indices, hurst_values

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

def hurst_trading_strategy(prices_input,
                          window_size=100, method='rs', max_lag=20,
                          mean_reversion_threshold=0.45, trend_threshold=0.55,
                          short_ma_period=20, long_ma_period=50,
                          atr_period=14,
                          stop_loss_atr_multiplier=2.0,
                          take_profit_atr_multiplier=3.0):
    """
    Implement a trading strategy based on the Hurst exponent
    
    Parameters:
    prices_input (DataFrame): DataFrame with 'High', 'Low', 'Close' columns
    window_size (int): Size of the rolling window for Hurst calculation
    method (str): Method to use ('rs', 'dfa', 'aggvar', 'higuchi')
    max_lag (int): Maximum lag to consider
    mean_reversion_threshold (float): Hurst threshold below which mean reversion is considered
    trend_threshold (float): Hurst threshold above which trending is considered
    short_ma_period (int): Short-term moving average period for trend filter
    long_ma_period (int): Long-term moving average period for trend filter
    atr_period (int): Period for ATR calculation
    stop_loss_atr_multiplier (float): ATR multiplier for stop loss
    take_profit_atr_multiplier (float): ATR multiplier for take profit

    Returns:
    DataFrame: DataFrame with strategy results
    """
    # --- 1. Validate inputs ---
    _validate_inputs(prices_input)
    
    # --- 2. Prepare data ---
    strategy_df = _prepare_strategy_dataframe(prices_input, short_ma_period, long_ma_period, atr_period)
    
    # --- 3. Calculate Hurst exponents ---
    hurst_values = _calculate_hurst_values(prices_input['Close'], window_size, method, max_lag)
    _add_hurst_to_dataframe(strategy_df, hurst_values, window_size)
    
    # --- 4. Generate signals and manage positions ---
    _process_signals_and_positions(
        strategy_df, 
        window_size, 
        long_ma_period, 
        atr_period, 
        mean_reversion_threshold, 
        trend_threshold,
        stop_loss_atr_multiplier,
        take_profit_atr_multiplier
    )
    
    # --- 5. Calculate returns ---
    _calculate_strategy_returns(strategy_df)
    
    return strategy_df

def _validate_inputs(prices_input):
    """Validate input data for the strategy"""
    if not isinstance(prices_input, pd.DataFrame) or not all(c in prices_input.columns for c in ['High', 'Low', 'Close']):
        raise ValueError("Input 'prices_input' must be a DataFrame with 'High', 'Low', 'Close' columns.")

def _prepare_strategy_dataframe(prices_input, short_ma_period, long_ma_period, atr_period):
    """Prepare the strategy DataFrame with necessary columns and indicators"""
    returns = prices_input['Close'].pct_change().dropna()
    
    strategy_df = pd.DataFrame(index=returns.index)
    strategy_df['returns'] = returns
    strategy_df['price'] = prices_input['Close'].loc[returns.index]
    strategy_df['hurst'] = np.nan
    strategy_df['raw_signal'] = 0
    strategy_df['position'] = 0
    strategy_df['entry_price'] = np.nan
    strategy_df['stop_loss'] = np.nan
    strategy_df['take_profit'] = np.nan
    strategy_df['strategy_returns'] = 0.0
    
    # Add indicators
    strategy_df['short_ma'] = prices_input['Close'].rolling(window=short_ma_period).mean()
    strategy_df['long_ma'] = prices_input['Close'].rolling(window=long_ma_period).mean()
    strategy_df['atr'] = calculate_atr(prices_input, period=atr_period)
    
    return strategy_df

def _get_hurst_function(method, max_lag):
    """Get the appropriate Hurst calculation function based on method"""
    if method == 'rs':
        return lambda x: hurst_rs(x, max_lag)
    elif method == 'dfa':
        return lambda x: hurst_dfa(x)
    elif method == 'aggvar':
        return lambda x: hurst_aggvar(x, max_lag)
    elif method == 'higuchi':
        return lambda x: hurst_higuchi(x, max_lag)
    else:
        raise ValueError("Method must be one of 'rs', 'dfa', 'aggvar', 'higuchi'")

def _calculate_hurst_values(price_series, window_size, method, max_lag):
    """Calculate Hurst exponent values for the price series"""
    hurst_func = _get_hurst_function(method, max_lag)
    
    hurst_values = []
    for i in range(window_size, len(price_series)):
        price_window = price_series.iloc[i-window_size:i].values
        h = hurst_func(price_window)
        hurst_values.append(h)
        
    return hurst_values

def _add_hurst_to_dataframe(strategy_df, hurst_values, window_size):
    """Add calculated Hurst values to the strategy DataFrame"""
    if len(hurst_values) > 0:
        hurst_series_index = strategy_df.index[window_size - 1 : window_size - 1 + len(hurst_values)]
        if len(hurst_series_index) == len(hurst_values):
            hurst_series = pd.Series(hurst_values, index=hurst_series_index)
            strategy_df['hurst'] = hurst_series
        else:
            print(f"Warning: Hurst series length mismatch. Index length: {len(hurst_series_index)}, Values length: {len(hurst_values)}")

def _process_signals_and_positions(strategy_df, window_size, long_ma_period, atr_period, 
                                  mean_reversion_threshold, trend_threshold,
                                  stop_loss_atr_multiplier, take_profit_atr_multiplier):
    """Process signals and manage positions based on Hurst values and indicators"""
    start_index = max(window_size, long_ma_period, atr_period)
    current_position_direction = 0
    
    for i in range(start_index, len(strategy_df)):
        current_index = strategy_df.index[i]
        
        # Get current values
        h = strategy_df.loc[current_index, 'hurst']
        current_price = strategy_df.loc[current_index, 'price']
        short_ma = strategy_df.loc[current_index, 'short_ma']
        long_ma = strategy_df.loc[current_index, 'long_ma']
        current_atr = strategy_df.loc[current_index, 'atr']

        # Skip if any indicator is missing
        if any(pd.isna(x) for x in [h, current_price, short_ma, long_ma, current_atr]):
            continue
            
        # Carry forward position state from previous day
        if i > start_index:
            prev_index = strategy_df.index[i]
            strategy_df.loc[current_index, 'position'] = strategy_df.loc[prev_index, 'position']
            strategy_df.loc[current_index, 'entry_price'] = strategy_df.loc[prev_index, 'entry_price']
            strategy_df.loc[current_index, 'stop_loss'] = strategy_df.loc[prev_index, 'stop_loss']
            strategy_df.loc[current_index, 'take_profit'] = strategy_df.loc[prev_index, 'take_profit']
            current_position_direction = np.sign(strategy_df.loc[current_index, 'position'])
        else:
            current_position_direction = 0

        # Check for stop loss / take profit hit
        if current_position_direction != 0:
            exit_trade = _check_exit_conditions(
                strategy_df, 
                current_index, 
                current_price, 
                current_position_direction
            )
            
            if exit_trade:
                strategy_df.loc[current_index, 'position'] = 0
                strategy_df.loc[current_index, 'entry_price'] = np.nan
                strategy_df.loc[current_index, 'stop_loss'] = np.nan
                strategy_df.loc[current_index, 'take_profit'] = np.nan
                current_position_direction = 0

        # Generate new signal if not in a position
        raw_signal_today = _generate_signal(
            h, 
            current_price, 
            short_ma, 
            long_ma, 
            current_position_direction,
            mean_reversion_threshold,
            trend_threshold
        )
        strategy_df.loc[current_index, 'raw_signal'] = raw_signal_today

        # Set position for next day
        if i + 1 < len(strategy_df):
            next_day_index = strategy_df.index[i + 1]
            _set_next_day_position(
                strategy_df, 
                current_index, 
                next_day_index, 
                raw_signal_today, 
                current_position_direction,
                h, 
                current_price, 
                current_atr,
                stop_loss_atr_multiplier,
                take_profit_atr_multiplier
            )

def _check_exit_conditions(strategy_df, current_index, current_price, current_position_direction):
    """Check if stop loss or take profit conditions are met"""
    entry_price = strategy_df.loc[current_index, 'entry_price']
    stop_loss_level = strategy_df.loc[current_index, 'stop_loss']
    take_profit_level = strategy_df.loc[current_index, 'take_profit']
    
    if current_position_direction > 0:  # Long position
        if current_price <= stop_loss_level or current_price >= take_profit_level:
            return True
    elif current_position_direction < 0:  # Short position
        if current_price >= stop_loss_level or current_price <= take_profit_level:
            return True
            
    return False

def _generate_signal(h, current_price, short_ma, long_ma, current_position_direction,
                    mean_reversion_threshold, trend_threshold):
    """Generate trading signal based on Hurst exponent and indicators"""
    if current_position_direction != 0:
        return 0
        
    if np.isnan(h) or np.isnan(short_ma) or np.isnan(long_ma):
        return 0
        
    if h < mean_reversion_threshold:
        if current_price > short_ma:
            return -1
        elif current_price < short_ma:
            return 1
    elif h > trend_threshold:
        if short_ma > long_ma * 1.001:
            return 1
        elif short_ma < long_ma * 0.999:
            return -1
            
    return 0

def _set_next_day_position(strategy_df, current_index, next_day_index, raw_signal_today, 
                          current_position_direction, h, current_price, current_atr,
                          stop_loss_atr_multiplier, take_profit_atr_multiplier):
    """Set position, entry price, stop loss and take profit for the next day"""
    # Default to carrying over the current state
    position_next_day = strategy_df.loc[current_index, 'position']
    entry_price_next_day = strategy_df.loc[current_index, 'entry_price']
    sl_next_day = strategy_df.loc[current_index, 'stop_loss']
    tp_next_day = strategy_df.loc[current_index, 'take_profit']

    # Check if we have a new signal and are currently flat
    if raw_signal_today != 0 and current_position_direction == 0 and not np.isnan(h) and not np.isnan(current_atr):
        # Calculate confidence scaling based on Hurst
        hurst_confidence = min(abs(h - 0.5) / 0.2, 1.0)
        position_next_day = raw_signal_today * hurst_confidence
        entry_price_next_day = current_price

        # Set stop loss and take profit levels
        if position_next_day > 0:  # Long position
            sl_next_day = current_price - (current_atr * stop_loss_atr_multiplier)
            tp_next_day = current_price + (current_atr * take_profit_atr_multiplier)
        elif position_next_day < 0:  # Short position
            sl_next_day = current_price + (current_atr * stop_loss_atr_multiplier)
            tp_next_day = current_price - (current_atr * take_profit_atr_multiplier)
        else:  # No position
            entry_price_next_day = np.nan
            sl_next_day = np.nan
            tp_next_day = np.nan

    # Update strategy DataFrame for next day
    strategy_df.loc[next_day_index, 'position'] = position_next_day
    strategy_df.loc[next_day_index, 'entry_price'] = entry_price_next_day
    strategy_df.loc[next_day_index, 'stop_loss'] = sl_next_day
    strategy_df.loc[next_day_index, 'take_profit'] = tp_next_day

def _calculate_strategy_returns(strategy_df):
    """Calculate strategy returns and cumulative performance metrics"""
    # Shift positions by 1 day because today's position determines tomorrow's return
    strategy_df['strategy_returns'] = strategy_df['position'].shift(1) * strategy_df['returns']
    strategy_df['strategy_returns'].fillna(0.0, inplace=True)
    
    # Calculate cumulative returns
    strategy_df['cum_strategy_returns'] = (1 + strategy_df['strategy_returns']).cumprod() - 1
    strategy_df['cum_returns'] = (1 + strategy_df['returns']).cumprod() - 1

def evaluate_strategy(strategy_df):
    """
    Evaluate strategy performance metrics
    
    Parameters:
    strategy_df (DataFrame): DataFrame containing strategy results
    
    Returns:
    dict: Dictionary of performance metrics
    """
    # Check if DataFrame is empty or missing required columns
    if strategy_df.empty or 'strategy_returns' not in strategy_df.columns:
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
    
    # Calculate cumulative returns
    strategy_df['cum_strategy_returns'] = (1 + strategy_df['strategy_returns']).cumprod() - 1
    
    # Check if cum_strategy_returns is empty
    if len(strategy_df['cum_strategy_returns'].dropna()) == 0:
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
    
    # Calculate total return
    total_return = strategy_df['cum_strategy_returns'].iloc[-1] if len(strategy_df['cum_strategy_returns']) > 0 else 0.0
    
    # Calculate annualized return
    days = len(strategy_df)
    annualized_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0.0
    
    # Calculate maximum drawdown
    cum_returns = (1 + strategy_df['strategy_returns']).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max) - 1
    max_drawdown = drawdown.min() if len(drawdown) > 0 else 0.0
    
    # Calculate Sharpe ratio
    sharpe_ratio = (strategy_df['strategy_returns'].mean() / strategy_df['strategy_returns'].std() * np.sqrt(252)) if strategy_df['strategy_returns'].std() > 0 else 0.0
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio
    }


# --- evaluate_strategy function remains the same ---
def evaluate_strategy(strategy_df):
    """
    Evaluate the performance of a trading strategy
    
    Parameters:
    strategy_df (DataFrame): DataFrame with strategy results
    
    Returns:
    dict: Dictionary with performance metrics
    """
    # Calculate performance metrics
    total_return = strategy_df['cum_strategy_returns'].iloc[-1]
    buy_hold_return = strategy_df['cum_returns'].iloc[-1]
    
    # Calculate annualized returns (assuming 252 trading days per year)
    n_days = len(strategy_df)
    annual_return = (1 + total_return) ** (252 / n_days) - 1
    annual_buy_hold = (1 + buy_hold_return) ** (252 / n_days) - 1
    
    # Calculate volatility
    daily_std = strategy_df['strategy_returns'].std()
    annual_std = daily_std * np.sqrt(252)
    
    # Calculate Sharpe ratio (assuming risk-free rate of 0)
    sharpe = annual_return / annual_std if annual_std != 0 else 0
    
    # Calculate maximum drawdown
    cum_returns = strategy_df['cum_strategy_returns']
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max - 1)
    max_drawdown = drawdown.min()
    
    # Calculate win rate
    wins = (strategy_df['strategy_returns'] > 0).sum()
    losses = (strategy_df['strategy_returns'] < 0).sum()
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
    
    return {
        'Total Return': total_return,
        'Buy & Hold Return': buy_hold_return,
        'Outperformance': total_return - buy_hold_return,
        'Annual Return': annual_return,
        'Annual Buy & Hold': annual_buy_hold,
        'Volatility': annual_std,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_drawdown, # Use corrected max_drawdown
        'Win Rate': win_rate # Use corrected win_rate
    }


# --- Keep the parameter testing block as is ---
if __name__ == "__main__":
    # Change logging level from DEBUG to INFO to reduce output
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Script started.")

    import yfinance as yf
    import itertools # Import itertools for parameter combinations

    # Download data
    ticker = 'AAPL'
    logging.info(f"Downloading data for {ticker}...")
    try:
        data = yf.download(ticker, start='2015-01-01', end='2023-01-01')
        if not all(c in data.columns for c in ['High', 'Low', 'Close']):
            logging.error("Downloaded data missing required columns (High, Low, Close).")
            raise ValueError("Downloaded data must include 'High', 'Low', 'Close' columns.")
        logging.info("Data download successful.")
    except Exception as e:
        logging.error(f"Failed to download data: {e}")
        exit() # Exit if data download fails

    # --- Define Parameter Ranges ---
    methods = ['rs', 'dfa']
    window_sizes = [60, 90, 120]
    mean_reversion_thresholds = [0.40, 0.45, 0.48]
    trend_thresholds = [0.52, 0.55, 0.60]
    stop_multipliers = [1.5, 2.0, 2.5]
    profit_multipliers = [2.5, 3.0, 4.0]

    results = {}
    best_sharpe = -np.inf
    best_params = None

    # --- Create combinations ---
    param_combinations = [
        params for params in
        itertools.product(methods, window_sizes, mean_reversion_thresholds, trend_thresholds, stop_multipliers, profit_multipliers)
        if params[3] > params[2] # Ensure tr_threshold > mr_threshold
    ]

    logging.info(f"Starting parameter testing for {len(param_combinations)} combinations...")

    for params in param_combinations:
        method, window_size, mr_threshold, tr_threshold, stop_mult, profit_mult = params
        current_params_str = f"method={method}, win={window_size}, mr={mr_threshold}, tr={tr_threshold}, sl={stop_mult}, tp={profit_mult}"
        logging.info(f"Testing parameters: {current_params_str}")

        try:
            # Log before calling the strategy function
            logging.debug(f"Calling hurst_trading_strategy with params: {current_params_str}")
            strategy_df = hurst_trading_strategy(
                data, # Pass the DataFrame with HLC
                window_size=window_size,
                method=method,
                mean_reversion_threshold=mr_threshold,
                trend_threshold=tr_threshold,
                stop_loss_atr_multiplier=stop_mult,
                take_profit_atr_multiplier=profit_mult
            )
            logging.debug(f"hurst_trading_strategy call successful for params: {current_params_str}")

            # Log before calling the evaluation function
            logging.debug(f"Calling evaluate_strategy for params: {current_params_str}")
            performance = evaluate_strategy(strategy_df)
            logging.debug(f"evaluate_strategy call successful for params: {current_params_str}")

            results[params] = performance

            logging.info(f"  Result: Sharpe={performance['Sharpe Ratio']:.4f}, Return={performance['Total Return']:.2%}, Drawdown={performance['Max Drawdown']:.2%}")

            # Track best Sharpe ratio
            current_sharpe = performance['Sharpe Ratio']
            if np.isfinite(current_sharpe) and current_sharpe > best_sharpe:
                 best_sharpe = current_sharpe
                 best_params = params
                 logging.info(f"  New best Sharpe found: {best_sharpe:.4f} with params: {current_params_str}")

        except NameError as ne:
             # Specifically catch NameError
             logging.error(f"NameError encountered with parameters {current_params_str}: {ne}", exc_info=True)
             # Optionally re-raise or break if needed: raise ne
        except Exception as e:
            # Catch other potential errors during strategy execution or evaluation
            logging.error(f"Error processing parameters {current_params_str}: {e}", exc_info=True)
        logging.debug("-" * 40) # Use debug level for separator

    # --- Output Best Result ---
    logging.info("Parameter testing finished.")
    if best_params:
        logging.info("--- Best Performing Combination (with Risk Management) ---")
        best_params_str = f"method={best_params[0]}, window={best_params[1]}, mr={best_params[2]}, tr={best_params[3]}, sl_atr={best_params[4]}, tp_atr={best_params[5]}"
        logging.info(f"Best Parameters: {best_params_str}")
        best_performance = results[best_params]
        logging.info(f"Sharpe Ratio: {best_performance['Sharpe Ratio']:.4f}")
        logging.info(f"Total Return: {best_performance['Total Return']:.2%}")
        logging.info(f"Max Drawdown: {best_performance['Max Drawdown']:.2%}")
        logging.info(f"Win Rate: {best_performance['Win Rate']:.2%}")

        # Optional: Plotting (keep commented out unless needed)
        # logging.info("Generating plot for best strategy...")
        # try:
        #     best_strategy_df = hurst_trading_strategy(data, window_size=best_params[1], method=best_params[0], mean_reversion_threshold=best_params[2], trend_threshold=best_params[3], stop_loss_atr_multiplier=best_params[4], take_profit_atr_multiplier=best_params[5])
        #     plt.figure(figsize=(12, 6))
        #     best_strategy_df['cum_returns'].plot(label='Buy & Hold')
        #     best_strategy_df['cum_strategy_returns'].plot(label='Optimized Hurst Strategy')
        #     plt.title(f'{ticker} - Optimized Hurst Strategy (Best Sharpe) vs Buy & Hold')
        #     plt.legend()
        #     plt.grid(True)
        #     # plt.show() # Consider saving instead of showing in automated runs
        #     # plt.savefig('best_strategy_performance.png')
        #     logging.info("Plot generated.")
        # except Exception as plot_err:
        #     logging.error(f"Error generating plot: {plot_err}")

    else:
        logging.warning("No valid results found or best Sharpe was not positive.")

    logging.info("Script finished.")