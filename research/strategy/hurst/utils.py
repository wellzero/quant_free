import numpy as np
import pandas as pd
from .core import hurst_rs, hurst_dfa, hurst_aggvar, hurst_higuchi

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

def _get_hurst_function(method, max_lag):
    """Get the appropriate Hurst calculation function based on method"""
    if method == 'rs':
        return lambda x: hurst_rs(x, max_lag)
    elif method == 'dfa':
        return lambda x: hurst_dfa(x)
    elif method == 'aggvar':
        # Fix: Remove max_lag parameter for aggvar method
        return lambda x: hurst_aggvar(x)
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

def _check_correlation_filter(strategy_df, i, lookback=30):
    """Check if strategy returns are correlated with market returns"""
    if 'market_returns' not in strategy_df.columns or i < lookback:
        return True
        
    # Get recent strategy and market returns
    recent_strategy = strategy_df['strategy_returns'].iloc[i-lookback:i]
    recent_market = strategy_df['market_returns'].iloc[i-lookback:i]
    
    # Calculate correlation
    if len(recent_strategy) > 5 and len(recent_market) > 5:
        correlation = recent_strategy.corr(recent_market)
        
        # Skip trades when highly correlated with market
        if not pd.isna(correlation) and abs(correlation) > 0.7:
            return False
    
    return True