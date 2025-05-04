import numpy as np
import pandas as pd
from .utils import (
    _validate_inputs, 
    _prepare_strategy_dataframe, 
    _calculate_hurst_values, 
    _add_hurst_to_dataframe,
    _check_correlation_filter,
    calculate_atr
)

def _process_signals_and_positions(
    strategy_df, 
    window_size, 
    long_ma_period, 
    atr_period, 
    mean_reversion_threshold, 
    trend_threshold,
    stop_loss_atr_multiplier,
    take_profit_atr_multiplier
):
    """Process signals and manage positions based on Hurst exponent values"""
    # Skip initial rows where we don't have enough data
    start_idx = max(window_size, long_ma_period)
    
    # Initialize position tracking variables
    current_position = 0
    entry_price = None
    
    # Process each row in the DataFrame
    for i in range(start_idx, len(strategy_df)):
        current_idx = strategy_df.index[i]
        prev_idx = strategy_df.index[i-1]
        
        # Skip if we don't have a valid Hurst value
        if np.isnan(strategy_df.loc[current_idx, 'hurst']):
            continue
        
        # Get current price and Hurst value
        current_price = strategy_df.loc[current_idx, 'price']
        current_hurst = strategy_df.loc[current_idx, 'hurst']
        current_atr = strategy_df.loc[current_idx, 'atr']
        
        # Check if we have an existing position
        if current_position != 0:
            # Check stop loss and take profit
            if entry_price is not None:
                # For long positions
                if current_position > 0:
                    stop_price = entry_price - (stop_loss_atr_multiplier * current_atr)
                    take_profit_price = entry_price + (take_profit_atr_multiplier * current_atr)
                    
                    if current_price <= stop_price:
                        # Stop loss hit
                        strategy_df.loc[current_idx, 'position'] = 0
                        current_position = 0
                        entry_price = None
                    elif current_price >= take_profit_price:
                        # Take profit hit
                        strategy_df.loc[current_idx, 'position'] = 0
                        current_position = 0
                        entry_price = None
                    else:
                        # Maintain position
                        strategy_df.loc[current_idx, 'position'] = current_position
                
                # For short positions
                elif current_position < 0:
                    stop_price = entry_price + (stop_loss_atr_multiplier * current_atr)
                    take_profit_price = entry_price - (take_profit_atr_multiplier * current_atr)
                    
                    if current_price >= stop_price:
                        # Stop loss hit
                        strategy_df.loc[current_idx, 'position'] = 0
                        current_position = 0
                        entry_price = None
                    elif current_price <= take_profit_price:
                        # Take profit hit
                        strategy_df.loc[current_idx, 'position'] = 0
                        current_position = 0
                        entry_price = None
                    else:
                        # Maintain position
                        strategy_df.loc[current_idx, 'position'] = current_position
        
        # Check for new entry signals if not in a position
        if current_position == 0:
            # Apply correlation filter
            if not _check_correlation_filter(strategy_df, i):
                continue
                
            # Mean reversion signal (Hurst < threshold)
            if current_hurst < mean_reversion_threshold:
                # Check if price is above/below moving average for mean reversion
                if current_price > strategy_df.loc[current_idx, 'short_ma']:
                    # Price above MA -> go short for mean reversion
                    strategy_df.loc[current_idx, 'raw_signal'] = -1
                    strategy_df.loc[current_idx, 'position'] = -1
                    current_position = -1
                    entry_price = current_price
                elif current_price < strategy_df.loc[current_idx, 'short_ma']:
                    # Price below MA -> go long for mean reversion
                    strategy_df.loc[current_idx, 'raw_signal'] = 1
                    strategy_df.loc[current_idx, 'position'] = 1
                    current_position = 1
                    entry_price = current_price
            
            # Trend following signal (Hurst > threshold)
            elif current_hurst > trend_threshold:
                # Check if short MA is above/below long MA for trend following
                if strategy_df.loc[current_idx, 'short_ma'] > strategy_df.loc[current_idx, 'long_ma']:
                    # Uptrend -> go long
                    strategy_df.loc[current_idx, 'raw_signal'] = 1
                    strategy_df.loc[current_idx, 'position'] = 1
                    current_position = 1
                    entry_price = current_price
                elif strategy_df.loc[current_idx, 'short_ma'] < strategy_df.loc[current_idx, 'long_ma']:
                    # Downtrend -> go short
                    strategy_df.loc[current_idx, 'raw_signal'] = -1
                    strategy_df.loc[current_idx, 'position'] = -1
                    current_position = -1
                    entry_price = current_price
        
        # Store entry price, stop loss and take profit levels
        if entry_price is not None:
            strategy_df.loc[current_idx, 'entry_price'] = entry_price
            
            if current_position > 0:
                strategy_df.loc[current_idx, 'stop_loss'] = entry_price - (stop_loss_atr_multiplier * current_atr)
                strategy_df.loc[current_idx, 'take_profit'] = entry_price + (take_profit_atr_multiplier * current_atr)
            elif current_position < 0:
                strategy_df.loc[current_idx, 'stop_loss'] = entry_price + (stop_loss_atr_multiplier * current_atr)
                strategy_df.loc[current_idx, 'take_profit'] = entry_price - (take_profit_atr_multiplier * current_atr)

def _calculate_strategy_returns(strategy_df):
    """Calculate strategy returns and cumulative returns"""
    # Calculate strategy returns
    strategy_df['strategy_returns'] = strategy_df['position'].shift(1) * strategy_df['returns']
    
    # Fill NaN values with 0
    strategy_df['strategy_returns'] = strategy_df['strategy_returns'].fillna(0)
    
    # Calculate cumulative returns
    strategy_df['cum_strategy_returns'] = (1 + strategy_df['strategy_returns']).cumprod()
    strategy_df['cum_market_returns'] = (1 + strategy_df['returns']).cumprod()

def hurst_trading_strategy(prices_input,
                          window_size=120,
                          method='rs',
                          max_lag=20,
                          mean_reversion_threshold=0.38,
                          trend_threshold=0.54,
                          short_ma_period=12,
                          long_ma_period=50,
                          atr_period=14,
                          stop_loss_atr_multiplier=1.8,
                          take_profit_atr_multiplier=3.0,
                          save_results=True,
                          csv_path='hurst_strategy_results.csv'):
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
    save_results (bool): Whether to save results to CSV
    csv_path (str): Path to save CSV results
    
    Returns:
    DataFrame: Strategy results
    """
    # Validate inputs
    _validate_inputs(prices_input)
    
    # Prepare data
    strategy_df = _prepare_strategy_dataframe(prices_input, short_ma_period, long_ma_period, atr_period)
    
    # Calculate Hurst exponents
    hurst_values = _calculate_hurst_values(prices_input['Close'], window_size, method, max_lag)
    _add_hurst_to_dataframe(strategy_df, hurst_values, window_size)
    
    # Process signals and manage positions
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
    
    # Calculate returns
    _calculate_strategy_returns(strategy_df)
    
    # Save results if requested
    if save_results and 'save_strategy_results_to_csv' in globals():
        save_strategy_results_to_csv(strategy_df, csv_path)
    
    return strategy_df

def mean_reversion_strategy(prices_input,
                           window_size=120,
                           method='rs',
                           max_lag=20,
                           hurst_threshold=0.38,
                           short_ma_period=12,
                           long_ma_period=50,
                           atr_period=14,
                           stop_loss_atr_multiplier=1.8,
                           take_profit_atr_multiplier=3.0,
                           save_results=True,
                           csv_path='mean_reversion_results.csv'):
    """
    Implement a mean reversion strategy based on the Hurst exponent
    
    This strategy only takes trades when the Hurst exponent indicates mean reversion
    (H < hurst_threshold)
    
    Parameters:
    prices_input (DataFrame): DataFrame with 'High', 'Low', 'Close' columns
    window_size (int): Size of the rolling window for Hurst calculation
    method (str): Method to use ('rs', 'dfa', 'aggvar', 'higuchi')
    max_lag (int): Maximum lag to consider
    hurst_threshold (float): Hurst threshold below which mean reversion is considered
    short_ma_period (int): Short-term moving average period for trend filter
    long_ma_period (int): Long-term moving average period for trend filter
    atr_period (int): Period for ATR calculation
    stop_loss_atr_multiplier (float): ATR multiplier for stop loss
    take_profit_atr_multiplier (float): ATR multiplier for take profit
    save_results (bool): Whether to save results to CSV
    csv_path (str): Path to save CSV results
    
    Returns:
    DataFrame: Strategy results
    """
    # Use the main strategy with a high trend threshold to effectively disable trend following
    return hurst_trading_strategy(
        prices_input,
        window_size=window_size,
        method=method,
        max_lag=max_lag,
        mean_reversion_threshold=hurst_threshold,
        trend_threshold=0.99,  # Set very high to disable trend following
        short_ma_period=short_ma_period,
        long_ma_period=long_ma_period,
        atr_period=atr_period,
        stop_loss_atr_multiplier=stop_loss_atr_multiplier,
        take_profit_atr_multiplier=take_profit_atr_multiplier,
        save_results=save_results,
        csv_path=csv_path
    )

def trend_following_strategy(prices_input,
                            window_size=120,
                            method='rs',
                            max_lag=20,
                            hurst_threshold=0.54,
                            short_ma_period=12,
                            long_ma_period=50,
                            atr_period=14,
                            stop_loss_atr_multiplier=1.8,
                            take_profit_atr_multiplier=3.0,
                            save_results=True,
                            csv_path='trend_following_results.csv'):
    """
    Implement a trend following strategy based on the Hurst exponent
    
    This strategy only takes trades when the Hurst exponent indicates trending
    (H > hurst_threshold)
    
    Parameters:
    prices_input (DataFrame): DataFrame with 'High', 'Low', 'Close' columns
    window_size (int): Size of the rolling window for Hurst calculation
    method (str): Method to use ('rs', 'dfa', 'aggvar', 'higuchi')
    max_lag (int): Maximum lag to consider
    hurst_threshold (float): Hurst threshold above which trending is considered
    short_ma_period (int): Short-term moving average period for trend filter
    long_ma_period (int): Long-term moving average period for trend filter
    atr_period (int): Period for ATR calculation
    stop_loss_atr_multiplier (float): ATR multiplier for stop loss
    take_profit_atr_multiplier (float): ATR multiplier for take profit
    save_results (bool): Whether to save results to CSV
    csv_path (str): Path to save CSV results
    
    Returns:
    DataFrame: Strategy results
    """
    # Use the main strategy with a low mean reversion threshold to effectively disable mean reversion
    return hurst_trading_strategy(
        prices_input,
        window_size=window_size,
        method=method,
        max_lag=max_lag,
        mean_reversion_threshold=0.01,  # Set very low to disable mean reversion
        trend_threshold=hurst_threshold,
        short_ma_period=short_ma_period,
        long_ma_period=long_ma_period,
        atr_period=atr_period,
        stop_loss_atr_multiplier=stop_loss_atr_multiplier,
        take_profit_atr_multiplier=take_profit_atr_multiplier,
        save_results=save_results,
        csv_path=csv_path
    )