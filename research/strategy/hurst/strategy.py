import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional, Dict, Any, Union

from .utils import (
    _validate_inputs, 
    _prepare_strategy_dataframe, 
    _calculate_hurst_values, 
    _add_hurst_to_dataframe,
    _check_correlation_filter,
    detect_market_regime,
    calculate_atr,
    calculate_rsi
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HurstStrategy:
    """
    A class implementing trading strategies based on the Hurst exponent.
    
    This class provides methods for both mean reversion and trend following
    strategies based on the Hurst exponent value.
    """
    
    def __init__(
        self,
        window_size: int = 90,
        method: str = 'rs',
        max_lag: int = 20,
        mean_reversion_threshold: float = 0.32,
        trend_threshold: float = 0.62,
        short_ma_period: int = 8,
        long_ma_period: int = 34,
        atr_period: int = 14,
        stop_loss_atr_multiplier: float = 2.5,
        take_profit_atr_multiplier: float = 5.0,
        adaptive_thresholds: bool = True
    ):
        """
        Initialize the Hurst trading strategy.
        
        Parameters:
        -----------
        window_size : int
            Size of the rolling window for Hurst calculation
        method : str
            Method to use ('rs', 'dfa', 'aggvar', 'higuchi')
        max_lag : int
            Maximum lag to consider
        mean_reversion_threshold : float
            Hurst threshold below which mean reversion is considered
        trend_threshold : float
            Hurst threshold above which trending is considered
        short_ma_period : int
            Short-term moving average period for trend filter
        long_ma_period : int
            Long-term moving average period for trend filter
        atr_period : int
            Period for ATR calculation
        stop_loss_atr_multiplier : float
            ATR multiplier for stop loss
        take_profit_atr_multiplier : float
            ATR multiplier for take profit
        adaptive_thresholds : bool
            Whether to use adaptive thresholds
        """
        self.window_size = window_size
        self.method = method
        self.max_lag = max_lag
        self.mean_reversion_threshold = mean_reversion_threshold
        self.trend_threshold = trend_threshold
        self.short_ma_period = short_ma_period
        self.long_ma_period = long_ma_period
        self.atr_period = atr_period
        self.stop_loss_atr_multiplier = stop_loss_atr_multiplier
        self.take_profit_atr_multiplier = take_profit_atr_multiplier
        self.adaptive_thresholds = adaptive_thresholds
    
    def calculate_adaptive_thresholds(self, strategy_df: pd.DataFrame, lookback: int = 100) -> Tuple[float, float]:
        """
        Calculate adaptive thresholds based on recent Hurst values.
        
        Parameters:
        -----------
        strategy_df : DataFrame
            Strategy DataFrame with Hurst values
        lookback : int
            Lookback period for threshold calculation
            
        Returns:
        --------
        Tuple[float, float]
            Mean reversion threshold and trend threshold
        """
        # Skip if we don't have enough data
        if len(strategy_df) < lookback:
            return self.mean_reversion_threshold, self.trend_threshold
        
        # Get recent Hurst values
        recent_hurst = strategy_df['hurst'].dropna().tail(lookback)
        
        if len(recent_hurst) < 10:  # Need at least 10 values for meaningful statistics
            return self.mean_reversion_threshold, self.trend_threshold
        
        # Calculate market regime indicators
        price_series = strategy_df['price'].tail(lookback)
        returns = price_series.pct_change().dropna()
        
        # Detect market regime (bull/bear/sideways)
        bull_market = price_series.iloc[-1] > price_series.iloc[0] * 1.05  # 5% up
        bear_market = price_series.iloc[-1] < price_series.iloc[0] * 0.95  # 5% down
        
        # Calculate percentiles of recent Hurst values
        hurst_15th = recent_hurst.quantile(0.15)
        hurst_85th = recent_hurst.quantile(0.85)
        
        # Calculate volatility of Hurst values
        hurst_std = recent_hurst.std()
        
        # Calculate volatility factor based on Hurst standard deviation
        volatility_factor = min(2.0, max(1.0, 1.0 + hurst_std * 3))
        
        # Adjust thresholds based on market regime
        if bull_market:
            # In bull markets, favor trend following by lowering the trend threshold
            mean_rev_threshold = max(0.25, min(0.40, hurst_15th))
            trend_threshold = max(0.52, min(0.85, hurst_85th - 0.03))
        elif bear_market:
            # In bear markets, be more selective with both strategies
            mean_rev_threshold = max(0.20, min(0.35, hurst_15th - 0.02))
            trend_threshold = max(0.58, min(0.90, hurst_85th + 0.02))
        else:
            # In sideways markets, favor mean reversion
            mean_rev_threshold = max(0.25, min(0.42, hurst_15th + 0.02))
            trend_threshold = max(0.55, min(0.88, hurst_85th))
        
        # Widen the gap in high volatility environments
        gap = trend_threshold - mean_rev_threshold
        if gap < 0.2 * volatility_factor:
            mean_rev_threshold = max(0.2, mean_rev_threshold - 0.03)
            trend_threshold = min(0.9, trend_threshold + 0.03)
        
        return mean_rev_threshold, trend_threshold
    
    def generate_signals(self, strategy_df: pd.DataFrame, i: int, 
                         mr_threshold: float, tr_threshold: float) -> Tuple[int, float, str]:
        """
        Generate trading signals based on Hurst exponent and other indicators.
        
        Parameters:
        -----------
        strategy_df : DataFrame
            Strategy DataFrame with price and indicator data
        i : int
            Current index in the DataFrame
        mr_threshold : float
            Mean reversion threshold
        tr_threshold : float
            Trend threshold
            
        Returns:
        --------
        Tuple[int, float, str]
            Position, position size factor, and strategy type
        """
        current_idx = strategy_df.index[i]
        current_price = strategy_df.loc[current_idx, 'price']
        current_hurst = strategy_df.loc[current_idx, 'hurst']
        current_rsi = strategy_df.loc[current_idx, 'rsi']
        
        # Calculate volatility-based position sizing
        recent_volatility = strategy_df['returns'].iloc[max(0, i-30):i].std() * np.sqrt(252)
        if recent_volatility > 0:
            position_size_factor = min(1.0, 0.2 / recent_volatility)  # Target 20% annualized volatility
        else:
            position_size_factor = 0.5  # Default if volatility calculation fails
        
        position = 0
        strategy_type = None
        
        # Mean reversion signal (Hurst < threshold)
        if current_hurst < mr_threshold:
            # Enhanced mean reversion logic with RSI filter
            if current_price > strategy_df.loc[current_idx, 'short_ma'] and current_rsi > 75:
                # Price above MA and overbought -> go short for mean reversion
                position = -1
                strategy_type = 'mean_reversion'
            elif current_price < strategy_df.loc[current_idx, 'short_ma'] and current_rsi < 25:
                # Price below MA and oversold -> go long for mean reversion
                position = 1
                strategy_type = 'mean_reversion'
        
        # Trend following signal (Hurst > threshold)
        elif current_hurst > tr_threshold:
            # Enhanced trend following with momentum confirmation
            short_ma = strategy_df.loc[current_idx, 'short_ma']
            long_ma = strategy_df.loc[current_idx, 'long_ma']
            
            # Calculate momentum (rate of change)
            if i >= 10:
                momentum = (current_price / strategy_df['price'].iloc[i-10] - 1) * 100
            else:
                momentum = 0
                
            # Add volume trend if available
            volume_trend = 1  # Default neutral
            if 'Volume' in strategy_df.columns:
                recent_volume = strategy_df['Volume'].iloc[max(0, i-10):i].mean()
                prev_volume = strategy_df['Volume'].iloc[max(0, i-20):max(0, i-10)].mean()
                if not np.isnan(recent_volume) and not np.isnan(prev_volume) and prev_volume > 0:
                    volume_trend = recent_volume / prev_volume
            
            if short_ma > long_ma and momentum > 1.0:
                # Strong uptrend with positive momentum -> go long
                position = 1
                strategy_type = 'trend_following'
            elif short_ma < long_ma and momentum < -1.0:
                # Strong downtrend with negative momentum -> go short
                position = -1
                strategy_type = 'trend_following'
        
        return position, position_size_factor, strategy_type
    
    def manage_position(self, strategy_df: pd.DataFrame, i: int, 
                        current_position: float, entry_price: Optional[float]) -> Tuple[float, Optional[float], str]:
        """
        Manage existing positions with stop loss, take profit, and time-based exits.
        
        Parameters:
        -----------
        strategy_df : DataFrame
            Strategy DataFrame with price and indicator data
        i : int
            Current index in the DataFrame
        current_position : float
            Current position size
        entry_price : float or None
            Entry price for the current position
            
        Returns:
        --------
        Tuple[float, float or None, str]
            Updated position, entry price, and exit reason
        """
        current_idx = strategy_df.index[i]
        current_price = strategy_df.loc[current_idx, 'price']
        current_atr = strategy_df.loc[current_idx, 'atr']
        exit_reason = None
        
        if current_position == 0 or entry_price is None:
            return current_position, entry_price, exit_reason
        
        # Get position duration
        try:
            # Find when the current position started
            position_mask = strategy_df['position'].iloc[:i].eq(current_position)
            if position_mask.any():
                position_start_idx = position_mask.iloc[::-1].idxmax()
                position_duration = (current_idx - position_start_idx).days
                if pd.isna(position_duration):  # Handle case where days calculation fails
                    position_duration = i - strategy_df.index.get_loc(position_start_idx)
            else:
                position_duration = 0
        except Exception as e:
            logger.warning(f"Error calculating position duration: {e}")
            position_duration = 0
        
        # For long positions
        if current_position > 0:
            # Calculate unrealized profit
            unrealized_profit_pct = (current_price - entry_price) / entry_price
            
            # Adjust stop loss based on profit
            if unrealized_profit_pct > 0.02:  # If we have more than 2% profit
                # Use trailing stop that's tighter than initial stop
                trailing_stop = max(
                    entry_price,  # Don't go below entry price
                    current_price * (1 - 0.5 * self.stop_loss_atr_multiplier * current_atr / current_price)
                )
                stop_price = trailing_stop
            else:
                # Normal stop loss
                stop_price = entry_price - (self.stop_loss_atr_multiplier * current_atr)
                
            take_profit_price = entry_price + (self.take_profit_atr_multiplier * current_atr)
            
            # Time-based exit for stale positions
            if position_duration > 20 and unrealized_profit_pct < 0.005:
                # Exit stale positions that aren't making progress
                current_position = 0
                entry_price = None
                exit_reason = 'time_exit'
            elif current_price <= stop_price:
                # Stop loss hit
                current_position = 0
                entry_price = None
                exit_reason = 'stop_loss'
            elif current_price >= take_profit_price:
                # Take profit hit
                current_position = 0
                entry_price = None
                exit_reason = 'take_profit'
        
        # For short positions
        elif current_position < 0:
            # Calculate unrealized profit
            unrealized_profit_pct = (entry_price - current_price) / entry_price
            
            # Adjust stop loss based on profit
            if unrealized_profit_pct > 0.02:  # If we have more than 2% profit
                # Use trailing stop that's tighter than initial stop
                trailing_stop = min(
                    entry_price,  # Don't go above entry price
                    current_price * (1 + 0.5 * self.stop_loss_atr_multiplier * current_atr / current_price)
                )
                stop_price = trailing_stop
            else:
                # Normal stop loss
                stop_price = entry_price + (self.stop_loss_atr_multiplier * current_atr)
                
            take_profit_price = entry_price - (self.take_profit_atr_multiplier * current_atr)
            
            # Time-based exit for stale positions
            if position_duration > 20 and unrealized_profit_pct < 0.005:
                # Exit stale positions that aren't making progress
                current_position = 0
                entry_price = None
                exit_reason = 'time_exit'
            elif current_price >= stop_price:
                # Stop loss hit
                current_position = 0
                entry_price = None
                exit_reason = 'stop_loss'
            elif current_price <= take_profit_price:
                # Take profit hit
                current_position = 0
                entry_price = None
                exit_reason = 'take_profit'
        
        # Add dynamic trailing stop logic for trend following positions
        if current_position != 0 and entry_price is not None:
            current_idx = strategy_df.index[i]
            current_price = strategy_df.loc[current_idx, 'price']
            current_atr = strategy_df.loc[current_idx, 'atr']
            strategy_type = strategy_df.loc[current_idx, 'strategy_type']
            
            # Calculate profit in ATR units
            if current_position > 0:
                profit_atr_units = (current_price - entry_price) / current_atr
            else:
                profit_atr_units = (entry_price - current_price) / current_atr
            
            # Implement trailing stops for trend following positions
            if strategy_type == 'trend_following' and profit_atr_units > 2.0:
                # Tighten stop loss as profit increases
                trailing_stop_multiplier = max(1.0, self.stop_loss_atr_multiplier - (profit_atr_units * 0.1))
                
                if current_position > 0:
                    new_stop = current_price - (current_atr * trailing_stop_multiplier)
                    if new_stop > strategy_df.loc[current_idx, 'stop_loss']:
                        strategy_df.loc[current_idx, 'stop_loss'] = new_stop
                else:
                    new_stop = current_price + (current_atr * trailing_stop_multiplier)
                    if new_stop < strategy_df.loc[current_idx, 'stop_loss']:
                        strategy_df.loc[current_idx, 'stop_loss'] = new_stop
    
        return current_position, entry_price, exit_reason
    
    def process_signals_and_positions(self, strategy_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process signals and manage positions based on Hurst exponent values.
        
        Parameters:
        -----------
        strategy_df : DataFrame
            Strategy DataFrame with price and indicator data
            
        Returns:
        --------
        DataFrame
            Updated strategy DataFrame with positions and signals
        """
        # Skip initial rows where we don't have enough data
        start_idx = max(self.window_size, self.long_ma_period)
        
        # Initialize position tracking variables
        current_position = 0
        entry_price = None
        
        # Process each row in the DataFrame
        for i in range(start_idx, len(strategy_df)):
            current_idx = strategy_df.index[i]
            
            # Detect market regime for the last 60 days
            if i >= 60 + self.window_size:
                market_regime = detect_market_regime(strategy_df['price'].iloc[i-60:i])
                strategy_df.loc[current_idx, 'market_regime'] = market_regime
            else:
                market_regime = 'unknown'

            # Skip if we don't have a valid Hurst value
            if np.isnan(strategy_df.loc[current_idx, 'hurst']):
                continue
            
            # Get current price and ATR
            current_price = strategy_df.loc[current_idx, 'price']
            current_atr = strategy_df.loc[current_idx, 'atr']
            
            # Use adaptive thresholds if enabled
            if self.adaptive_thresholds:
                mr_threshold, tr_threshold = self.calculate_adaptive_thresholds(strategy_df.iloc[:i])
                # Store the adaptive thresholds for analysis
                strategy_df.loc[current_idx, 'adaptive_mr_threshold'] = mr_threshold
                strategy_df.loc[current_idx, 'adaptive_tr_threshold'] = tr_threshold
            else:
                mr_threshold, tr_threshold = self.mean_reversion_threshold, self.trend_threshold

            # Adjust thresholds based on market regime
            if market_regime == 'trending_up' or market_regime == 'trending_down':
                # In trending markets, lower the trend threshold
                tr_threshold = max(0.51, tr_threshold - 0.03)
            elif market_regime == 'mean_reverting':
                # In mean-reverting markets, raise the mean reversion threshold
                mr_threshold = min(0.42, mr_threshold + 0.03)
            elif market_regime == 'volatile':
                # In volatile markets, be more conservative with both strategies
                mr_threshold = max(0.25, mr_threshold - 0.05)
                tr_threshold = min(0.65, tr_threshold + 0.05)

            # Check if we have an existing position
            if current_position != 0:
                # Manage existing position
                updated_position, updated_entry_price, exit_reason = self.manage_position(
                    strategy_df, i, current_position, entry_price
                )
                
                # Update position if it changed
                if updated_position != current_position:
                    strategy_df.loc[current_idx, 'position'] = updated_position
                    strategy_df.loc[current_idx, 'exit_reason'] = exit_reason
                    current_position = updated_position
                    entry_price = updated_entry_price
                else:
                    # Maintain position
                    strategy_df.loc[current_idx, 'position'] = current_position
            
            # Check for new entry signals if not in a position
            if current_position == 0:
                # Apply correlation filter
                if not _check_correlation_filter(strategy_df, i):
                    continue
                
                # Generate signals
                position, position_size_factor, strategy_type = self.generate_signals(
                    strategy_df, i, mr_threshold, tr_threshold
                )
                
                if position != 0:
                    # Apply position sizing
                    sized_position = position * position_size_factor
                    
                    # Update strategy DataFrame
                    strategy_df.loc[current_idx, 'raw_signal'] = position
                    # Explicitly convert to float to avoid dtype warning
                    strategy_df.loc[current_idx, 'position'] = float(sized_position)
                    strategy_df.loc[current_idx, 'strategy_type'] = strategy_type
                    
                    # Update tracking variables
                    current_position = sized_position
                    entry_price = current_price
            
            # Store entry price, stop loss and take profit levels
            if entry_price is not None:
                strategy_df.loc[current_idx, 'entry_price'] = entry_price
                
                if current_position > 0:
                    strategy_df.loc[current_idx, 'stop_loss'] = entry_price - (self.stop_loss_atr_multiplier * current_atr)
                    strategy_df.loc[current_idx, 'take_profit'] = entry_price + (self.take_profit_atr_multiplier * current_atr)
                elif current_position < 0:
                    strategy_df.loc[current_idx, 'stop_loss'] = entry_price + (self.stop_loss_atr_multiplier * current_atr)
                    strategy_df.loc[current_idx, 'take_profit'] = entry_price - (self.take_profit_atr_multiplier * current_atr)
        
        return strategy_df
    
    def calculate_strategy_returns(self, strategy_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate strategy returns based on positions.
        
        Parameters:
        -----------
        strategy_df : DataFrame
            Strategy DataFrame with positions
            
        Returns:
        --------
        DataFrame
            Updated strategy DataFrame with returns
        """
        # Calculate strategy returns
        strategy_df['strategy_returns'] = strategy_df['position'].shift(1) * strategy_df['returns']
        
        # Calculate cumulative returns
        strategy_df['cum_returns'] = (1 + strategy_df['returns']).cumprod()
        strategy_df['cum_strategy_returns'] = (1 + strategy_df['strategy_returns']).cumprod()
        
        return strategy_df
    
    def run(self, prices_input: pd.DataFrame, save_results: bool = True, 
            csv_path: str = 'hurst_strategy_results.csv') -> pd.DataFrame:
        """
        Run the Hurst trading strategy on the input price data.
        
        Parameters:
        -----------
        prices_input : DataFrame
            DataFrame with 'High', 'Low', 'Close' columns
        save_results : bool
            Whether to save results to CSV
        csv_path : str
            Path to save CSV results
            
        Returns:
        --------
        DataFrame
            Strategy results
        """
        try:
            # Validate inputs
            _validate_inputs(prices_input)
            
            # Prepare data
            strategy_df = _prepare_strategy_dataframe(
                prices_input, 
                self.short_ma_period, 
                self.long_ma_period, 
                self.atr_period
            )
            
            # Add columns for strategy type tracking
            strategy_df['strategy_type'] = None
            if self.adaptive_thresholds:
                strategy_df['adaptive_mr_threshold'] = np.nan
                strategy_df['adaptive_tr_threshold'] = np.nan
            
            # Calculate Hurst exponents
            hurst_values = _calculate_hurst_values(
                prices_input['Close'], 
                self.window_size, 
                self.method, 
                self.max_lag
            )
            _add_hurst_to_dataframe(strategy_df, hurst_values, self.window_size)
            
            # Process signals and manage positions
            strategy_df = self.process_signals_and_positions(strategy_df)
            
            # Calculate returns
            strategy_df = self.calculate_strategy_returns(strategy_df)
            
            # Save results if requested
            if save_results:
                try:
                    strategy_df.to_csv(csv_path)
                    logger.info(f"Strategy results saved to {csv_path}")
                except Exception as e:
                    logger.error(f"Error saving results to CSV: {e}")
            
            return strategy_df
            
        except Exception as e:
            logger.error(f"Error running Hurst strategy: {e}")
            raise

# Wrapper functions for backward compatibility

def hurst_trading_strategy(
    prices_input: pd.DataFrame,
    window_size: int = 90,
    method: str = 'rs',
    max_lag: int = 20,
    mean_reversion_threshold: float = 0.32,
    trend_threshold: float = 0.62,
    short_ma_period: int = 8,
    long_ma_period: int = 34,
    atr_period: int = 14,
    stop_loss_atr_multiplier: float = 2.5,
    take_profit_atr_multiplier: float = 5.0,
    adaptive_thresholds: bool = True,
    save_results: bool = True,
    csv_path: str = 'hurst_strategy_results.csv'
) -> pd.DataFrame:
    """
    Implement a trading strategy based on the Hurst exponent.
    
    Parameters:
    -----------
    prices_input : DataFrame
        DataFrame with 'High', 'Low', 'Close' columns
    window_size : int
        Size of the rolling window for Hurst calculation
    method : str
        Method to use ('rs', 'dfa', 'aggvar', 'higuchi')
    max_lag : int
        Maximum lag to consider
    mean_reversion_threshold : float
        Hurst threshold below which mean reversion is considered
    trend_threshold : float
        Hurst threshold above which trending is considered
    short_ma_period : int
        Short-term moving average period for trend filter
    long_ma_period : int
        Long-term moving average period for trend filter
    atr_period : int
        Period for ATR calculation
    stop_loss_atr_multiplier : float
        ATR multiplier for stop loss
    take_profit_atr_multiplier : float
        ATR multiplier for take profit
    adaptive_thresholds : bool
        Whether to use adaptive thresholds
    save_results : bool
        Whether to save results to CSV
    csv_path : str
        Path to save CSV results
    
    Returns:
    --------
    DataFrame
        Strategy results
    """
    strategy = HurstStrategy(
        window_size=window_size,
        method=method,
        max_lag=max_lag,
        mean_reversion_threshold=mean_reversion_threshold,
        trend_threshold=trend_threshold,
        short_ma_period=short_ma_period,
        long_ma_period=long_ma_period,
        atr_period=atr_period,
        stop_loss_atr_multiplier=stop_loss_atr_multiplier,
        take_profit_atr_multiplier=take_profit_atr_multiplier,
        adaptive_thresholds=adaptive_thresholds
    )
    
    return strategy.run(prices_input, save_results, csv_path)

def mean_reversion_strategy(
    prices_input: pd.DataFrame,
    window_size: int = 90,
    method: str = 'rs',
    max_lag: int = 20,
    hurst_threshold: float = 0.38,
    short_ma_period: int = 8,
    long_ma_period: int = 34,
    atr_period: int = 14,
    stop_loss_atr_multiplier: float = 2.5,
    take_profit_atr_multiplier: float = 5.0,
    save_results: bool = True,
    csv_path: str = 'mean_reversion_results.csv'
) -> pd.DataFrame:
    """
    Implement a mean reversion strategy based on the Hurst exponent.
    
    This strategy only takes trades when the Hurst exponent indicates mean reversion
    (H < hurst_threshold).
    
    Parameters:
    -----------
    prices_input : DataFrame
        DataFrame with 'High', 'Low', 'Close' columns
    window_size : int
        Size of the rolling window for Hurst calculation
    method : str
        Method to use ('rs', 'dfa', 'aggvar', 'higuchi')
    max_lag : int
        Maximum lag to consider
    hurst_threshold : float
        Hurst threshold below which mean reversion is considered
    short_ma_period : int
        Short-term moving average period for trend filter
    long_ma_period : int
        Long-term moving average period for trend filter
    atr_period : int
        Period for ATR calculation
    stop_loss_atr_multiplier : float
        ATR multiplier for stop loss
    take_profit_atr_multiplier : float
        ATR multiplier for take profit
    save_results : bool
        Whether to save results to CSV
    csv_path : str
        Path to save CSV results
    
    Returns:
    --------
    DataFrame
        Strategy results
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

def trend_following_strategy(
    prices_input: pd.DataFrame,
    window_size: int = 90,
    method: str = 'rs',
    max_lag: int = 20,
    hurst_threshold: float = 0.54,
    short_ma_period: int = 8,
    long_ma_period: int = 34,
    atr_period: int = 14,
    stop_loss_atr_multiplier: float = 2.5,
    take_profit_atr_multiplier: float = 5.0,
    save_results: bool = True,
    csv_path: str = 'trend_following_results.csv'
) -> pd.DataFrame:
    """
    Implement a trend following strategy based on the Hurst exponent.
    
    This strategy only takes trades when the Hurst exponent indicates trending
    (H > hurst_threshold).
    
    Parameters:
    -----------
    prices_input : DataFrame
        DataFrame with 'High', 'Low', 'Close' columns
    window_size : int
        Size of the rolling window for Hurst calculation
    method : str
        Method to use ('rs', 'dfa', 'aggvar', 'higuchi')
    max_lag : int
        Maximum lag to consider
    hurst_threshold : float
        Hurst threshold above which trending is considered
    short_ma_period : int
        Short-term moving average period for trend filter
    long_ma_period : int
        Long-term moving average period for trend filter
    atr_period : int
        Period for ATR calculation
    stop_loss_atr_multiplier : float
        ATR multiplier for stop loss
    take_profit_atr_multiplier : float
        ATR multiplier for take profit
    save_results : bool
        Whether to save results to CSV
    csv_path : str
        Path to save CSV results
    
    Returns:
    --------
    DataFrame
        Strategy results
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