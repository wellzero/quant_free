import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional, Dict, Any, Union
from scipy import stats  # Add missing import for stats module

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
        # Validate inputs
        if method not in ['rs', 'dfa', 'aggvar', 'higuchi']:
            raise ValueError("Method must be one of 'rs', 'dfa', 'aggvar', 'higuchi'")
            
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
        """
        try:
            # Skip if we don't have enough data
            if len(strategy_df) < lookback:
                return self.mean_reversion_threshold, self.trend_threshold
            
            # Get recent Hurst values
            recent_hurst = strategy_df['hurst'].dropna().tail(lookback)
            
            if len(recent_hurst) < 10:  # Need at least 10 values for meaningful statistics
                return self.mean_reversion_threshold, self.trend_threshold
            
            # Calculate market regime indicators
            price_series = strategy_df['price'].tail(lookback)
            returns_series = strategy_df['returns'].tail(lookback) if 'returns' in strategy_df.columns else pd.Series([0])
            
            # Calculate volatility
            volatility = returns_series.std() * np.sqrt(252) if len(returns_series) > 1 else 0.2
            
            # Detect market regime (bull/bear/sideways)
            bull_market = price_series.iloc[-1] > price_series.iloc[0] * 1.05  # 5% up
            bear_market = price_series.iloc[-1] < price_series.iloc[0] * 0.95  # 5% down
            
            # Calculate percentiles of recent Hurst values
            hurst_10th = recent_hurst.quantile(0.10)
            hurst_25th = recent_hurst.quantile(0.25)
            hurst_75th = recent_hurst.quantile(0.75)
            hurst_90th = recent_hurst.quantile(0.90)
            
            # Calculate volatility of Hurst values
            hurst_std = recent_hurst.std()
            
            # Calculate volatility factor based on Hurst standard deviation
            volatility_factor = min(2.0, max(1.0, 1.0 + hurst_std * 3))
            
            # Calculate performance of recent trades
            recent_returns = []
            if 'strategy_returns' in strategy_df.columns:
                recent_returns = strategy_df['strategy_returns'].tail(lookback).dropna()
            
            # Calculate win rate if we have enough trades
            win_rate = 0.5  # Default
            if len(recent_returns) > 5:
                win_rate = (recent_returns > 0).mean()
            
            # Make thresholds more conservative to improve win rate
            # For mean reversion, use a lower threshold (more extreme values)
            mean_rev_threshold = max(0.25, min(0.35, hurst_10th))
            
            # For trend following, use a higher threshold (more extreme values)
            trend_threshold = max(0.65, min(0.85, hurst_90th))
            
            # If win rate is low, make thresholds even more conservative
            if win_rate < 0.3:
                mean_rev_threshold = max(0.20, mean_rev_threshold - 0.05)
                trend_threshold = min(0.90, trend_threshold + 0.05)
            
            # Ensure minimum gap between thresholds
            min_gap = 0.25 * volatility_factor
            if (trend_threshold - mean_rev_threshold) < min_gap:
                mean_rev_threshold = max(0.2, (mean_rev_threshold + trend_threshold - min_gap) / 2)
                trend_threshold = min(0.9, mean_rev_threshold + min_gap)
            
            return mean_rev_threshold, trend_threshold
        except Exception as e:
            logger.error(f"Error calculating adaptive thresholds: {e}")
            return self.mean_reversion_threshold, self.trend_threshold
    
    def generate_signals(self, strategy_df: pd.DataFrame, i: int, 
                 mr_threshold: float, tr_threshold: float) -> Tuple[int, float, str]:
        """
        Generate trading signals based on Hurst exponent and other indicators.
        """
        try:
            current_idx = strategy_df.index[i]
            current_price = strategy_df.loc[current_idx, 'price']
            current_hurst = strategy_df.loc[current_idx, 'hurst']
            
            # Handle missing RSI values
            if 'rsi' in strategy_df.columns:
                current_rsi = strategy_df.loc[current_idx, 'rsi']
            else:
                current_rsi = 50  # Default neutral value
            
            # Add Hurst consistency check - more stringent requirements
            hurst_trend = []
            if i >= 7:  # Check last 7 Hurst values for stronger consistency
                for j in range(1, 8):
                    if i-j >= 0 and not pd.isna(strategy_df['hurst'].iloc[i-j]):
                        hurst_trend.append(strategy_df['hurst'].iloc[i-j])
            
            hurst_consistent = False
            if len(hurst_trend) >= 4:  # Need at least 4 values to check consistency
                if current_hurst < mr_threshold:
                    # For mean reversion, check if Hurst has been consistently low
                    hurst_consistent = all(h < mr_threshold + 0.03 for h in hurst_trend)
                elif current_hurst > tr_threshold:
                    # For trend following, check if Hurst has been consistently high
                    hurst_consistent = all(h > tr_threshold - 0.03 for h in hurst_trend)
            else:
                hurst_consistent = True  # Default to true if we don't have enough data
            
            # Calculate volatility-based position sizing with Hurst confidence
            recent_volatility = strategy_df['returns'].iloc[max(0, i-30):i].std() * np.sqrt(252) if i > 0 and 'returns' in strategy_df.columns else 0.2
            
            # Calculate Hurst confidence - how far from neutral (0.5)
            hurst_confidence = min(1.0, 2.0 * abs(current_hurst - 0.5))
            
            # Calculate position size based on volatility and Hurst confidence
            if recent_volatility > 0:
                position_size_factor = min(1.0, (0.2 / recent_volatility) * hurst_confidence)
            else:
                position_size_factor = 0.5 * hurst_confidence
            
            position = 0
            strategy_type = None
            
            # Add market regime detection
            market_regime = 'neutral'
            if i >= 50:
                # Simple trend detection based on price vs moving average
                price_50d_ago = strategy_df['price'].iloc[i-50]
                price_change_pct = (current_price / price_50d_ago - 1) * 100
                
                if price_change_pct > 5:
                    market_regime = 'bullish'
                elif price_change_pct < -5:
                    market_regime = 'bearish'
            
            # Mean reversion signal (Hurst < threshold) with stronger confirmation
            if current_hurst < mr_threshold and hurst_consistent:
                # Enhanced mean reversion logic with RSI filter and stronger deviation
                if current_price > strategy_df.loc[current_idx, 'short_ma'] * 1.015 and current_rsi > 75:
                    # Price significantly above MA and strongly overbought -> go short for mean reversion
                    position = -1
                    strategy_type = 'mean_reversion'
                elif current_price < strategy_df.loc[current_idx, 'short_ma'] * 0.985 and current_rsi < 25:
                    # Price significantly below MA and strongly oversold -> go long for mean reversion
                    position = 1
                    strategy_type = 'mean_reversion'
            
            # Trend following signal (Hurst > threshold) with stronger confirmation
            elif current_hurst > tr_threshold and hurst_consistent:
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
                
                # Stronger trend confirmation requirements
                if short_ma > long_ma * 1.01 and momentum > 1.5 and (market_regime != 'bearish' or momentum > 3.0):
                    # Strong uptrend with positive momentum -> go long
                    position = 1
                    strategy_type = 'trend_following'
                elif short_ma < long_ma * 0.99 and momentum < -1.5 and (market_regime != 'bullish' or momentum < -3.0):
                    # Strong downtrend with negative momentum -> go short
                    position = -1
                    strategy_type = 'trend_following'
            
            # Reduce position size in high volatility environments
            if recent_volatility > 0.25:  # High volatility
                position_size_factor *= 0.7
            
            # Reduce position size when trading against the market regime
            if (market_regime == 'bullish' and position < 0) or (market_regime == 'bearish' and position > 0):
                position_size_factor *= 0.8
            
            return position, position_size_factor, strategy_type
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return 0, 0.0, None
    
    def manage_position(self, strategy_df: pd.DataFrame, i: int, 
                    current_position: float, entry_price: Optional[float]) -> Tuple[float, Optional[float], str]:
        """
        Manage existing positions with stop loss, take profit, and time-based exits.
        """
        try:
            current_idx = strategy_df.index[i]
            current_price = strategy_df.loc[current_idx, 'price']
            current_atr = strategy_df.loc[current_idx, 'atr']
            current_hurst = strategy_df.loc[current_idx, 'hurst'] if 'hurst' in strategy_df.columns else 0.5
            exit_reason = None
            
            # Early return if no position or no entry price
            if current_position == 0 or entry_price is None:
                return current_position, entry_price, exit_reason
                
            # Check for None values in critical variables
            if current_price is None or current_atr is None:
                logger.warning(f"Missing critical values at index {i}: price={current_price}, atr={current_atr}")
                return current_position, entry_price, exit_reason
            
            # Get strategy type - handle case where it might be None
            strategy_type = None
            if 'strategy_type' in strategy_df.columns:
                strategy_type = strategy_df.loc[current_idx, 'strategy_type']
            
            # For long positions
            if current_position > 0:
                # Calculate unrealized profit
                unrealized_profit_pct = (current_price - entry_price) / entry_price
                
                # Check for Hurst reversal - exit if Hurst moves significantly against our position
                if strategy_type == 'trend_following' and current_hurst < self.trend_threshold - 0.15:
                    current_position = 0
                    entry_price = None
                    exit_reason = 'hurst_reversal'
                    return current_position, entry_price, exit_reason
                elif strategy_type == 'mean_reversion' and current_hurst > self.mean_reversion_threshold + 0.15:
                    current_position = 0
                    entry_price = None
                    exit_reason = 'hurst_reversal'
                    return current_position, entry_price, exit_reason
                
                # Adjust stop loss based on profit - more aggressive trailing stops
                if unrealized_profit_pct > 0.03:  # If we have more than 3% profit
                    # Use tighter trailing stop
                    trailing_stop = max(
                        entry_price,  # Don't go below entry price
                        current_price * (1 - 0.4 * self.stop_loss_atr_multiplier * current_atr / current_price)
                    )
                    stop_price = trailing_stop
                elif unrealized_profit_pct > 0.015:  # If we have more than 1.5% profit
                    # Use medium trailing stop
                    trailing_stop = max(
                        entry_price * 0.998,  # Allow tiny loss to secure most profit
                        current_price * (1 - 0.6 * self.stop_loss_atr_multiplier * current_atr / current_price)
                    )
                    stop_price = trailing_stop
                else:
                    # Normal stop loss
                    stop_price = entry_price - (self.stop_loss_atr_multiplier * current_atr)
                    
                # Tighter take profit for mean reversion
                if strategy_type == 'mean_reversion':
                    take_profit_price = entry_price + (1.8 * current_atr)  # Smaller target for mean reversion
                else:
                    take_profit_price = entry_price + (self.take_profit_atr_multiplier * current_atr)
                
                # Time-based exit for stale positions - more aggressive for mean reversion
                max_duration = 8 if strategy_type == 'mean_reversion' else 15
                
                # Safely calculate position duration
                try:
                    position_mask = strategy_df['position'].iloc[:i].eq(current_position)
                    if position_mask.any():
                        position_start_idx = position_mask.iloc[::-1].idxmax()
                        position_duration = i - strategy_df.index.get_loc(position_start_idx)
                    else:
                        position_duration = 0
                except Exception as e:
                    logger.warning(f"Error calculating position duration: {e}")
                    position_duration = 0
                
                # Exit stale positions more aggressively
                if position_duration > max_duration and unrealized_profit_pct < 0.01:
                    # Exit stale positions that aren't making progress
                    current_position = 0
                    entry_price = None
                    exit_reason = 'time_exit'
                elif position_duration > max_duration * 1.5:
                    # Force exit very old positions regardless of profit
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
            
            # For short positions (similar logic with inverted price comparisons)
            elif current_position < 0:
                # Calculate unrealized profit
                unrealized_profit_pct = (entry_price - current_price) / entry_price
                
                # Check for Hurst reversal
                if strategy_type == 'trend_following' and current_hurst < self.trend_threshold - 0.15:
                    current_position = 0
                    entry_price = None
                    exit_reason = 'hurst_reversal'
                    return current_position, entry_price, exit_reason
                elif strategy_type == 'mean_reversion' and current_hurst > self.mean_reversion_threshold + 0.15:
                    current_position = 0
                    entry_price = None
                    exit_reason = 'hurst_reversal'
                    return current_position, entry_price, exit_reason
                
                # Adjust stop loss based on profit
                if unrealized_profit_pct > 0.03:  # If we have more than 3% profit
                    # Use tighter trailing stop
                    trailing_stop = min(
                        entry_price,  # Don't go above entry price
                        current_price * (1 + 0.4 * self.stop_loss_atr_multiplier * current_atr / current_price)
                    )
                    stop_price = trailing_stop
                elif unrealized_profit_pct > 0.015:  # If we have more than 1.5% profit
                    # Use medium trailing stop
                    trailing_stop = min(
                        entry_price * 1.002,  # Allow tiny loss to secure most profit
                        current_price * (1 + 0.6 * self.stop_loss_atr_multiplier * current_atr / current_price)
                    )
                    stop_price = trailing_stop
                else:
                    # Normal stop loss
                    stop_price = entry_price + (self.stop_loss_atr_multiplier * current_atr)
                    
                # Tighter take profit for mean reversion
                if strategy_type == 'mean_reversion':
                    take_profit_price = entry_price - (1.8 * current_atr)
                else:
                    take_profit_price = entry_price - (self.take_profit_atr_multiplier * current_atr)
                
                # Time-based exit
                max_duration = 8 if strategy_type == 'mean_reversion' else 15
                
                # Safely calculate position duration
                try:
                    position_mask = strategy_df['position'].iloc[:i].eq(current_position)
                    if position_mask.any():
                        position_start_idx = position_mask.iloc[::-1].idxmax()
                        position_duration = i - strategy_df.index.get_loc(position_start_idx)
                    else:
                        position_duration = 0
                except Exception as e:
                    logger.warning(f"Error calculating position duration: {e}")
                    position_duration = 0
                
                # Exit stale positions more aggressively
                if position_duration > max_duration and unrealized_profit_pct < 0.01:
                    current_position = 0
                    entry_price = None
                    exit_reason = 'time_exit'
                elif position_duration > max_duration * 1.5:
                    # Force exit very old positions regardless of profit
                    current_position = 0
                    entry_price = None
                    exit_reason = 'time_exit'
                elif current_price >= stop_price:
                    current_position = 0
                    entry_price = None
                    exit_reason = 'stop_loss'
                elif current_price <= take_profit_price:
                    current_position = 0
                    entry_price = None
                    exit_reason = 'take_profit'
            
            return current_position, entry_price, exit_reason
            
        except Exception as e:
            logger.error(f"Error managing position: {e}")
            return current_position, entry_price, 'error'
    
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
        try:
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
                    try:
                        market_regime = detect_market_regime(strategy_df['price'].iloc[i-60:i])
                        strategy_df.loc[current_idx, 'market_regime'] = market_regime
                    except Exception as e:
                        logger.warning(f"Error detecting market regime: {e}")
                        market_regime = 'unknown'
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
                        strategy_df.loc[current_idx, 'position'] = current_position
                else:
                    # Generate new signal
                    try:
                        # Apply correlation filter
                        if not _check_correlation_filter(strategy_df, i):
                            continue
                        
                        # Generate signals
                        position, position_size, strategy_type = self.generate_signals(
                            strategy_df, i, mr_threshold, tr_threshold
                        )
                        
                        # Update position if we have a signal
                        if position != 0:
                            current_position = position
                            entry_price = current_price
                            
                            # Set stop loss and take profit levels
                            if current_position > 0:
                                stop_loss = entry_price - (current_atr * self.stop_loss_atr_multiplier)
                                take_profit = entry_price + (current_atr * self.take_profit_atr_multiplier)
                            else:
                                stop_loss = entry_price + (current_atr * self.stop_loss_atr_multiplier)
                                take_profit = entry_price - (current_atr * self.take_profit_atr_multiplier)
                            
                            # Update strategy DataFrame
                            strategy_df.loc[current_idx, 'position'] = current_position
                            strategy_df.loc[current_idx, 'entry_price'] = entry_price
                            strategy_df.loc[current_idx, 'stop_loss'] = stop_loss
                            strategy_df.loc[current_idx, 'take_profit'] = take_profit
                            strategy_df.loc[current_idx, 'strategy_type'] = strategy_type
                        else:
                            strategy_df.loc[current_idx, 'position'] = 0
                    except Exception as e:
                        logger.error(f"Error generating new signal: {e}")
                        strategy_df.loc[current_idx, 'position'] = 0
            
            # Calculate strategy returns
            strategy_df['strategy_returns'] = strategy_df['position'].shift(1) * strategy_df['returns']
            strategy_df['strategy_returns'] = strategy_df['strategy_returns'].fillna(0)
            
            # Calculate cumulative returns
            strategy_df['cum_returns'] = (1 + strategy_df['returns']).cumprod()
            strategy_df['cum_strategy_returns'] = (1 + strategy_df['strategy_returns']).cumprod()
            
            return strategy_df
            
        except Exception as e:
            logger.error(f"Error processing signals and positions: {e}")
            raise
    
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