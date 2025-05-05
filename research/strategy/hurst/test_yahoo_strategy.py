import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import os
import sys
# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import the strategy module
from hurst.strategy import hurst_trading_strategy, mean_reversion_strategy, trend_following_strategy
from hurst.core import hurst_rs, hurst_dfa, hurst_aggvar
from hurst.evaluation import evaluate_strategy
from hurst.visualization import plot_strategy_results

class TestHurstStrategyWithYahoo(unittest.TestCase):
    """Test cases for Hurst trading strategy using Yahoo Finance data"""
    
    def setUp(self):
        """Set up test data from Yahoo Finance"""
        # Create directory for test results
        os.makedirs('test_results', exist_ok=True)
        
        # Define test period
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=365*3)  # 3 years of data
        
        # Download data for different market types
        # Trending market (tech stock)
        self.trending_df = yf.download('MSFT', 
                                      start=self.start_date.strftime('%Y-%m-%d'),
                                      end=self.end_date.strftime('%Y-%m-%d'))
        
        # Mean-reverting market (utility stock)
        self.mean_reverting_df = yf.download('XLU', 
                                           start=self.start_date.strftime('%Y-%m-%d'),
                                           end=self.end_date.strftime('%Y-%m-%d'))
        
        # Random-like market (crypto)
        self.random_df = yf.download('BTC-USD', 
                                    start=self.start_date.strftime('%Y-%m-%d'),
                                    end=self.end_date.strftime('%Y-%m-%d'))
        
        # Commodity
        self.commodity_df = yf.download('GC=F',  # Gold futures
                                      start=self.start_date.strftime('%Y-%m-%d'),
                                      end=self.end_date.strftime('%Y-%m-%d'))
        
        # Forex
        self.forex_df = yf.download('EURUSD=X',
                                   start=self.start_date.strftime('%Y-%m-%d'),
                                   end=self.end_date.strftime('%Y-%m-%d'))
    
    def test_hurst_calculation_on_real_data(self):
        """Test Hurst exponent calculation on real market data"""
        # Calculate Hurst exponents for different assets
        assets = {
            'MSFT (Tech)': self.trending_df,
            'XLU (Utilities)': self.mean_reverting_df,
            'BTC-USD (Crypto)': self.random_df,
            'Gold': self.commodity_df,
            'EUR/USD': self.forex_df
        }
        
        results = {}
        window_size = 252  # One year of trading days
        
        for name, data in assets.items():
            if len(data) < window_size:
                print(f"Warning: Not enough data for {name}, skipping")
                continue
                
            # Calculate Hurst using different methods
            price_series = data['Close'].values[-window_size:]
            
            try:
                rs_hurst = hurst_rs(price_series, max_lag=20)
                dfa_hurst = hurst_dfa(price_series)
                aggvar_hurst = hurst_aggvar(price_series)
                
                results[name] = {
                    'RS': rs_hurst,
                    'DFA': dfa_hurst,
                    'AggVar': aggvar_hurst
                }
            except Exception as e:
                print(f"Error calculating Hurst for {name}: {e}")
        
        # Print results
        print("\nHurst Exponent Calculation on Real Market Data:")
        for asset, methods in results.items():
            print(f"{asset} - RS: {methods['RS']:.3f}, DFA: {methods['DFA']:.3f}, AggVar: {methods['AggVar']:.3f}")
    
    def test_strategy_on_trending_market(self):
        """Test strategy performance on trending market (MSFT) with different Hurst methods"""
        # Define Hurst calculation methods to test
        hurst_methods = ['rs', 'dfa', 'aggvar']
        
        best_method = None
        best_sharpe = -float('inf')
        best_results = None
        
        print("\nStrategy Performance on Trending Market (MSFT) with Different Hurst Methods:")
        
        for method in hurst_methods:
            # Run strategy on trending data with current method
            strategy_results = hurst_trading_strategy(
                self.trending_df,
                window_size=100,
                method=method,  # Use current method in iteration
                max_lag=20,
                mean_reversion_threshold=0.38,
                trend_threshold=0.54,
                short_ma_period=12,
                long_ma_period=50,
                atr_period=14,
                stop_loss_atr_multiplier=1.8,
                take_profit_atr_multiplier=3.0,
                save_results=False
            )
            
            # Evaluate strategy
            metrics = evaluate_strategy(strategy_results)
            
            print(f"\nMethod: {method.upper()}")
            for metric, value in metrics.items():
                print(f"{metric}: {value}")
            
            # Check if this is the best method
            if metrics['sharpe_ratio'] > best_sharpe:
                best_sharpe = metrics['sharpe_ratio']
                best_method = method
                best_results = strategy_results
        
        # Plot results for the best method
        if best_results is not None:
            print(f"\nBest Hurst Method for MSFT: {best_method.upper()} (Sharpe: {best_sharpe:.4f})")
            fig = plot_strategy_results(best_results, title=f"Hurst Strategy on MSFT (Method: {best_method.upper()})")
            plt.savefig(f'test_results/hurst_strategy_msft_{best_method}.png')
            plt.close(fig)
    
    def test_mean_reversion_strategy(self):
        """Test mean reversion strategy on utilities ETF (XLU) with different Hurst methods"""
        # Define Hurst calculation methods to test
        hurst_methods = ['rs', 'dfa', 'aggvar']
        
        best_method = None
        best_sharpe = -float('inf')
        best_results = None
        
        print("\nMean Reversion Strategy Performance on XLU with Different Hurst Methods:")
        
        for method in hurst_methods:
            # Run mean reversion strategy with current method
            strategy_results = mean_reversion_strategy(
                self.mean_reverting_df,
                window_size=100,
                method=method,  # Use current method in iteration
                max_lag=20,
                hurst_threshold=0.38,
                short_ma_period=12,
                long_ma_period=50,
                atr_period=14,
                stop_loss_atr_multiplier=1.8,
                take_profit_atr_multiplier=3.0,
                save_results=False
            )
            
            # Evaluate strategy
            metrics = evaluate_strategy(strategy_results)
            
            print(f"\nMethod: {method.upper()}")
            for metric, value in metrics.items():
                print(f"{metric}: {value}")
            
            # Check if this is the best method
            if metrics['sharpe_ratio'] > best_sharpe:
                best_sharpe = metrics['sharpe_ratio']
                best_method = method
                best_results = strategy_results
        
        # Plot results for the best method
        if best_results is not None:
            print(f"\nBest Hurst Method for XLU Mean Reversion: {best_method.upper()} (Sharpe: {best_sharpe:.4f})")
            fig = plot_strategy_results(best_results, title=f"Mean Reversion Strategy on XLU (Method: {best_method.upper()})")
            plt.savefig(f'test_results/mean_reversion_xlu_{best_method}.png')
            plt.close(fig)
    
    def test_trend_following_strategy(self):
        """Test trend following strategy on crypto (BTC-USD) with different Hurst methods"""
        # Define Hurst calculation methods to test
        hurst_methods = ['rs', 'dfa', 'aggvar']
        
        best_method = None
        best_sharpe = -float('inf')
        best_results = None
        
        print("\nTrend Following Strategy Performance on BTC-USD with Different Hurst Methods:")
        
        for method in hurst_methods:
            # Run trend following strategy with current method
            strategy_results = trend_following_strategy(
                self.random_df,
                window_size=100,
                method=method,  # Use current method in iteration
                max_lag=20,
                hurst_threshold=0.54,
                short_ma_period=12,
                long_ma_period=50,
                atr_period=14,
                stop_loss_atr_multiplier=1.8,
                take_profit_atr_multiplier=3.0,
                save_results=False
            )
            
            # Evaluate strategy
            metrics = evaluate_strategy(strategy_results)
            
            print(f"\nMethod: {method.upper()}")
            for metric, value in metrics.items():
                print(f"{metric}: {value}")
            
            # Check if this is the best method
            if metrics['sharpe_ratio'] > best_sharpe:
                best_sharpe = metrics['sharpe_ratio']
                best_method = method
                best_results = strategy_results
        
        # Plot results for the best method
        if best_results is not None:
            print(f"\nBest Hurst Method for BTC-USD Trend Following: {best_method.upper()} (Sharpe: {best_sharpe:.4f})")
            fig = plot_strategy_results(best_results, title=f"Trend Following Strategy on BTC-USD (Method: {best_method.upper()})")
            plt.savefig(f'test_results/trend_following_btc_{best_method}.png')
            plt.close(fig)
    
    def test_parameter_optimization(self):
        """Test parameter optimization on Gold futures with different Hurst methods"""
        # Define parameter ranges
        window_sizes = [60, 100, 150]
        mean_reversion_thresholds = [0.35, 0.38, 0.42]
        trend_thresholds = [0.52, 0.54, 0.58]
        hurst_methods = ['rs', 'dfa', 'aggvar']
        
        best_params = None
        best_sharpe = -float('inf')
        
        print("\nParameter Optimization Results for Gold Futures with Different Hurst Methods:")
        
        # Test on Gold futures data
        for method in hurst_methods:
            print(f"\nTesting Hurst method: {method.upper()}")
            
            for window_size in window_sizes:
                for mr_threshold in mean_reversion_thresholds:
                    for tr_threshold in trend_thresholds:
                        # Skip invalid combinations
                        if mr_threshold >= tr_threshold:
                            continue
                            
                        # Run strategy
                        strategy_results = hurst_trading_strategy(
                            self.commodity_df,
                            window_size=window_size,
                            method=method,  # Use current method in iteration
                            max_lag=20,
                            mean_reversion_threshold=mr_threshold,
                            trend_threshold=tr_threshold,
                            short_ma_period=12,
                            long_ma_period=50,
                            atr_period=14,
                            stop_loss_atr_multiplier=1.8,
                            take_profit_atr_multiplier=3.0,
                            save_results=False
                        )
                        
                        # Evaluate strategy
                        metrics = evaluate_strategy(strategy_results)
                        
                        # Check if this is the best set of parameters
                        if metrics['sharpe_ratio'] > best_sharpe:
                            best_sharpe = metrics['sharpe_ratio']
                            best_params = {
                                'method': method,
                                'window_size': window_size,
                                'mean_reversion_threshold': mr_threshold,
                                'trend_threshold': tr_threshold,
                                'metrics': metrics
                            }
        
        # Print best parameters
        print("\nBest Parameters for Gold Futures:")
        print(f"Hurst Method: {best_params['method'].upper()}")
        print(f"Window Size: {best_params['window_size']}")
        print(f"Mean Reversion Threshold: {best_params['mean_reversion_threshold']}")
        print(f"Trend Threshold: {best_params['trend_threshold']}")
        print(f"Sharpe Ratio: {best_params['metrics']['sharpe_ratio']:.4f}")
        print(f"Total Return: {best_params['metrics']['total_return']:.2f}%")
        print(f"Max Drawdown: {best_params['metrics']['max_drawdown']:.2f}%")
    
    def test_forex_strategy(self):
        """Test strategy on forex data (EUR/USD) with different Hurst methods"""
        # Define Hurst calculation methods to test
        hurst_methods = ['rs', 'dfa', 'aggvar']
        
        best_method = None
        best_sharpe = -float('inf')
        best_results = None
        
        print("\nStrategy Performance on EUR/USD with Different Hurst Methods:")
        
        for method in hurst_methods:
            # Run strategy on forex data with current method
            strategy_results = hurst_trading_strategy(
                self.forex_df,
                window_size=100,
                method=method,  # Use current method in iteration
                max_lag=20,
                mean_reversion_threshold=0.38,
                trend_threshold=0.54,
                short_ma_period=12,
                long_ma_period=50,
                atr_period=14,
                stop_loss_atr_multiplier=1.8,
                take_profit_atr_multiplier=3.0,
                save_results=False
            )
            
            # Evaluate strategy
            metrics = evaluate_strategy(strategy_results)
            
            print(f"\nMethod: {method.upper()}")
            for metric, value in metrics.items():
                print(f"{metric}: {value}")
            
            # Check if this is the best method
            if metrics['sharpe_ratio'] > best_sharpe:
                best_sharpe = metrics['sharpe_ratio']
                best_method = method
                best_results = strategy_results
        
        # Plot results for the best method
        if best_results is not None:
            print(f"\nBest Hurst Method for EUR/USD: {best_method.upper()} (Sharpe: {best_sharpe:.4f})")
            fig = plot_strategy_results(best_results, title=f"Hurst Strategy on EUR/USD (Method: {best_method.upper()})")
            plt.savefig(f'test_results/hurst_strategy_eurusd_{best_method}.png')
            plt.close(fig)

if __name__ == "__main__":
    unittest.main()