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
from hurst.core import hurst_rs, hurst_dfa, hurst_aggvar, hurst_higuchi
from hurst.evaluation import evaluate_strategy
from hurst.visualization import plot_strategy_results

class TestHurstStrategyWithYahoo(unittest.TestCase):
    """Test cases for Hurst trading strategy using Yahoo Finance data"""
    
    def setUp(self):
        """Set up test data from Yahoo Finance"""
        # Create directory for test results
        os.makedirs('test_results', exist_ok=True)

        self.mag_lag = 50
        self.window_size = 252
        
        # Define test period
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=365*3)  # 3 years of data

        # Define tickers to test
        # self.tickers = {'RTY=F', '^IXIC', 'BTC-USD', 'GC=F', 'EURUSD=X'}
        self.tickers = {'RTY=F'}
        
        # Dictionary to store dataframes for each ticker
        self.ticker_data = {}
        
        # Download data for each ticker
        for ticker in self.tickers:
            try:
                data = yf.download(ticker, 
                                  start=self.start_date.strftime('%Y-%m-%d'),
                                  end=self.end_date.strftime('%Y-%m-%d'))
                if not data.empty:
                    self.ticker_data[ticker] = data
                    print(f"Successfully downloaded data for {ticker}")
                else:
                    print(f"No data available for {ticker}")
            except Exception as e:
                print(f"Error downloading {ticker}: {str(e)}")
    
    def test_hurst_calculation_on_all_tickers(self):
        """Test Hurst exponent calculation on all tickers"""
        results = {}
        
        print("\nHurst Exponent Calculation on All Tickers:")
        
        for ticker, data in self.ticker_data.items():
            if len(data) < self.window_size:
                print(f"Warning: Not enough data for {ticker}, skipping")
                continue
                
            # Calculate Hurst using different methods
            price_series = data['Close'].values[-self.window_size:]
            
            try:
                rs_hurst = hurst_rs(price_series, max_lag=self.mag_lag)
                dfa_hurst = hurst_dfa(price_series)
                aggvar_hurst = hurst_aggvar(price_series)
                higuchi_hurst = hurst_higuchi(price_series)
                
                results[ticker] = {
                    'RS': rs_hurst,
                    'DFA': dfa_hurst,
                    'AggVar': aggvar_hurst,
                    'Higuchi': higuchi_hurst
                }
                
                print(f"{ticker} - RS: {rs_hurst:.3f}, DFA: {dfa_hurst:.3f}, AggVar: {aggvar_hurst:.3f}, Higuchi: {higuchi_hurst:.3f}")
            except Exception as e:
                print(f"Error calculating Hurst for {ticker}: {e}")
        
        return results
    
    def test_strategy_on_all_tickers(self):
        """Test strategy performance on all tickers with different Hurst methods"""
        # Define Hurst calculation methods to test
        hurst_methods = ['rs', 'dfa', 'aggvar', 'higuchi']
        
        # Store best results for each ticker
        best_results = {}
        
        print("\nStrategy Performance on All Tickers with Different Hurst Methods:")
        
        for ticker, data in self.ticker_data.items():
            print(f"\n=== Testing {ticker} ===")
            
            best_method = None
            best_sharpe = -float('inf')
            ticker_best_results = None
            
            for method in hurst_methods:
                print(f"\nMethod: {method.upper()}")
                
                # Run strategy on current ticker with current method
                try:
                    strategy_results = hurst_trading_strategy(
                        data,
                        window_size=self.window_size,
                        method=method,
                        max_lag=self.mag_lag,
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
                    
                    for metric, value in metrics.items():
                        print(f"{metric}: {value}")
                    
                    # Check if this is the best method for this ticker
                    if metrics['sharpe_ratio'] > best_sharpe:
                        best_sharpe = metrics['sharpe_ratio']
                        best_method = method
                        ticker_best_results = strategy_results
                    
                    # Plot results for each method
                    fig = plot_strategy_results(strategy_results, 
                                               title=f"Hurst Strategy on {ticker} (Method: {method.upper()})")
                    plt.savefig(f'test_results/hurst_strategy_{ticker.replace("=", "_").replace("^", "")}_{method}.png')
                    plt.close(fig)
                    
                except Exception as e:
                    print(f"Error running strategy on {ticker} with method {method}: {e}")
            
            # Store best results for this ticker
            if best_method is not None:
                best_results[ticker] = {
                    'method': best_method,
                    'sharpe': best_sharpe,
                    'results': ticker_best_results
                }
                print(f"\nBest Hurst Method for {ticker}: {best_method.upper()} (Sharpe: {best_sharpe:.4f})")
        
        # Print summary of best methods for all tickers
        print("\n=== Summary of Best Methods ===")
        for ticker, result in best_results.items():
            print(f"{ticker}: {result['method'].upper()} (Sharpe: {result['sharpe']:.4f})")
        
        return best_results
    

if __name__ == "__main__":
    # Create test instance
    test = TestHurstStrategyWithYahoo()
    
    # Setup test data
    test.setUp()
    
    # Run tests
    print("\n=== CALCULATING HURST EXPONENTS FOR ALL TICKERS ===")
    hurst_results = test.test_hurst_calculation_on_all_tickers()
    
    print("\n=== TESTING HURST STRATEGY ON ALL TICKERS ===")
    strategy_results = test.test_strategy_on_all_tickers()
