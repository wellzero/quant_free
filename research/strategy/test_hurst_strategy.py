import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from hurst_exponent import hurst_trading_strategy, evaluate_strategy

def test_strategy_parameters(ticker='SPY', start_date='2010-01-01', end_date='2023-01-01'):
    """
    Test different parameter combinations for the Hurst trading strategy
    
    Parameters:
    ticker (str): Ticker symbol
    start_date (str): Start date for backtest
    end_date (str): End date for backtest
    
    Returns:
    dict: Best parameter combinations
    """
    # Download data
    data = yf.download(ticker, start=start_date, end=end_date)
    prices = data['Close']
    
    # Define parameter ranges to test
    methods = ['rs', 'dfa', 'aggvar']
    window_sizes = [60, 90, 120, 150]
    mean_reversion_thresholds = [0.40, 0.42, 0.45]
    trend_thresholds = [0.55, 0.58, 0.60]
    
    # Store results
    results = {}
    
    # Test combinations
    for method in methods:
        for window_size in window_sizes:
            for mr_threshold in mean_reversion_thresholds:
                for tr_threshold in trend_thresholds:
                    print(f"Testing method={method}, window={window_size}, mr={mr_threshold}, tr={tr_threshold}")
                    
                    try:
                        strategy_df = hurst_trading_strategy(
                            prices, 
                            window_size=window_size, 
                            method=method,
                            mean_reversion_threshold=mr_threshold,
                            trend_threshold=tr_threshold
                        )
                        
                        performance = evaluate_strategy(strategy_df)
                        results[(method, window_size, mr_threshold, tr_threshold)] = performance
                        
                        print(f"Total Return: {performance['Total Return']:.2%}")
                        print(f"Sharpe Ratio: {performance['Sharpe Ratio']:.4f}")
                        print(f"Max Drawdown: {performance['Max Drawdown']:.2%}")
                        print("-" * 40)
                    except Exception as e:
                        print(f"Error with parameters: {e}")
    
    # Find best performing combinations
    if results:
        best_sharpe = max(results.items(), key=lambda x: x[1]['Sharpe Ratio'])
        best_return = max(results.items(), key=lambda x: x[1]['Total Return'])
        best_drawdown = max(results.items(), key=lambda x: -abs(x[1]['Max Drawdown']))
        
        print("\nBest combination by Sharpe Ratio:")
        print(f"Method: {best_sharpe[0][0]}, Window: {best_sharpe[0][1]}, MR: {best_sharpe[0][2]}, TR: {best_sharpe[0][3]}")
        print(f"Sharpe Ratio: {best_sharpe[1]['Sharpe Ratio']:.4f}")
        print(f"Total Return: {best_sharpe[1]['Total Return']:.2%}")
        
        print("\nBest combination by Total Return:")
        print(f"Method: {best_return[0][0]}, Window: {best_return[0][1]}, MR: {best_return[0][2]}, TR: {best_return[0][3]}")
        print(f"Total Return: {best_return[1]['Total Return']:.2%}")
        print(f"Sharpe Ratio: {best_return[1]['Sharpe Ratio']:.4f}")
        
        print("\nBest combination by Max Drawdown:")
        print(f"Method: {best_drawdown[0][0]}, Window: {best_drawdown[0][1]}, MR: {best_drawdown[0][2]}, TR: {best_drawdown[0][3]}")
        print(f"Max Drawdown: {best_drawdown[1]['Max Drawdown']:.2%}")
        print(f"Sharpe Ratio: {best_drawdown[1]['Sharpe Ratio']:.4f}")
        
        return {
            'best_sharpe': best_sharpe,
            'best_return': best_return,
            'best_drawdown': best_drawdown
        }
    
    return None

if __name__ == "__main__":
    # Test on multiple tickers
    tickers = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'AMZN']
    
    for ticker in tickers:
        print(f"\n\nTesting {ticker}:")
        print("=" * 50)
        best_params = test_strategy_parameters(ticker=ticker)
        
        if best_params:
            # Use the best Sharpe ratio parameters
            best_combo = best_params['best_sharpe'][0]
            
            # Download data again
            data = yf.download(ticker, start='2010-01-01', end='2023-01-01')
            prices = data['Close']
            
            # Run the strategy with best parameters
            strategy_df = hurst_trading_strategy(
                prices, 
                window_size=best_combo[1], 
                method=best_combo[0],
                mean_reversion_threshold=best_combo[2],
                trend_threshold=best_combo[3]
            )
            
            # Plot results
            plt.figure(figsize=(12, 6))
            strategy_df['cum_returns'].plot(label='Buy & Hold')
            strategy_df['cum_strategy_returns'].plot(label='Hurst Strategy')
            plt.title(f'{ticker} - Optimized Hurst Trading Strategy vs Buy & Hold')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Returns')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'hurst_strategy_{ticker}.png')
            plt.close()