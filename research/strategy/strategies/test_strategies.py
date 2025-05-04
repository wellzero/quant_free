import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import sys
import os

# Add strategy directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import strategies
from equity.momentum import time_series_momentum, plot_momentum_strategy
from equity.mean_reversion import bollinger_band_strategy, plot_mean_reversion_strategy
from fixed_income.yield_curve import yield_curve_steepener, butterfly_trade, plot_yield_curve_strategy
from forex.carry_trade import carry_trade_strategy, interest_rate_differential_strategy, plot_carry_trade_strategy
from commodity.trend_following import commodity_trend_following, commodity_term_structure, plot_commodity_strategy
from multi_asset.risk_parity import risk_parity_portfolio, plot_risk_parity_portfolio
from utils.performance import calculate_returns_metrics, plot_equity_curve

def download_sample_data(tickers, start_date='2018-01-01', end_date='2023-01-01'):
    """
    Download sample price data for testing strategies
    
    Parameters:
    tickers (list): List of ticker symbols
    start_date (str): Start date for data download
    end_date (str): End date for data download
    
    Returns:
    dict: Dictionary of DataFrames with price data for each ticker
    """
    data_dict = {}
    
    for ticker in tickers:
        try:
            # Download data
            data = yf.download(ticker, start=start_date, end=end_date)
            
            # Check if data is valid
            if len(data) > 0:
                data_dict[ticker] = data
            else:
                print(f"No data available for {ticker}")
        except Exception as e:
            print(f"Error downloading {ticker}: {str(e)}")
    
    return data_dict

def test_momentum_strategy():
    """
    Test time series momentum strategy
    """
    print("\n=== Testing Time Series Momentum Strategy ===")
    
    # Download sample data
    data_dict = download_sample_data(['SPY'])
    
    if 'SPY' not in data_dict:
        print("Failed to download SPY data")
        return
    
    # Run momentum strategy
    results = time_series_momentum(
        prices_df=data_dict['SPY'],
        lookback_period=252,  # 1 year
        holding_period=63,    # 3 months
        volatility_scaling=True,
        target_volatility=0.15
    )
    
    # Print performance metrics
    strategy_returns = results['strategy_returns'].dropna()
    metrics = calculate_returns_metrics(strategy_returns)
    
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annualized Return: {metrics['ann_return']:.2%}")
    print(f"Annualized Volatility: {metrics['ann_volatility']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    
    # Plot strategy performance
    fig = plot_momentum_strategy(results)
    plt.savefig('momentum_strategy.png')
    plt.close(fig)
    
    print("Strategy plot saved as 'momentum_strategy.png'")

def test_mean_reversion_strategy():
    """
    Test Bollinger Band mean reversion strategy
    """
    print("\n=== Testing Bollinger Band Mean Reversion Strategy ===")
    
    # Download sample data
    data_dict = download_sample_data(['QQQ'])
    
    if 'QQQ' not in data_dict:
        print("Failed to download QQQ data")
        return
    
    # Run mean reversion strategy
    results = bollinger_band_strategy(
        prices_df=data_dict['QQQ'],
        window=20,           # 20-day moving average
        num_std=2.0,         # 2 standard deviations
        holding_period=5,    # 5-day holding period
        partial_exit=True,   # Exit partially when price reverts to mean
        stop_loss_std=3.0    # Stop loss at 3 standard deviations
    )
    
    # Print performance metrics
    strategy_returns = results['strategy_returns'].dropna()
    metrics = calculate_returns_metrics(strategy_returns)
    
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annualized Return: {metrics['ann_return']:.2%}")
    print(f"Annualized Volatility: {metrics['ann_volatility']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    
    # Plot strategy performance
    fig = plot_mean_reversion_strategy(results)
    plt.savefig('mean_reversion_strategy.png')
    plt.close(fig)
    
    print("Strategy plot saved as 'mean_reversion_strategy.png'")

def test_risk_parity():
    """
    Test risk parity portfolio strategy
    """
    print("\n=== Testing Risk Parity Portfolio Strategy ===")
    
    # Download sample data for multiple assets
    tickers = ['SPY', 'TLT', 'GLD', 'IWM', 'EEM']
    data_dict = download_sample_data(tickers)
    
    if len(data_dict) < 3:
        print("Failed to download enough data for risk parity")
        return
    
    # Run risk parity strategy
    results = risk_parity_portfolio(
        prices_dict=data_dict,
        lookback_period=252,       # 1 year
        rebalance_frequency=21,    # Monthly rebalancing
        target_volatility=0.10     # 10% target volatility
    )
    
    # Print performance metrics
    strategy_returns = results['returns'].dropna()
    metrics = calculate_returns_metrics(strategy_returns)
    
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annualized Return: {metrics['ann_return']:.2%}")
    print(f"Annualized Volatility: {metrics['ann_volatility']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    
    # Print final portfolio weights
    final_weights = results['weights'].iloc[-1]
    print("\nFinal Portfolio Weights:")
    for asset, weight in final_weights.items():
        print(f"{asset}: {weight:.2%}")
    
    # Plot strategy performance
    fig = plot_risk_parity_portfolio(results)
    plt.savefig('risk_parity_strategy.png')
    plt.close(fig)
    
    print("Strategy plot saved as 'risk_parity_strategy.png'")

def test_yield_curve_strategy():
    """
    Test yield curve strategies
    """
    print("\n=== Testing Yield Curve Strategies ===")
    
    # Download yield data (using Treasury yield data as example)
    print("Downloading yield data...")
    try:
        # Use common Treasury ETFs as proxies for different points on the yield curve
        # SHY - 1-3 Year Treasury Bond ETF
        # IEI - 3-7 Year Treasury Bond ETF
        # IEF - 7-10 Year Treasury Bond ETF
        short_etf = yf.download('SHY', start='2018-01-01', end='2023-01-01')
        mid_etf = yf.download('IEI', start='2018-01-01', end='2023-01-01')
        long_etf = yf.download('IEF', start='2018-01-01', end='2023-01-01')
        
        # Convert prices to yields (simplified approximation)
        # In practice, would use actual yield data
        short_yield = 100 / short_etf['Close']
        mid_yield = 100 / mid_etf['Close']
        long_yield = 100 / long_etf['Close']
        
        # Test yield curve steepener strategy
        results_steepener = yield_curve_steepener(short_yield, long_yield)
        print("\nYield Curve Steepener Strategy Results:")
        metrics = calculate_returns_metrics(results_steepener['returns'].dropna())
        for key in ['total_return', 'ann_return', 'sharpe_ratio', 'max_drawdown']:
            print(f"{key}: {metrics[key]:.4f}")
        
        # Test butterfly trade strategy
        results_butterfly = butterfly_trade(short_yield, mid_yield, long_yield)
        print("\nButterfly Trade Strategy Results:")
        metrics = calculate_returns_metrics(results_butterfly['returns'].dropna())
        for key in ['total_return', 'ann_return', 'sharpe_ratio', 'max_drawdown']:
            print(f"{key}: {metrics[key]:.4f}")
            
        # Plot results
        plot_yield_curve_strategy(results_steepener, title="Yield Curve Steepener Strategy")
        plot_yield_curve_strategy(results_butterfly, title="Butterfly Trade Strategy")
        
    except Exception as e:
        print(f"Error testing yield curve strategies: {str(e)}")

def test_carry_trade_strategy():
    """
    Test forex carry trade strategies
    """
    print("\n=== Testing Forex Carry Trade Strategies ===")
    
    # Download forex data
    print("Downloading forex data...")
    try:
        # Use currency ETFs as proxies
        # FXA - Australian Dollar
        # FXB - British Pound
        # FXC - Canadian Dollar
        # FXE - Euro
        # FXY - Japanese Yen
        currencies = ['FXA', 'FXB', 'FXC', 'FXE', 'FXY', 'UUP']
        forex_data = download_sample_data(currencies, start_date='2018-01-01', end_date='2023-01-01')
        
        # Create a simplified interest rate DataFrame (for demonstration)
        # In practice, would use actual interest rate data
        dates = forex_data['FXA'].index
        interest_rates = pd.DataFrame(index=dates)
        
        # Approximate interest rates based on historical averages
        interest_rates['FXA'] = 0.02  # AUD
        interest_rates['FXB'] = 0.01  # GBP
        interest_rates['FXC'] = 0.015  # CAD
        interest_rates['FXE'] = 0.0   # EUR
        interest_rates['FXY'] = -0.001  # JPY
        interest_rates['UUP'] = 0.025  # USD
        
        # Add some time variation
        for i, currency in enumerate(['FXA', 'FXB', 'FXC', 'FXE', 'FXY']):
            trend = np.linspace(-0.01, 0.01, len(dates))
            noise = np.random.normal(0, 0.002, len(dates))
            interest_rates[currency] += trend + noise
        
        # Create exchange rates DataFrame
        exchange_rates = pd.DataFrame(index=dates)
        for currency in currencies:
            exchange_rates[currency] = forex_data[currency]['Close']
        
        # Test carry trade strategy
        results_carry = carry_trade_strategy(exchange_rates, interest_rates)
        print("\nCarry Trade Strategy Results:")
        metrics = calculate_returns_metrics(results_carry['portfolio_returns'].dropna())
        for key in ['total_return', 'ann_return', 'sharpe_ratio', 'max_drawdown']:
            print(f"{key}: {metrics[key]:.4f}")
        
        # Test interest rate differential strategy
        results_ird = interest_rate_differential_strategy(exchange_rates, interest_rates)
        print("\nInterest Rate Differential Strategy Results:")
        metrics = calculate_returns_metrics(results_ird['portfolio_returns'].dropna())
        for key in ['total_return', 'ann_return', 'sharpe_ratio', 'max_drawdown']:
            print(f"{key}: {metrics[key]:.4f}")
            
        # Plot results
        plot_carry_trade_strategy(results_carry)
        
    except Exception as e:
        print(f"Error testing carry trade strategies: {str(e)}")

def test_commodity_strategies():
    """
    Test commodity trading strategies
    """
    print("\n=== Testing Commodity Trading Strategies ===")
    
    # Download commodity data
    print("Downloading commodity data...")
    try:
        # Use commodity ETFs as proxies
        # USO - Oil
        # GLD - Gold
        # SLV - Silver
        # JJC - Copper
        # CORN - Corn
        commodities = ['USO', 'GLD', 'SLV', 'JJC', 'CORN']
        commodity_data = download_sample_data(commodities, start_date='2018-01-01', end_date='2023-01-01')
        
        # Test trend following strategy
        results_trend = commodity_trend_following(commodity_data['GLD'])
        print("\nCommodity Trend Following Strategy Results (Gold):")
        metrics = calculate_returns_metrics(results_trend['returns'].dropna())
        for key in ['total_return', 'ann_return', 'sharpe_ratio', 'max_drawdown']:
            print(f"{key}: {metrics[key]:.4f}")
        
        # Test term structure strategy
        # For demonstration, we'll use two ETFs as proxies for different contract months
        # In practice, would use actual futures contract data
        if 'USO' in commodity_data and 'BNO' in commodity_data:
            # USO tracks front-month oil futures
            # BNO tracks longer-dated oil futures
            front_contract = commodity_data['USO']['Close']
            back_contract = commodity_data['BNO']['Close']
            
            results_term = commodity_term_structure(front_contract, back_contract)
            print("\nCommodity Term Structure Strategy Results (Oil):")
            metrics = calculate_returns_metrics(results_term['returns'].dropna())
            for key in ['total_return', 'ann_return', 'sharpe_ratio', 'max_drawdown']:
                print(f"{key}: {metrics[key]:.4f}")
                
            # Plot results
            plot_commodity_strategy(results_term, strategy_type="term_structure")
        else:
            print("BNO data not available for term structure strategy test")
            
        # Plot trend following results
        plot_commodity_strategy(results_trend, strategy_type="trend")
        
    except Exception as e:
        print(f"Error testing commodity strategies: {str(e)}")

if __name__ == "__main__":
    # Run tests
    test_momentum_strategy()
    test_mean_reversion_strategy()
    test_risk_parity()
    test_yield_curve_strategy()
    test_carry_trade_strategy()
    test_commodity_strategies()