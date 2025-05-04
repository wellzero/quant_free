# 151 Trading Strategies Implementation

This project implements trading strategies from the book "151 Trading Strategies" by Zura Kakushadze and Juan A. Serur. The strategies are organized by asset class and strategy type, with a focus on practical implementation and backtesting.

## Overview

The implementation covers strategies across multiple asset classes:

- **Equity Strategies**: Momentum, mean reversion, pairs trading, and factor-based strategies
- **Fixed Income Strategies**: Yield curve, credit spread, and duration-based strategies
- **Forex Strategies**: Carry trade, momentum, and volatility strategies
- **Commodity Strategies**: Trend following and term structure strategies
- **Multi-Asset Strategies**: Risk parity, trend following, and regime-switching strategies

## Directory Structure

```
strategies/
├── equity/
│   ├── momentum.py
│   ├── mean_reversion.py
├── fixed_income/
│   ├── yield_curve.py
├── forex/
│   ├── carry_trade.py
├── commodity/
│   ├── trend_following.py
├── multi_asset/
│   ├── risk_parity.py
└── utils/
    ├── performance.py
    ├── risk.py
    └── optimization.py
```

## Implemented Strategies

### Equity Strategies

1. **Time Series Momentum (Strategy 1)**
   - Generates signals based on past returns over a lookback period
   - Includes volatility scaling for risk management

2. **Bollinger Band Mean Reversion (Strategy 7)**
   - Trades mean reversion using Bollinger Bands
   - Includes stop-loss and take-profit mechanisms

### Fixed Income Strategies

1. **Yield Curve Steepener/Flattener (Strategy 30)**
   - Trades based on the slope of the yield curve
   - Uses z-score to identify extreme yield curve positions

2. **Butterfly Trade (Strategy 31)**
   - Exploits non-parallel shifts in the yield curve
   - Trades the relationship between short, medium, and long-term rates

### Forex Strategies

1. **Currency Carry Trade (Strategy 59)**
   - Goes long high-yielding currencies and short low-yielding currencies
   - Includes interest rate differential component

2. **Interest Rate Differential Strategy (Strategy 64)**
   - Trades based on extreme z-scores in interest rate differentials
   - Mean-reversion approach to currency valuation

### Commodity Strategies

1. **Commodity Trend Following (Strategy 88)**
   - Uses multiple moving averages to identify trends
   - Includes volatility scaling for risk management

2. **Commodity Term Structure Strategy (Strategy 93)**
   - Exploits the relationship between front and back contracts
   - Trades contango and backwardation extremes

### Multi-Asset Strategies

1. **Risk Parity Portfolio (Strategy 101)**
   - Allocates risk equally across assets
   - Includes volatility targeting

## Usage

Each strategy is implemented as a function that takes price data and parameters as inputs and returns a DataFrame with strategy results. Here's an example of how to use a strategy:

```python
from strategies.equity.momentum import time_series_momentum
import yfinance as yf

# Download price data
data = yf.download('SPY', start='2018-01-01', end='2023-01-01')

# Run the strategy
results = time_series_momentum(data, lookback_period=252, holding_period=63)

# Analyze results
print(f"Total Return: {results['cumulative_returns'].iloc[-1]:.2f}")
```

## Testing

Use the `test_strategies.py` script to run tests for all implemented strategies:

```bash
python test_strategies.py
```

This will download sample data, run each strategy, and display performance metrics and plots.

## Implementation Plan

See the `IMPLEMENTATION_PLAN.md` file for details on the planned implementation of all 151 strategies.

## Requirements

The implementation requires the following Python packages:

- numpy
- pandas
- matplotlib
- scipy
- statsmodels
- yfinance

Install the requirements using:

```bash
pip install -r requirements.txt
```

## References

- Zura Kakushadze and Juan A. Serur. 151 Trading Strategies.
- Additional strategy-specific references are included in each implementation file.