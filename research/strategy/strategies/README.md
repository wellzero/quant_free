# Trading Strategies Implementation

This directory contains implementations of various trading strategies from the book "151 Trading Strategies". The strategies are organized by asset class and strategy type.

## Directory Structure

```
strategies/
├── README.md
├── equity/
│   ├── momentum.py
│   ├── mean_reversion.py
│   ├── pairs_trading.py
│   └── factor_based.py
├── fixed_income/
│   ├── yield_curve.py
│   ├── credit_spread.py
│   └── duration_based.py
├── forex/
│   ├── carry_trade.py
│   ├── momentum.py
│   └── volatility.py
├── multi_asset/
│   ├── risk_parity.py
│   ├── trend_following.py
│   └── regime_switching.py
└── utils/
    ├── performance.py
    ├── risk.py
    └── optimization.py
```

## Implementation Status

- [x] Equity Momentum Strategies
- [x] Equity Mean Reversion Strategies
- [x] Multi-Asset Risk Parity
- [ ] Fixed Income Yield Curve Strategies
- [ ] Forex Carry Trade Strategies
- [ ] More strategies to be added...

## Usage

Each strategy file contains one or more strategy implementations with proper documentation, parameter settings, and backtest functionality. To use a strategy, import it and pass the required data:

```python
from strategies.equity.momentum import time_series_momentum

# Load your price data into a DataFrame with 'Open', 'High', 'Low', 'Close' columns
results = time_series_momentum(prices_df, lookback_period=252, holding_period=63)
```

## References

- Zura Kakushadze and Juan A. Serur. 151 Trading Strategies.
- Additional strategy-specific references are included in each implementation file.