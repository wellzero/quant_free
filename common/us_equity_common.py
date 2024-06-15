
import json
import os
from pathlib import Path
import yfinance as yf

_this_dir = Path(__file__).parent.parent


def us_equity_get_trade_date_within_range(symbol = "AAPL", start_date = '2023-05-29', end_date = '2024-05-29'):

    # Download historical stock data
  stock_data = yf.download(symbol)

  filtered_data = stock_data.loc[start_date:end_date]
  
  trade_dates = filtered_data.index.date

  return trade_dates

def us_equity_get_current_trade_date(symbol = "AAPL"):
    # Download historical stock data
  stock_data = yf.download(symbol)

  # Extract the date part from the Timestamp index
  trade_dates = stock_data.index.date

  return trade_dates[-1].strftime('%Y-%m-%d')


if __name__ == "__main__":
  trade_dates = us_equity_get_trade_date_within_range()
  print(len(trade_dates))
