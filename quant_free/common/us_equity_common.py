
import json
import os
from pathlib import Path
import pandas as pd
import yfinance as yf
from quant_free.utils.us_equity_utils import *
from quant_free.dataset.us_equity_load import *

_this_dir = Path(__file__).parent.parent


def us_equity_get_trade_dates():
  df = us_equity_data_load()
  return df.index

def us_equity_get_trade_date_within_range(symbol = "AAPL", start_date = '2023-05-29', end_date = '2024-05-29', dir_option = ''):

  # Download historical stock data
  trade_dates_time = us_equity_load_trade_date_within_range(symbol, start_date, end_date, dir_option)

  return trade_dates_time

def us_equity_get_current_trade_date(symbol = "AAPL"):
    # Download historical stock data
  stock_data = yf.download(symbol)

  # Extract the date part from the Timestamp index
  trade_dates = stock_data.index.date

  return trade_dates[-1].strftime('%Y-%m-%d')


if __name__ == "__main__":
  trade_dates = us_equity_get_trade_date_within_range()
  print(len(trade_dates))
