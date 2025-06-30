
import json
import os

import pandas as pd
import yfinance as yf
from quant_free.utils.us_equity_utils import *
from quant_free.dataset.xq_daily_data import *
from quant_free.dataset.xq_symbol import *




def us_equity_get_trade_dates(market = 'us'):
  df = daily_trade_load(market)
  return df.index

def us_equity_get_trade_date_within_range(market = 'us', symbol = "AAPL", start_date = '2023-05-29', end_date = '2024-05-29', dir_option = ''):

  # Download historical stock data
  trade_dates_time = equity_tradedate_load_within_range(market, symbol, start_date, end_date, dir_option)

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
