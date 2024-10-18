
import json
import os
from pathlib import Path
import pandas as pd
import yfinance as yf
from quant_free.utils.us_equity_utils import *

_this_dir = Path(__file__).parent.parent

def us_equity_data_load(symbol = 'AAPL', dir_option = '', flie_name = 'daily.csv'):
  equity_folder = us_equity_folder(symbol = symbol)
  if dir_option == '':
    equity_file = os.path.join(equity_folder, flie_name)
  else:
    equity_file = os.path.join(equity_folder, dir_option, flie_name)
  data = pd.read_csv(equity_file)
  if 'timestamp' in data.columns:
    data.rename(columns={'timestamp': 'date'}, inplace=True)
  data.set_index('date', inplace=True)
  data = data.sort_index()
  return data

def us_equity_get_trade_dates():
  df = us_equity_data_load()
  return df.index

def us_equity_get_trade_date_within_range(symbol = "AAPL", start_date = '2023-05-29', end_date = '2024-05-29', dir_option = ''):

    # Download historical stock data
  stock_data = us_equity_data_load(symbol = symbol, dir_option = dir_option)

  filtered_data = stock_data.loc[start_date:end_date]
  
  trade_dates = filtered_data.index

  trade_dates_time = pd.to_datetime(pd.to_datetime(trade_dates).date)

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
