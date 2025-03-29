
import json
import os
from pathlib import Path
import pandas as pd
import yfinance as yf
from quant_free.utils.us_equity_utils import *
from quant_free.dataset.us_equity_load import *

_this_dir = Path(__file__).parent.parent


def us_equity_get_trade_dates(market = 'us'):
  df = us_equity_data_load(market)
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

def us_equity_get_sector(symbol = "AAPL", dir_option = "xq"):
    
    sector_file = 'equity_sector.csv'
    df_sector = us_dir1_load_csv(dir0 = 'symbol', dir1 = dir_option, filename = sector_file)

    if dir_option == "xq":
      sectors = list(df_sector['name'].values)
    else:
      sectors = list(df_sector['Sector'].values)

    for sector in sectors:
      data_symbols = us_dir1_load_csv(dir0 = 'symbol', dir1 = dir_option, filename= sector +'.csv')
      if (data_symbols.empty == False):
        symbols = data_symbols['symbol'].values
        if symbol in symbols:
          return sector

    return None


if __name__ == "__main__":
  trade_dates = us_equity_get_trade_date_within_range()
  print(len(trade_dates))
