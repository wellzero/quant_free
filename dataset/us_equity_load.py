import os
import pandas as pd
from utils.us_equity_symbol import *
from utils.us_equity_utils import *
from common.us_equity_common import *


def us_equity_symbol_load():
  df = us_dir0_load_csv(dir0 = 'symbol', filename='us_equity_symbol.csv')
  return df.loc[:, 'symbol'].values

def us_equity_daily_data_load(symbols = ['AAPL'], start_date = '2023-05-29', end_date = '2024-05-29', trade_option = all, dir_option = ''):
  data = {}
  trade_date_len = len(us_equity_get_trade_date_within_range(start_date = start_date, end_date = end_date, dir_option = dir_option))
  for symbol in symbols:
    try:
      # print(f"loading {symbol} trade data...")

      data_tmp = us_equity_daily_data_read_csv(symbol, dir_option)
      data_tmp = data_tmp.loc[start_date:end_date]
      # data_tmp = data_tmp[(data_tmp['date'] >= start_date) & (data_tmp['date'] <= end_date)]
      rows, columns = data_tmp.shape
      
      if rows == trade_date_len:
        # data_tmp.set_index('date', inplace=True)
        if(trade_option == all):
          data[symbol] = data_tmp
        else:
          data[symbol] = data_tmp.loc[:, trade_option]
      else:
        print("lack of some trade date skip ", symbol)
    except:
      print(f"failed to load equity {symbol}")
  return data