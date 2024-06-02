import os
import pandas as pd
from utils.config import *


def us_equity_symbol_load():
  us_symbol_f = us_symbol_file()
  us_all_companies = pd.read_csv(us_symbol_f)
  return us_all_companies.loc[:, 'symbol']

def us_equity_daily_data_load(symbols = ['AAPL'], start_date = '2023-05-29', end_date = '2024-05-29', option = all):
  data = {}
  for symbol in symbols:
    try:
      # print(f"loading {symbol} trade data...")

      equity_folder = us_equity_folder(symbol = symbol)
      equity_file = os.path.join(equity_folder, 'daily.csv')
      data_tmp = pd.read_csv(equity_file)
      data_tmp = data_tmp[(data_tmp['date'] >= start_date) & (data_tmp['date'] <= end_date)]
      data_tmp.set_index('date', inplace=True)
      if(option == all):
        data[symbol] = data_tmp
      else:
        data[symbol] = data_tmp.loc[:, option]
    except:
      print(f"failed to load equity {symbol}")
  return data