import os
import pandas as pd
from utils.config import *
from common.us_equity_common import *


def us_equity_symbol_load():
  us_symbol_f = us_symbol_file()
  us_all_companies = pd.read_csv(us_symbol_f)
  return us_all_companies.loc[:, 'symbol']

def us_equity_daily_data_load(symbols = ['AAPL'], start_date = '2023-05-29', end_date = '2024-05-29', option = all):
  data = {}
  trade_date_len = len(us_equity_get_trade_date_within_range(start_date = start_date, end_date = end_date))
  for symbol in symbols:
    try:
      # print(f"loading {symbol} trade data...")

      equity_folder = us_equity_folder(symbol = symbol)
      equity_file = os.path.join(equity_folder, 'daily.csv')
      data_tmp = pd.read_csv(equity_file)
      data_tmp = data_tmp[(data_tmp['date'] >= start_date) & (data_tmp['date'] <= end_date)]
      rows, columns = data_tmp.shape
      
      if rows == trade_date_len:
        data_tmp.set_index('date', inplace=True)
        if(option == all):
          data[symbol] = data_tmp
        else:
          data[symbol] = data_tmp.loc[:, option]
      else:
        print("lack of some trade date skip ", symbol)
    except:
      print(f"failed to load equity {symbol}")
  return data

def us_equity_common_shares_load(symbols = 'AAPL'):
  data = {}
  for symbol in symbols:
    try:
      # print(f"loading {symbol} trade data...")
      equity_folder = us_equity_sub_folder(symbol = symbol, sub_dir = 'efinance')
      equity_file = os.path.join(equity_folder, 'info.csv')
      data = pd.read_csv(equity_file)
      return data.loc[:, 'issued_common_shares'][0]
    except:
      print(f"failed to load equity {symbol}")
  return data


def us_equity_efinance_finance_data_load(symbol = 'AAPL'):

  try:
    # efinance_symbol = datacenter.get_secucode("MMM")
    print(f"Downloading {symbol} finance data...")
    income = us_equity_efinance_load_csv(symbol, 'income')
    cash = us_equity_efinance_load_csv(symbol, 'cash')
    balance = us_equity_efinance_load_csv(symbol, 'balance')

    data = pd.concat([income, cash, balance], axis = 1)
    return data
    # us_equity_efinance_finance_store_csv(equity_folder, data, 'metrics')
  except:
    print(f"failed to download equity {symbol}")