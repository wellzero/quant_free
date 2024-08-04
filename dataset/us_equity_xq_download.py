import os
import pandas as pd
from utils.us_equity_symbol import *
from utils.us_equity_utils import *
from common.us_equity_common import *
import efinance as ef

# Read directory from JSON file

def us_equity_xq_finance_store_csv(equity_folder, data, file_name):
    file = os.path.join(equity_folder, file_name + '.csv')
    # if os.path.exists(file):
      # data_local = pd.read_csv(file)

      # # Drop the 'Unnamed: 0.1' column if it exists
      # if 'Unnamed: 0.1' in data_local.columns:
      #     data_local.drop(columns=['Unnamed: 0.1'], inplace=True)

      # # Drop the 'Unnamed: 0' column if it exists
      # if 'Unnamed: 0' in data_local.columns:
      #     data_local.drop(columns=['Unnamed: 0'], inplace=True)

      # merged_data = pd.concat([data_local, data])
      # if 'REPORT_DATE' in data_local.columns:
      #   merged_data.drop_duplicates(subset='REPORT_DATE', inplace=True)
      # data = merged_data
    data.to_csv(file, index=False)
    print("store file: ", file)


def us_equity_xq_finance_data_download(symbols = ['AAPL'], provider="xq"):

  datacenter = ef.stock.us_finance_xq_getter()

  for symbol in symbols:
    try:
      # xq_symbol = datacenter.get_secucode("MMM")
      print(f"Downloading {symbol} finance data...")
      equity_folder = us_equity_sub_folder(symbol = symbol, sub_dir = provider)

      data = datacenter.get_us_finance_income(symbol = symbol)
      us_equity_xq_finance_store_csv(equity_folder, data, 'income')

      data = datacenter.get_us_finance_cash(symbol = symbol)
      us_equity_xq_finance_store_csv(equity_folder, data, 'cash')

      data = datacenter.get_us_finance_balance(symbol = symbol)
      us_equity_xq_finance_store_csv(equity_folder, data, 'balance')

      data = datacenter.get_us_finance_main_factor(symbol = symbol)
      us_equity_xq_finance_store_csv(equity_folder, data, 'metrics')
    except:
      print(f"function {__name__} error!!")

def us_equity_xq_daily_data_download(symbols = ['AAPL'], provider="xq"):
  datacenter_xq = ef.stock.us_finance_xq_getter()
  for symbol in symbols:
    try:
      
      print(f"Downloading {symbol} trade data...")
      data = datacenter_xq.get_us_finance_daily_trade(symbol = symbol)

      equity_folder = us_equity_sub_folder(symbol = symbol, sub_dir = provider)
      equity_file = os.path.join(equity_folder, 'daily.csv')

      data.to_csv(equity_file, index=False)
    except:
      print(f"function {__name__} {us_equity_xq_daily_data_download} error!!")

def us_equity_xq_euquity_data_download():
  datacenter_xq = ef.stock.us_finance_xq_sector_getter()
  data = datacenter_xq.get_all_us_equity()
  us_dir1_store_csv(dir0 = 'symbol', dir1 = 'xq', filename='us_equity_symbol.csv', data = data)

  print(f"downloading us_china")
  data = datacenter_xq.get_all_us_us_china_equity()
  us_dir1_store_csv(dir0 = 'symbol', dir1 = 'xq', filename='us_china.csv', data = data)

  print(f"downloading listed")
  data = datacenter_xq.get_all_us_listed_equity()
  us_dir1_store_csv(dir0 = 'symbol', dir1 = 'xq', filename='listed.csv', data = data)

  print(f"downloading us_star")
  data = datacenter_xq.get_all_us_star_equity()
  us_dir1_store_csv(dir0 = 'symbol', dir1 = 'xq', filename='us_star.csv', data = data)

def us_equity_xq_sector_data_download():
  
  datacenter_xq = ef.stock.us_finance_xq_sector_getter()
  data = datacenter_xq.get_all_us_sector_name()
  us_dir1_store_csv(dir0 = 'symbol', dir1 = 'xq', filename='us_equity_sector.csv', data = data)

  for index, row in data.iterrows():
    print(f"downloading {row['name']}")
    data = datacenter_xq.get_all_us_equity(row['encode'])
    us_dir1_store_csv(dir0 = 'symbol', dir1 = 'xq', filename=row['name'] + '.csv', data = data)
