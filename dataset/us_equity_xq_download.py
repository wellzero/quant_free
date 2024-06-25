import os
import pandas as pd
from utils.us_equity_symbol import *
from utils.us_equity_utils import *
from common.us_equity_common import *

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

  import efinance as ef
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

