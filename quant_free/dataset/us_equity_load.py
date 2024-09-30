import os
import pandas as pd
from quant_free.utils.us_equity_symbol import *
from quant_free.utils.us_equity_utils import *
from quant_free.common.us_equity_common import *


def us_equity_symbol_load():
  df = us_dir0_load_csv(dir0 = 'symbol', filename='us_equity_symbol.csv')
  return df.loc[:, 'symbol'].values

def convert_to_string_if_number(value):
    if isinstance(value, (int, float, complex)):
        return str(value)
    return value

def us_equity_daily_data_load(symbols = ['AAPL'], start_date = '2023-05-29', end_date = '2024-05-29', trade_option = all, dir_option = ''):
  data = {}
  trade_date_time = us_equity_get_trade_date_within_range(start_date = start_date, end_date = end_date, dir_option = dir_option)

  # symbols = symbols.remove(0)

  for symbol in symbols:
    # print(symbol)
    equity_folder = us_equity_folder(symbol = convert_to_string_if_number(symbol))
    if dir_option == '':
      equity_file = os.path.join(equity_folder, 'daily.csv')
    else:
      equity_file = os.path.join(equity_folder, dir_option, 'daily.csv')
    
    if os.path.exists(equity_file):
      # try:
        # print(f"loading {symbol} trade data...")

        data_tmp = us_equity_daily_data_read_csv(symbol, dir_option)
        # data_tmp = data_tmp.loc[start_date:end_date]
        # data_tmp = data_tmp[(data_tmp['date'] >= start_date) & (data_tmp['date'] <= end_date)]
        rows, columns = data_tmp.shape

        data_tmp.index = pd.to_datetime(pd.to_datetime(data_tmp.index).date)

        if rows > 3:
          data_tmp = data_tmp.resample('B').asfreq()
          # Optionally fill missing values (for example, forward fill)
          df_filled = data_tmp.ffill()

          # fill begining
          df_filled = df_filled.reindex(trade_date_time, method = 'bfill')

          df_filled_select = df_filled.loc[trade_date_time,:]
          # df_filled_select.index = df_filled_select.index.strftime('%Y-%m-%d')

          if(trade_option == all):
            data[symbol] = df_filled_select
          else:
            data[symbol] = df_filled_select.loc[:, trade_option]
        else:
          print(f"lack of some trade date skip {symbol}, date_start {df_filled.head(1).index}")
      # except:
      #   print(f"lack of some trade date skip {symbol}, data row {data_tmp.shape[0]}, date_start {df_filled.index[0]}, end_start {df_filled.index[-1]}")
    else:
      print(f"file {equity_file} not exsist!!")
  return data

def us_equity_sector_daily_data_load(sector_name = '半导体产品与设备', start_date = '2023-05-29', end_date = '2024-05-29', trade_option = all, dir_option = 'xq'):

  symbols = us_dir1_load_csv(dir0 = 'symbol', dir1 = 'xq', filename= sector_name +'.csv')['symbol'].values

  return us_equity_daily_data_load(symbols = symbols, start_date = start_date, end_date = end_date, trade_option = trade_option, dir_option = dir_option)