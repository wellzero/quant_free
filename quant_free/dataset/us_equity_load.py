import os
import pandas as pd
from quant_free.utils.us_equity_symbol import *
from quant_free.utils.us_equity_utils import *
from quant_free.common.us_equity_common import *

def us_equity_data_load(
    market = 'us', 
    symbol = 'AAPL',
    dir_option = '',
    flie_name = 'daily.csv',
    interval = 'day'):
  
  equity_folder = us_equity_folder(market, symbol = symbol)
  if dir_option == '':
    equity_file = os.path.join(equity_folder, flie_name)
  else:
    equity_file = os.path.join(equity_folder, dir_option, flie_name)
  data = pd.read_csv(equity_file)

  if 'timestamp' in data.columns:
    data.rename(columns={'timestamp': 'date'}, inplace=True)
  elif 'Unnamed: 0' in data.columns and 'Unnamed: 1' in data.columns :
    # Assuming df is your DataFrame
    data.rename(columns = {'Unnamed: 0' : 'symbol', 'Unnamed: 1' : 'date'}, inplace=True)

  data.set_index('date', inplace=True)
  data = data.sort_index()

  if interval == 'day':
    data.index = pd.to_datetime(pd.to_datetime(data.index).date)
  else:
    data.index = pd.to_datetime(data.index)

  return data


def equity_tradedate_load_within_range(
    market = 'us', 
    symbol = "AAPL",
    start_date = '2023-05-29',
    end_date = '2024-05-29',
    dir_option = ''):

    # Download historical stock data
  stock_data = us_equity_data_load(market, symbol = symbol, dir_option = dir_option)

  if isinstance(start_date, str):
    start_date = pd.to_datetime(start_date)
  if isinstance(end_date, str):
    end_date = pd.to_datetime(end_date)

  filtered_data = stock_data.loc[start_date:end_date]
  trade_dates_time = filtered_data.index

  return trade_dates_time

def us_equity_symbol_load(market = 'us'):
  df = us_dir1_load_csv(market, dir0 = 'symbol',  dir1 = 'xq', filename='equity_symbol.csv', dtype = str)
  return df.loc[:, 'symbol'].values

def convert_to_string_if_number(value):
    if isinstance(value, (int, float, complex)):
        return str(value)
    return value

def equity_daily_data_load_within_range(
    market = 'us', 
    symbols = ['AAPL'],
    start_date = '2023-05-29',
    end_date = '2024-05-29',
    column_option = "all",
    dir_option = 'xq',
    file_name = 'daily.csv'):
  
  data = {}
  trade_date_time = equity_tradedate_load_within_range(
    market,
    start_date = start_date,
    end_date = end_date,
    dir_option = 'xq')

  # symbols = symbols.remove(0)

  lack_list = []
  for symbol in symbols:
    # print(symbol)
    equity_folder = us_equity_folder(market, symbol = convert_to_string_if_number(symbol))
    if dir_option == '':
      equity_file = os.path.join(equity_folder, file_name)
    else:
      equity_file = os.path.join(equity_folder, dir_option, file_name)

    
    if os.path.exists(equity_file):
      # try:
        # print(f"loading {symbol} trade data...")

        data_tmp = us_equity_data_load(market, symbol, dir_option, file_name)
        # data_tmp = data_tmp.loc[start_date:end_date]
        # data_tmp = data_tmp[(data_tmp['date'] >= start_date) & (data_tmp['date'] <= end_date)]
        rows, columns = data_tmp.shape

        data_tmp.index = pd.to_datetime(pd.to_datetime(data_tmp.index).date)

        if rows > 3:
          data_tmp = data_tmp.resample('B').asfreq()
          # Optionally fill missing values (for example, forward fill)
          pd.set_option('future.no_silent_downcasting', True)
          df_filled = data_tmp.ffill()

          # fill begining
          df_filled = df_filled.reindex(trade_date_time, method = 'bfill')
          # df_filled = df_filled.reindex(trade_date_time, fill_value=0)

          df_filled_select = df_filled.loc[trade_date_time,:]

          # fill end
          df_filled_select = df_filled_select.bfill()
          df_filled_select = df_filled_select.ffill()

          if(column_option == "all"):
            data[symbol] = df_filled_select
          else:
            data[symbol] = df_filled_select.loc[:, column_option]
        else:
          print(f"lack of some trade date skip {symbol}, date_start {df_filled.head(1).index}")
      # except:
      #   print(f"lack of some trade date skip {symbol}, data row {data_tmp.shape[0]}, date_start {df_filled.index[0]}, end_start {df_filled.index[-1]}")
    else:
      lack_list.append(symbol)

  if len(lack_list) > 0:
    print(f"miss these daily trade files {lack_list}")
  return data

# load data fro a sector
def us_equity_sector_daily_data_load(
    market = 'us',
    sector_name = 'Semiconductor Products and Equipment',
    start_date = '2023-05-29',
    end_date = '2024-05-29',
    column_option = 'all',
    dir_option = 'xq'):

  symbols = us_dir1_load_csv(market,
    dir0 = 'symbol', dir1 = 'xq',
    filename= sector_name +'.csv')['symbol'].values

  return equity_daily_data_load_within_range(
    market,
    symbols = symbols,
    start_date = start_date,
    end_date = end_date,
    column_option = column_option,
    dir_option = dir_option)

# multi index load [date ticker]
def us_quity_multi_index_data_load(
    market = 'us',
    sector_name = '半导体产品与设备', 
    start_date = '2023-05-29',
    end_date = '2024-05-29', 
    dir_option = 'xq',
    file_name = 'daily.csv'):

  """Converts a dictionary of DataFrames to a single MultiIndex DataFrame.

  Args:
      dict_df: A dictionary where keys represent the second level index 
                and values are the corresponding DataFrames.

  Returns:
      A pandas DataFrame with a MultiIndex. 
      Returns an empty DataFrame if the input dictionary is empty.
  """

  df_symbol = us_dir1_load_csv(
    dir0 = 'symbol',
    dir1 = dir_option,
    filename= sector_name +'.csv')

  if df_symbol is None or df_symbol.empty == True:
    return None
  else:
    symbols = df_symbol['symbol'].values
  
  dict_df = equity_daily_data_load_within_range(
    market,
    symbols = symbols,
    start_date = start_date,
    end_date = end_date,
    column_option = 'all',
    dir_option = dir_option,
    file_name = file_name)

  # Ensure all dataframes have the same columns
  first_key = next(iter(dict_df))
  columns = dict_df[first_key].columns

  for key, df in dict_df.items():
    if not df.columns.equals(columns):
      raise ValueError("All DataFrames in the dictionary must have the same columns.")


  list_df = []
  for key, df in dict_df.items():
      df.index.name = 'date'
      df['ticker'] = key  # Add the dictionary key as a column
      df = df.set_index('ticker', append=True) # Make the key a level in MultiIndex
      list_df.append(df)


  multi_index_df = pd.concat(list_df)

  return multi_index_df
