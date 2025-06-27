import os
import pandas as pd
# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from quant_free.utils.us_equity_symbol import *
from quant_free.utils.us_equity_utils import *
from quant_free.common.us_equity_common import *

# def daily_trade_load(market, symbol='AAPL', options=['close'], sub_dir='xq'):
#     """Loads daily equity data."""
#     try:
#         data = us_dir1_load_csv(market,
#                  dir0 = 'equity',
#                  dir1 = symbol,
#                  dir2 = sub_dir,
#                  filename= 'daily.csv')

#         data.set_index('timestamp', inplace=True)
#         data = data.replace(['--', '_', 'None'], 0).fillna(0)
#         return data.loc[:, options]
#     except Exception as e:
#         logger.error(f"Failed to load daily data for {symbol}: {e}")
#         return pd.DataFrame()
    
def daily_trade_load(
    market = 'us', 
    equity='equity',
    symbol = 'AAPL',
    dir_option = 'xq',
    file_name = 'daily.csv',
    interval = 'day'):
  """Load equity data from a CSV file.
  Args:
      market (str): The market type, e.g., 'us'.
      equity (str): The type of equity, e.g., 'equity' or 'index'.
      symbol (str): The stock symbol.
      dir_option (str): Optional directory option for additional subdirectories.
      file_name (str): The name of the CSV file to load.
      interval (str): The time interval for the data, e.g., 'day' or 'minute'.
  Returns:
      pd.DataFrame: The loaded DataFrame with the date as the index.
  """
  data = us_dir1_load_csv(
     market,
     dir0 = equity,
     dir1 = dir_option,
     dir2 = symbol,
     filename = file_name)
  
  if 'timestamp' in data.columns:
    data.rename(columns={'timestamp': 'date'}, inplace=True)

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
  if market == "us":
    stock_data = daily_trade_load(
                  market,
                  symbol = "AAPL",
                  dir_option = dir_option)
  elif market == "cn":
    stock_data = daily_trade_load(
                  market = market,
                  equity = 'index',
                  symbol = 'SH000001',
                  file_name = 'daily.csv')
  elif market == "hk":
    stock_data = daily_trade_load(
                  market = market,
                  equity = 'index',
                  symbol = 'HSCI',
                  file_name = 'daily.csv')
  filtered_data = stock_data.loc[start_date:end_date]
  trade_dates_time = filtered_data.index

  return trade_dates_time

def multi_sym_daily_load(
    market = 'us', 
    symbols = ['AAPL'],
    start_date = '2023-05-29',
    end_date = '2024-05-29',
    column_option = "all",
    dir_option = '',
    equity='equity',
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
    try:
      # try:
        # print(f"loading {symbol} trade data...")

        data_tmp = daily_trade_load(market = market,
                                       equity = equity,
                                       symbol = symbol,
                                       dir_option = dir_option,
                                       file_name = file_name)
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
    except Exception as e:
      print(f"market {market} lack of some trade date skip {symbol}, file not exist {file_name}")
      lack_list.append(symbol)

  if len(lack_list) > 0:
    print(f"miss these daily trade files {lack_list}")
  return data

# load data fro a sector
def equity_sector_daily_load(
    market = 'us',
    sector_name = 'Semiconductor Products and Equipment',
    start_date = '2023-05-29',
    end_date = '2024-05-29',
    column_option = 'all',
    dir_option = 'xq'):

  symbols = us_dir1_load_csv(market,
    dir0 = 'symbol', dir1 = 'xq',
    filename= sector_name +'.csv')['symbol'].values

  return multi_sym_daily_load(
    market,
    symbols = symbols,
    start_date = start_date,
    end_date = end_date,
    column_option = column_option,
    dir_option = dir_option)

# multi index load [date ticker]
def multi_sym_daily_load_multi_index(
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
  
  dict_df = multi_sym_daily_load(
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

def factor_load(
      market,
      symbol,
      file_name,
      start_time,
      end_time,
      factors=['roe'],
      provider="xq"):
    """Loads financial factors from a CSV within a date range."""
    try:
        data = us_dir1_load_csv(market,
                 dir0 = 'equity',
                 dir1 = symbol,
                 dir2 = provider,
                 filename= file_name)

        data.set_index('REPORT_DATE', inplace=True)
        data.index = pd.to_datetime(data.index)

        start_date = pd.to_datetime(start_time)
        end_date = pd.to_datetime(end_time)

        mask = (data.index >= start_date) & (data.index <= end_date)
        return data.loc[mask, factors]
    except Exception as e:
        logger.error(f"Failed to load factors for {symbol}: {e}")
        return pd.DataFrame()

