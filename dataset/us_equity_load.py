import os
import pandas as pd
from utils.us_equity_symbol import *
from utils.us_equity_utils import *
from common.us_equity_common import *


def us_equity_symbol_load():
  us_symbol_f = us_symbol_file()
  us_all_companies = pd.read_csv(us_symbol_f)
  return us_all_companies.loc[:, 'symbol']

def us_equity_daily_data_read_csv(symbol = 'AAPL'):
  equity_folder = us_equity_folder(symbol = symbol)
  equity_file = os.path.join(equity_folder, 'daily.csv')
  data = pd.read_csv(equity_file)
  data.set_index('date', inplace=True)
  return data

def us_equity_daily_data_load(symbols = ['AAPL'], start_date = '2023-05-29', end_date = '2024-05-29', option = all):
  data = {}
  trade_date_len = len(us_equity_get_trade_date_within_range(start_date = start_date, end_date = end_date))
  for symbol in symbols:
    try:
      # print(f"loading {symbol} trade data...")

      data_tmp = us_equity_daily_data_read_csv(symbol)
      data_tmp = data_tmp.loc[start_date:end_date]
      # data_tmp = data_tmp[(data_tmp['date'] >= start_date) & (data_tmp['date'] <= end_date)]
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

def us_equity_common_shares_load(symbols = ['AAPL']):
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


def us_equity_efinance_search_sort_date(report_dates):
   
   dates = [date for date in report_dates if date.split('/')[-1] in ['Q1', 'Q6', 'Q9', 'FY']]
   dates = [x.replace("FY", "QA") for x in dates]
   dates.sort()
   dates = [x.replace("QA", "FY") for x in dates]
   return dates

def us_equity_efinance_load_csv(symbol, file_name, dates = None, provider = "efinance" ):

    equity_folder = us_equity_sub_folder(symbol = symbol, sub_dir = provider)
    file = os.path.join(equity_folder, file_name + '.csv')

    data = pd.read_csv(file)

    if dates is None:
      dates = us_equity_efinance_search_sort_date(data['REPORT'])

    data.set_index('REPORT', inplace=True)

    df = data.loc[dates,:]

    return df

def us_equity_efinance_balance_load_csv(symbol, dates, file_name, provider = "efinance" ):

    equity_folder = us_equity_sub_folder(symbol = symbol, sub_dir = provider)
    file = os.path.join(equity_folder, file_name + '.csv')
    data = pd.read_csv(file)
    data.set_index('REPORT', inplace=True)
    
    if dates is not None:
      replacements = {'Q3': 'Q9', 'Q2': 'Q6', 'Q4': 'FY'}
      updated_date = [d.replace(d[-2:], replacements.get(d[-2:], d[-2:])) for d in dates]
      data = data.loc[updated_date]
      data.index =  dates

    return data

def us_equity_efinance_store_csv(symbol, file_name, data, provider = "efinance" ):

    equity_folder = us_equity_sub_folder(symbol = symbol, sub_dir = provider)
    file = os.path.join(equity_folder, file_name + '.csv')

    data.to_csv(file)

# Function to perform the operations
def calculate_quarterly_differences(df, str1, str2):
    # Define the variables
    Q1 = 'Q1'
    Q6 = 'Q6'
    Q9 = 'Q9'
    FY = 'FY'
    # Create new DataFrame to store the results
    result_df = df.copy()
    
    # Get the unique years from the index
    years = sorted(set(idx.split('/')[0] for idx in df.index))
    
    for year in years:
        if f'{year}/{Q9}' in df.index and f'{year}/{FY}' in df.index:
            result_df.loc[f'{year}/{FY}'] -= df.loc[f'{year}/{Q9}']
        if f'{year}/{Q6}' in df.index and f'{year}/{Q9}' in df.index:
            result_df.loc[f'{year}/{Q9}'] -= df.loc[f'{year}/{Q6}']
        if f'{year}/{Q1}' in df.index and f'{year}/{Q6}' in df.index:
            result_df.loc[f'{year}/{Q6}'] -= df.loc[f'{year}/{Q1}']
    if str1 in df.columns:
      result_df.loc[:,str1] = df.loc[:, str1]
    if str2 in df.columns:
      result_df.loc[:,str2] = df.loc[:, str2]
    return result_df

def us_equity_efinance_finance_data_load(symbol = 'AAPL', dates = ["2021/Q1"]):

  try:
    # efinance_symbol = datacenter.get_secucode("MMM")
    print(f"Loading {symbol} finance data...")
    income = us_equity_efinance_load_csv(symbol, 'income', dates)
    # Apply the function to the DataFrame
    income = calculate_quarterly_differences(income, 'basic_weighted_average_shares_common_stock', 'diluted_weighted_average_shares_common_stock')

    cash = us_equity_efinance_load_csv(symbol, 'cash', dates)
    cash = calculate_quarterly_differences(cash, 'cash_and_cash_equivalents_at_beginning_of_period', 'cash_and_cash_equivalents_at_end_of_period')
    
    balance = us_equity_efinance_balance_load_csv(symbol, dates, 'balance')

    data = pd.concat([income, cash, balance], axis = 1)
    
    data = data.loc[:, ~data.columns.duplicated()]
    return data
    # us_equity_efinance_finance_store_csv(equity_folder, data, 'metrics')
  except:
    print(f"failed to download equity {symbol}")