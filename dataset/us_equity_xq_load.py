import os
import pandas as pd
import ast
from utils.us_equity_symbol import *
from utils.us_equity_utils import *
from common.us_equity_common import *
from utils.xq_parse_js import *


GLOBAL_XQ_IGNORE_COLUMNS = 7

def us_equity_xq_daily_data_load(symbol = 'AAPL', options = ['close'], sub_dir = 'xq'):

  try:
    # print(f"loading {symbol} trade data...")
    equity_folder = us_equity_sub_folder(symbol = symbol, sub_dir = sub_dir)
    equity_file = os.path.join(equity_folder, 'daily.csv')
    data = pd.read_csv(equity_file)
    data.set_index('timestamp', inplace=True)
    data = data.replace('--', 0)
    data = data.replace('_', 0)
    data = data.replace('None', 0)
    data = data.fillna(0)
    return data.loc[:, options]
  except:
    print(f"failed to load equity {symbol}")
  return data

def us_equity_xq_search_sort_date(report_dates):
   
   dates = [date for date in report_dates if date.split('/')[-1] in ['Q1', 'Q6', 'Q9', 'FY']]
   dates = [x.replace("FY", "QA") for x in dates]
   dates.sort()
   dates = [x.replace("QA", "FY") for x in dates]
   return dates

def us_equity_xq_load_csv(symbol, file_name, title, provider = "xq" ):

    equity_folder = us_equity_sub_folder(symbol = symbol, sub_dir = provider)
    file = os.path.join(equity_folder, file_name + '.csv')

    df_ = pd.read_csv(file)
    # Perform the operations
    title_cn = [title[key] if key in title else key for key in df_.columns[GLOBAL_XQ_IGNORE_COLUMNS:]]
    df_.columns = list(df_.columns[:GLOBAL_XQ_IGNORE_COLUMNS]) + title_cn  # Rename columns
    
    df_['report_name'] = df_['report_name'].str.replace('年', '/')  # Replace '年' with '/'
    df_.set_index('report_name', inplace=True)  # Set 'report_date' as index
    
    # # Function to extract the first value from lists with more than one element
    # def get_first_value(cell):
    #     if isinstance(cell, list) and len(cell) > 1:
    #         return cell[0]
    #     return cell

    # # Apply the function to each cell in the DataFrame
    # df_first_values = df_.applymap(get_first_value)
    
    # Function to convert string representation of lists to actual lists
    def str_to_list(cell):
        try:
            return ast.literal_eval(cell)
        except (ValueError, SyntaxError):
            return cell

    # Apply the conversion function to each cell in the DataFrame
    df_ = df_.applymap(str_to_list)

    # Function to extract the first value from lists with more than one element
    def get_first_value(cell):
        if isinstance(cell, list) and len(cell) > 1:
            return cell[0]
        return cell

    # Apply the function to get the first value in each cell
    df_first_values = df_.applymap(get_first_value)
    
    df_first_values = df_first_values.sort_values(by='report_date', ascending=True)

    df_first_values = df_first_values.replace('--', 0)
    df_first_values = df_first_values.replace('_', 0)
    df_first_values = df_first_values.replace('None', 0)
    df_first_values = df_first_values.fillna(0)
    
    return df_first_values

def us_equity_xq_balance_load_csv(symbol, dates, file_name, provider = "xq" ):

    equity_folder = us_equity_sub_folder(symbol = symbol, sub_dir = provider)
    file = os.path.join(equity_folder, file_name + '.csv')
    data = pd.read_csv(file)
    data.set_index('REPORT', inplace=True)
    
    if dates is not None:
      replacements = {'Q3': 'Q9', 'Q2': 'Q6', 'Q4': 'FY'}
      updated_date = [d.replace(d[-2:], replacements.get(d[-2:], d[-2:])) for d in dates]
      data = data.loc[updated_date]
      data.index =  dates
    else:
      dates = us_equity_xq_search_sort_date(data.index)

    df = data.loc[dates,:]
    return df

def us_equity_xq_store_csv(symbol, file_name, data, provider = "xq" ):

    equity_folder = us_equity_sub_folder(symbol = symbol, sub_dir = provider)
    file = os.path.join(equity_folder, file_name + '.csv')

    data.to_csv(file)

def us_equity_xq_factors_load_csv(symbol, file_name, start_time, end_time, factors = ['roe'], provider = "xq"):

    equity_folder = us_equity_sub_folder(symbol = symbol, sub_dir = provider)
    file = os.path.join(equity_folder, file_name + '.csv')
    df = pd.read_csv(file)
    df.set_index('REPORT_DATE', inplace=True)
    df.index = pd.to_datetime(df.index)

    start_date = pd.to_datetime(start_time)
    end_date = pd.to_datetime(end_time)

    # Find indices within the date range
    mask = (df.index >= start_date) & (df.index <= end_date)
    indices_in_range = df[mask].index


    # If you want to get the data for these indices:
    data_in_range = df.loc[indices_in_range]

    return data_in_range.loc[:, factors]

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
    years = sorted(set(idx.split('/')[0] for idx in df.index.get_level_values('report_name')))

    columns = df.columns[GLOBAL_XQ_IGNORE_COLUMNS:]
    
    for year in years:
        if f'{year}/{Q9}' in df.index and f'{year}/{FY}' in df.index:
            result_df.loc[f'{year}/{FY}', columns] -= df.loc[f'{year}/{Q9}', columns]
        if f'{year}/{Q6}' in df.index and f'{year}/{Q9}' in df.index:
            result_df.loc[f'{year}/{Q9}', columns] -= df.loc[f'{year}/{Q6}', columns]
        if f'{year}/{Q1}' in df.index and f'{year}/{Q6}' in df.index:
            result_df.loc[f'{year}/{Q6}', columns] -= df.loc[f'{year}/{Q1}', columns]

    if str1 in df.columns:
      result_df.loc[:,str1] = df.loc[:, str1]
    if str2 in df.columns:
      result_df.loc[:,str2] = df.loc[:, str2]
    return result_df

def us_equity_xq_finance_data_load(symbol = 'AAPL'):

  try:
    xq_us_name = xq_js_to_dict('us')
    # xq_symbol = datacenter.get_secucode("MMM")
    print(f"Loading {symbol} xq data...")
    income = us_equity_xq_load_csv(symbol, 'income', xq_us_name['incomes'])
    # Apply the function to the DataFrame
    income = calculate_quarterly_differences(income, '基本每股收益', '稀释每股收益')

    cash = us_equity_xq_load_csv(symbol, 'cash', xq_us_name['cashes'])
    cash = calculate_quarterly_differences(cash, '期初现金及现金等价物余额', '期末现金及现金等价物余额')
    
    balance = us_equity_xq_load_csv(symbol,'balance', xq_us_name['balances'])

    indicators = us_equity_xq_load_csv(symbol,'metrics', xq_us_name['indicators'])

    data = pd.concat([income, cash, balance, indicators], axis = 1, join='inner')
    
    data = data.loc[:, ~data.columns.duplicated()]
    return data
    # us_equity_xq_finance_store_csv(equity_folder, data, 'metrics')
  except:
    print(f"function {__name__}  error!!", symbol)