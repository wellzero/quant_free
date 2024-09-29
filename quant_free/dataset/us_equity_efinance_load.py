import os
import pandas as pd
from quant_free.utils.us_equity_symbol import *
from quant_free.utils.us_equity_utils import *
from quant_free.common.us_equity_common import *

def us_equity_efinance_common_shares_load(symbols = ['AAPL']):
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

    data.set_index(['REPORT'], inplace=True)

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
    else:
      dates = us_equity_efinance_search_sort_date(data.index)

    df = data.loc[dates,:]
    return df

def us_equity_efinance_store_csv(symbol, file_name, data, provider = "efinance" ):

    equity_folder = us_equity_sub_folder(symbol = symbol, sub_dir = provider)
    file = os.path.join(equity_folder, file_name + '.csv')

    data.to_csv(file)

def us_equity_efinance_factors_load_csv(symbol, file_name, start_time, end_time, factors = ['roe'], provider = "efinance"):

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
    years = sorted(set(idx.split('/')[0] for idx in df.index.get_level_values('REPORT')))

    columns = df.columns[2:]
    
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

def us_equity_efinance_finance_data_load(symbol = 'AAPL', dates = ["2021/Q1"]):

  try:
    # efinance_symbol = datacenter.get_secucode("MMM")
    print(f"Loading {symbol} finance data...")
    income = us_equity_efinance_load_csv(symbol, 'income', dates)
    # Apply the function to the DataFrame
    income = calculate_quarterly_differences(income, '基本加权平均股数-普通股', '摊薄加权平均股数-普通股')

    cash = us_equity_efinance_load_csv(symbol, 'cash', dates)
    cash = calculate_quarterly_differences(cash, '现金及现金等价物期初余额', '现金及现金等价物期末余额')
    
    balance = us_equity_efinance_balance_load_csv(symbol, dates, 'balance')

    data = pd.concat([income, cash, balance], axis = 1, join='inner')
    
    data = data.loc[:, ~data.columns.duplicated()]
    return data
    # us_equity_efinance_finance_store_csv(equity_folder, data, 'metrics')
  except:
    print(f"function {__name__}  error!!")