import os
import pandas as pd
from utils.us_equity_symbol import *
from utils.us_equity_utils import *
from common.us_equity_common import *
from openbb import obb
import yfinance as yf

# Read directory from JSON file

def us_equity_symbol_download(provider="sec"):
  # Get all companies from SEC
  us_all_companies = obb.equity.search("", provider=provider)
  csv_us_all_companies = us_all_companies.to_dataframe()

  us_symbol_f = us_symbol_file()

  # Save the DataFrame to a CSV file in the output directory
  csv_us_all_companies.to_csv(us_symbol_f, index=False)
  print(us_equity_symbol_download.__name__, "download successfully!", f"and stored to {us_symbol_f}")

def us_equity_daily_data_download(symbols = ['AAPL'], provider="yfinance"):
  for symbol in symbols:
    try:
      print(f"Downloading {symbol} trade data...")
      df_daily = obb.equity.price.historical(symbol = symbol, start_date = "1990-01-01", provider=provider).to_df()
      equity_folder = us_equity_folder(symbol = symbol)
      equity_file = os.path.join(equity_folder, 'daily.csv')
      # data = obb.equity.download(symbol, provider=provider, interval="daily", start_date="2010-01-01").to_df()
      data = obb.equity.price.historical(symbol = symbol, start_date = "1990-01-01", provider=provider).to_df()
      data.to_csv(equity_file)
    except:
      print(f"function {__name__} error!!")

def us_equity_yfinance_finance_store_csv(equity_folder, data, file_name):
    file = os.path.join(equity_folder, file_name + '.csv')
    if os.path.exists(file):
      data_local = pd.read_csv(file)

      # Drop the 'Unnamed: 0.1' column if it exists
      if 'Unnamed: 0.1' in data_local.columns:
          data_local.drop(columns=['Unnamed: 0.1'], inplace=True)

      # Drop the 'Unnamed: 0' column if it exists
      if 'Unnamed: 0' in data_local.columns:
          data_local.drop(columns=['Unnamed: 0'], inplace=True)

      merged_data = pd.concat([data_local, data])
      if 'period_ending' in data_local.columns:
        merged_data.drop_duplicates(subset='period_ending', inplace=True)
      data = merged_data
    data.to_csv(file, index=False)

def us_equity_yfinance_finance_data_download(symbols = ['AAPL'], provider="yfinance"):
  for symbol in symbols:
    try:
      print(f"Downloading {symbol} finance data...")
      equity_folder = us_equity_folder(symbol = symbol)

      data = obb.equity.fundamental.income(symbol, provider="yfinance", limit=3, period="quarter").to_df()
      us_equity_yfinance_finance_store_csv(equity_folder, data, 'income')

      data = obb.equity.fundamental.net_cash_flow_from_operating_activities(symbol, provider="yfinance", limit=3, period="quarter").to_df()
      us_equity_yfinance_finance_store_csv(equity_folder, data, 'cash')

      data = obb.equity.fundamental.balance(symbol, provider="yfinance", limit=3, period="quarter").to_df()
      us_equity_yfinance_finance_store_csv(equity_folder, data, 'balance')

      data = obb.equity.fundamental.metrics(symbol, provider="yfinance", limit=3, period="quarter").to_df()
      us_equity_yfinance_finance_store_csv(equity_folder, data, 'metrics')

    except:
      print(f"function {__name__} error!!")


def us_equity_efinance_finance_store_csv(equity_folder, data, file_name):
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
    data.to_csv(file)
    print("store file: ", file)


def us_equity_efinance_finance_data_download(symbols = ['AAPL'], provider="efinance"):

  import efinance as ef
  datacenter = ef.stock.us_finance_getter()

  for symbol in symbols:
    try:
      # efinance_symbol = datacenter.get_secucode("MMM")
      print(f"Downloading {symbol} finance data...")
      efinance_symbol = datacenter.get_secucode(symbol)
      equity_folder = us_equity_sub_folder(symbol = symbol, sub_dir = provider)

      data = datacenter.get_us_finance_income(symbol = efinance_symbol)
      us_equity_efinance_finance_store_csv(equity_folder, data, 'income')

      data = datacenter.get_us_finance_cash(symbol = efinance_symbol)
      us_equity_efinance_finance_store_csv(equity_folder, data, 'net_cash_flow_from_operating_activities')

      data = datacenter.get_us_finance_balance(symbol = efinance_symbol)
      us_equity_efinance_finance_store_csv(equity_folder, data, 'balance')

      data = datacenter.get_us_finance_main_factor(symbol = efinance_symbol)
      us_equity_efinance_finance_store_csv(equity_folder, data, 'metrics')
    except:
      print(f"function {__name__} error!!")

def us_equity_option_data_download(symbols = ['AAPL']):
  trade_date = us_equity_get_current_trade_date()
  for ticker_symbol in symbols:
    try:
      # Define the ticker symbol
      equity_folder_date = us_equity_sub_folder(symbol = ticker_symbol, sub_dir = trade_date)

      print("option folder: ", equity_folder_date)

      # Fetch the ticker object
      ticker = yf.Ticker(ticker_symbol)

      # Fetch the available expiration dates for options
      exp_dates = ticker.options
      print(f"Available expiration dates for {ticker_symbol}: {exp_dates}")

      # Loop through each expiration date and save the options data
      for exp_date in exp_dates:
          # Fetch the option chain for the current expiration date
          option_chain = ticker.option_chain(exp_date)
          
          # Extract call and put options data
          calls = option_chain.calls
          puts = option_chain.puts
          underlying = option_chain.underlying
          
          # Save the calls and puts to separate CSV files
          call_file = f'calls_{exp_date}.csv'
          put_file = f'puts_{exp_date}.csv'
          calls.to_csv(os.path.join(equity_folder_date, call_file), index=False)
          puts.to_csv(os.path.join(equity_folder_date, put_file), index=False)
          
          print(f"Saved options data for expiration date {call_file}")
          print(f"Saved options data for expiration date {put_file}")

          import pickle
          piklefile = os.path.join(equity_folder_date, 'underlying.pkl')
          # Writing the dictionary to a file
          with open(piklefile, "wb") as file:
              pickle.dump(underlying, file)
              print("dump file ", piklefile)

    except:
      print(f"function {__name__} error!!")

  print("All options data have been downloaded and saved to CSV files.")



def us_equity_info_data_download(symbols = ['AAPL'], provider="efinance"):

  import efinance as ef
  datacenter = ef.stock.us_equity_getter()

  for symbol in symbols:
    try:
      # efinance_symbol = datacenter.get_secucode("MMM")
      print(f"Downloading {symbol} finance data...")
      efinance_symbol = datacenter.get_secucode(symbol)
      equity_folder = us_equity_sub_folder(symbol = symbol, sub_dir = 'efinance')

      data = datacenter.get_us_equity_info(symbol = efinance_symbol)
      us_equity_efinance_finance_store_csv(equity_folder, data, 'info')
    except:
      print(f"function {__name__} error!!")


