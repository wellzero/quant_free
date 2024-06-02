import json
import os
from pathlib import Path
import yfinance as yf

_this_dir = Path(__file__).parent.parent

def us_symbol_file():
  
  # Read directory from JSON file
  conf = os.path.join(_this_dir, 'config.json')
  with open(conf) as f:
      config = json.load(f)
  root_dir = config['output_dir']

  # Create the output directory if it doesn't exist
  os.makedirs(root_dir, exist_ok=True)

  # Create the subfolder
  subfolder_path = os.path.join(root_dir, 'symbol')
  os.makedirs(subfolder_path, exist_ok=True)

  csv_file_path = os.path.join(subfolder_path, 'us_equity_symbol.csv')
  
  return csv_file_path

def us_equity_folder(symbol = "AAPL"):
  
  # Read directory from JSON file
  conf = os.path.join(_this_dir, 'config.json')
  with open(conf) as f:
      config = json.load(f)
  root_dir = config['output_dir']

  # Create the output directory if it doesn't exist
  os.makedirs(root_dir, exist_ok=True)

  # Create the subfolder
  equity_path = os.path.join(root_dir, 'equity', symbol)
  os.makedirs(equity_path, exist_ok=True)

  return equity_path

def us_equity_folder_date(symbol = "AAPL"):

  # Download historical stock data
  stock_data = yf.download(symbol)

  # Extract the date part from the Timestamp index
  trade_dates = stock_data.index.date

  trade_dates_str = [date.strftime('%Y-%m-%d') for date in trade_dates]

  # Read directory from JSON file
  root_dir = us_equity_folder(symbol)

  # Create the subfolder
  equity_path_date = os.path.join(root_dir, trade_dates_str[-1])
  os.makedirs(equity_path_date, exist_ok=True)

  return equity_path_date

def us_equity_research_folder(sub_folder = "price"):
  
  # Read directory from JSON file
  conf = os.path.join(_this_dir, 'config.json')
  with open(conf) as f:
      config = json.load(f)
  root_dir = config['output_dir']

  # Create the output directory if it doesn't exist
  os.makedirs(root_dir, exist_ok=True)

  # Create the subfolder
  equity_path = os.path.join(root_dir, 'research', sub_folder)
  os.makedirs(equity_path, exist_ok=True)

  return equity_path