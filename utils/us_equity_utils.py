import json
import os
from pathlib import Path
import yfinance as yf
import pandas as pd

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

def us_equity_sub_folder(symbol = "AAPL", sub_dir = "2024-06-06"):

  # Read directory from JSON file
  root_dir = us_equity_folder(symbol)

  # Create the subfolder
  dir = os.path.join(root_dir, sub_dir)
  os.makedirs(dir, exist_ok=True)

  return dir

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
