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
