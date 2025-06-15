import json
import os

import yfinance as yf
import pandas as pd




def sort_symbols(symbols: list[str]) -> list[str]:
    """Sort a list of US equity symbols alphabetically.
    
    Args:
        symbols: List of ticker symbols to sort
        
    Returns:
        List of symbols sorted alphabetically
    """
    return sorted(symbols, key=lambda s: s.upper())


def us_symbol_file():
  
  # Read directory from JSON file
  conf = os.path.join(os.getenv("QUANT_FREE_ROOT"), 'config.json')
  with open(conf) as f:
      config = json.load(f)
  root_dir = config['data_dir']

  # Create the output directory if it doesn't exist
  os.makedirs(root_dir, exist_ok=True)

  # Create the subfolder
  subfolder_path = os.path.join(root_dir, 'symbol')
  os.makedirs(subfolder_path, exist_ok=True)

  csv_file_path = os.path.join(subfolder_path, 'equity_symbol.csv')
  
  return csv_file_path
