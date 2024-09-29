import json
import os
from pathlib import Path
import yfinance as yf
import pandas as pd

_this_dir = Path(__file__).parent.parent

def create_directory(root_dir, *subdirs):
    """Create a directory or nested directories if they don't exist."""
    path = os.path.join(root_dir, *subdirs)
    os.makedirs(path, exist_ok=True)
    return path

def get_root_directory():
    """Read the root directory path from the config.json file."""
    config_path = os.path.join(_this_dir, '../config.json')
    with open(config_path) as file:
        config = json.load(file)
    return config['output_dir']

def create_common_directory(*subdirs):
    """Create a common directory or nested directories under the root directory."""
    root_dir = get_root_directory()
    return create_directory(root_dir, *subdirs)

def us_symbol_file(filename='us_equity_symbol.csv'):
    """Get the file path for the symbol file."""
    symbol_dir = create_common_directory('symbol')
    return os.path.join(symbol_dir, filename)

def us_equity_folder(symbol='AAPL'):
    """Get the folder path for a specific equity."""
    return create_common_directory('equity', symbol)

def us_equity_sub_folder(symbol='AAPL', sub_dir='2024-06-06'):
    """Get the subfolder path for a specific equity and date."""
    return create_common_directory('equity', symbol, sub_dir)

def us_equity_research_folder(sub_folder='price', file_name='default.csv', data=None):
    """Save equity research data to a CSV file if data is provided."""
    research_dir = create_common_directory('research', sub_folder)
    file_path = os.path.join(research_dir, file_name)
    
    if data is not None:
        data.to_csv(file_path)
    return file_path

def us_dir1_store_csv(dir0 = 'symbol', dir1 = 'xq', filename='industry.csv', encoding='utf-8', data = None):
    """Get the file path for the symbol file."""
    symbol_dir = create_common_directory(dir0, dir1)
    file_path = os.path.join(symbol_dir, filename)
    if data is not None:
        data.to_csv(file_path, encoding = encoding)

def us_dir0_store_csv(dir0 = 'symbol', filename='industry.csv', data = None):
    """Get the file path for the symbol file."""
    symbol_dir = create_common_directory(dir0)
    file_path = os.path.join(symbol_dir, filename)
    if data is not None:
        data.to_csv(file_path)

def us_dir1_load_csv(dir0 = 'symbol', dir1 = 'xq', filename='industry.csv'):
    """Get the file path for the symbol file."""
    symbol_dir = create_common_directory(dir0, dir1)
    file_path = os.path.join(symbol_dir, filename)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col=0)

        df.index = pd.to_datetime(df.index).date
        # Drop the 'Unnamed: 0.1' column if it exists
        if 'Unnamed: 0.1' in df.columns:
            df.drop(columns=['Unnamed: 0.1'], inplace=True)

        # Drop the 'Unnamed: 0' column if it exists
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)
        df = df.fillna(0)
        return df
    else:
        print(f"File {filename} does not exist in the {symbol_dir} directory.")
        return None

def us_dir0_load_csv(dir0 = 'symbol', filename='industry.csv'):
    """Get the file path for the symbol file."""
    symbol_dir = create_common_directory(dir0)
    file_path = os.path.join(symbol_dir, filename)
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        print(f"File {filename} does not exist in the {symbol_dir} directory.")
        return None