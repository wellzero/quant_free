import json
import os

import yfinance as yf
import pandas as pd

def ensure_csv_extension(filename):
    if not filename.endswith('.csv'):
        filename += '.csv'
    return filename

def create_directory(root_dir, *subdirs):
    """Create a directory or nested directories if they don't exist."""
    path = os.path.join(root_dir, *subdirs)
    os.makedirs(path, exist_ok=True)
    return path

def get_json_config_value(key = None):
    config_path = os.path.join(os.getenv("QUANT_FREE_ROOT"), 'config.json')
    with open(config_path) as file:
        config = json.load(file)
    return config[key]

def get_root_directory():
    """Read the root directory path from the config.json file."""
    config_path = os.path.join(os.getenv("QUANT_FREE_ROOT"), 'config.json')
    with open(config_path) as file:
        config = json.load(file)
    return config['data_dir']

def create_common_directory(*subdirs):
    """Create a common directory or nested directories under the root directory."""
    root_dir = get_json_config_value("data_dir")
    return create_directory(root_dir, *subdirs)

def us_symbol_file(filename='equity_symbol.csv'):
    """Get the file path for the symbol file."""
    filename = ensure_csv_extension(filename)
    symbol_dir = create_common_directory('symbol')
    return os.path.join(symbol_dir, filename)

def us_equity_folder(market, equity='equity', symbol='AAPL'):
    """Get the folder path for a specific equity."""
    return create_common_directory(market, equity, symbol)

def equity_sub_folder(market = 'us', symbol='AAPL', sub_dir='2024-06-06'):
    """Get the subfolder path for a specific equity and date."""
    return create_common_directory(market, 'equity', symbol, sub_dir)

def us_equity_research_folder(market = 'us', sub_folder='price', file_name='default.csv', data=None):
    """Save equity research data to a CSV file if data is provided."""
    research_dir = create_common_directory(market, 'research', sub_folder)
    file_path = os.path.join(research_dir, file_name)
    
    if data is not None:
        data.to_csv(file_path)
    return file_path

def us_dir1_store_csv(market='us', dir0='equity', dir1=None, dir2=None,
                      filename='industry.csv',
                      encoding='utf-8',
                      data=None,
                      index=True):
    """Get the file path for the symbol file."""
    filename = ensure_csv_extension(filename)
    dirs = [d for d in [market, dir0, dir1, dir2] if d is not None]
    symbol_dir = create_common_directory(*dirs)
    file_path = os.path.join(symbol_dir, filename)
    if data is not None:
        data.to_csv(file_path, encoding=encoding, index=index)
        print(f"stored to folder {file_path}")

def us_dir0_store_csv(market = 'us', dir0 = 'symbol', filename='industry.csv', data = None):
    """Get the file path for the symbol file."""
    filename = ensure_csv_extension(filename)
    symbol_dir = create_common_directory(market, dir0)
    file_path = os.path.join(symbol_dir, filename)
    if data is not None:
        data.to_csv(file_path)
        print(f"stored to folder {file_path}")

def us_dir1_load_csv(market='us', dir0='equity', dir1=None, dir2=None, filename='industry.csv', dtype=None, index_col=0):
    """
    Load a CSV file from a directory structure with a variable number of subdirectories.

    Args:
        market (str): The main market directory (e.g., 'us').
        dir0 (str): The first subdirectory.
        dir1 (str, optional): The second subdirectory. Defaults to None.
        dir2 (str, optional): The third subdirectory. Defaults to None.
        filename (str): The name of the CSV file.
        dtype (dict, optional): Data types for columns in the CSV. Defaults to None.
        index_col (int or str, optional): The column to use as the index. Can be a column name or index. Defaults to 0.

    Returns:
        pandas.DataFrame: The loaded DataFrame, or None if the file does not exist.
    """
    filename = ensure_csv_extension(filename)
    dirs = [d for d in [market, dir0, dir1, dir2] if d is not None]
    symbol_dir = create_common_directory(*dirs)
    file_path = os.path.join(symbol_dir, filename)

    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, index_col=index_col, dtype=dtype)
            
            # # Convert index to datetime if it's a datetime-like column
            # if df.index.dtype == 'object':
            #     try:
            #         df.index = pd.to_datetime(df.index)
            #     except (ValueError, TypeError):
            #         pass  # Skip if index cannot be converted to datetime

            # Clean up unnamed columns
            for col in ['Unnamed: 0.1', 'Unnamed: 0']:
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)

            df = df.fillna(0)
            return df
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None
    else:
        print(f"File {filename} does not exist in the {symbol_dir} directory.")
        return None
    
def us_dir0_load_csv(market = 'us', dir0 = 'symbol', filename='industry.csv'):
    """Get the file path for the symbol file."""
    filename = ensure_csv_extension(filename)
    symbol_dir = create_common_directory(market, dir0)
    file_path = os.path.join(symbol_dir, filename)
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        print(f"File {filename} does not exist in the {symbol_dir} directory.")
        return None