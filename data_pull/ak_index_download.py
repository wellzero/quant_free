import akshare as ak
import pandas as pd
from pathlib import Path
from quant_free.utils.us_equity_utils import create_directory, us_dir1_store_csv

def download_all_index_symbols():
    """Download all index symbols using akshare"""
    try:
        # Get all index symbols
        index_df = ak.index_zh_a_spot_em()
        return index_df
    except Exception as e:
        print(f"Error downloading index symbols: {e}")
        return pd.DataFrame()

def download_index_daily_data(symbol):
    """Download daily trade data for a specific index symbol"""
    try:
        # Get historical data for the index
        df = ak.index_zh_a_hist(symbol=symbol, period="daily")
        return df
    except Exception as e:
        print(f"Error downloading data for index {symbol}: {e}")
        return pd.DataFrame()

def download_all_index_data():
    """Download all index symbols and their daily trade data"""
    # Get all index symbols
    index_df = download_all_index_symbols()
    
    if index_df.empty:
        return False
        
    # Create directory structure
    market = 'cn'
    create_directory(market, 'index')
    
    # Save index symbols
    us_dir1_store_csv(
        market=market,
        dir0='index',
        dir1='ak',
        filename='index_symbols.csv',
        data=index_df
    )
    
    # Download daily data for each index
    for _, row in index_df.iterrows():
        symbol = row['代码']
        print(f"Downloading data for index: {symbol}")
        
        # Get daily data
        daily_df = download_index_daily_data(symbol)
        
        if not daily_df.empty:
            # Save daily data
            us_dir1_store_csv(
                market=market,
                dir0='index',
                dir1='ak',
                filename=f'{symbol}_daily.csv',
                data=daily_df
            )
    
    return True

if __name__ == "__main__":
    download_all_index_data()
