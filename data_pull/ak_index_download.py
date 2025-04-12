import akshare as ak
import pandas as pd
from pathlib import Path
from quant_free.utils.us_equity_utils import create_directory, us_dir1_store_csv

def download_all_index_symbols(market='cn'):
    """Download all index symbols using akshare for specified market"""
    try:
        if market == 'cn':
            # Get all Chinese A-share index symbols
            index_df = ak.index_zh_a_spot_em()
        elif market == 'hk':
            # Get all Hong Kong index symbols
            index_df = ak.index_hk_spot_em()
        elif market == 'us':
            # Get all US index symbols
            index_df = ak.index_us_spot_em()
        else:
            print(f"Unsupported market: {market}")
            return pd.DataFrame()
            
        return index_df
    except Exception as e:
        print(f"Error downloading {market} index symbols: {e}")
        return pd.DataFrame()

def download_index_daily_data(symbol, market='cn'):
    """Download daily trade data for a specific index symbol in specified market"""
    try:
        if market == 'cn':
            df = ak.index_zh_a_hist(symbol=symbol, period="daily")
        elif market == 'hk':
            df = ak.index_hk_hist(symbol=symbol, period="daily")
        elif market == 'us':
            df = ak.index_us_hist(symbol=symbol, period="daily")
        else:
            print(f"Unsupported market: {market}")
            return pd.DataFrame()
            
        return df
    except Exception as e:
        print(f"Error downloading data for {market} index {symbol}: {e}")
        return pd.DataFrame()

def download_all_index_data(market='cn'):
    """Download all index symbols and their daily trade data for specified market"""
    # Get all index symbols
    index_df = download_all_index_symbols(market)
    
    if index_df.empty:
        return False
        
    # Create directory structure
    create_directory(market, 'index')
    
    # Determine symbol column based on market
    symbol_col = {
        'cn': '代码',
        'hk': '代码',
        'us': 'symbol'
    }.get(market, '代码')
    
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
        symbol = row[symbol_col]
        print(f"Downloading data for {market} index: {symbol}")
        
        # Get daily data
        daily_df = download_index_daily_data(symbol, market)
        
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
    # Download data for all supported markets
    for market in ['cn', 'hk', 'us']:
        print(f"Processing {market.upper()} market...")
        download_all_index_data(market)
