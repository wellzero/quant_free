import akshare as ak
import pandas as pd
from pathlib import Path
import os
import sys
_this_dir = Path(__file__).parent.parent
sys.path.append(str(_this_dir))
from quant_free.utils.us_equity_utils import create_directory, us_dir1_store_csv





def download_all_index_symbols(market='cn'):
    """Download all index symbols using akshare for specified market"""
    try:
        if market == 'cn':
            # Get all Chinese index symbols from Sina
            index_df = ak.stock_zh_index_spot_sina()
        elif market == 'hk':
            # Get all Hong Kong index symbols
            index_df = ak.stock_hk_index_spot_em()
        elif market == 'us':
            # Get US index symbols from a predefined list since akshare doesn't support US indexes
            us_indexes = [
                {'symbol': '^GSPC', 'name': 'S&P 500'},
                {'symbol': '^DJI', 'name': 'Dow Jones Industrial Average'},
                {'symbol': '^IXIC', 'name': 'NASDAQ Composite'},
                {'symbol': '^RUT', 'name': 'Russell 2000'},
                {'symbol': '^VIX', 'name': 'CBOE Volatility Index'},
            ]
            index_df = pd.DataFrame(us_indexes)
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
            df = ak.stock_zh_index_daily_em(symbol=symbol)
        elif market == 'hk':
            df = ak.stock_hk_index_daily_em(symbol=symbol)
        elif market == 'us':
            # For US indexes, use stock_zh_a_hist instead
            df = ak.stock_us_daily(symbol=symbol)
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
        'cn': 'symbol',
        'hk': '代码',
        'us': 'symbol'
    }.get(market, 'symbol')
    
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
