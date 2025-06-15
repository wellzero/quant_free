import akshare as ak
import pandas as pd

import os
import sys

from quant_free.utils.us_equity_utils import create_directory, us_dir1_store_csv
import efinance as ef


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
                {'symbol': '.GSPC', 'name': 'S&P 500'},
                {'symbol': '.DJI', 'name': 'Dow Jones Industrial Average'},
                {'symbol': '.IXIC', 'name': 'NASDAQ Composite'},
                {'symbol': '.RUT', 'name': 'Russell 2000'},
                {'symbol': '.VIX', 'name': 'CBOE Volatility Index'},
            ]
            index_df = pd.DataFrame(us_indexes)
        else:
            print(f"Unsupported market: {market}")
            return pd.DataFrame()

        if not index_df.empty:
            column_mapping = {
                # '序号': 'index',
                # '内部编号': 'internal_code',
                '代码': 'symbol',
                '名称': 'name',
                '最新价': 'latest_price',
                '涨跌额': 'price_change',
                '涨跌幅': 'pct_change',
                '今开': 'open',
                '最高': 'high',
                '最低': 'low',
                '昨收': 'prev_close',
                '成交量': 'volume',
                '成交额': 'turnover'
            }
            index_df = index_df.rename(columns=column_mapping)
        return index_df
    except Exception as e:
        print(f"Error downloading {market} index symbols: {e}")
        return pd.DataFrame()

def download_all_index_data(market='cn'):
    """Download all index symbols and their daily trade data for specified market"""
    # Get all index symbols
    datacenter_xq = ef.stock.us_finance_xq_getter()

    index_df = download_all_index_symbols(market)
    
    if index_df.empty:
        return False
        
    # Save index symbols
    us_dir1_store_csv(
        market=market,
        dir0='symbol',
        dir1='xq',
        filename='index_symbol.csv',
        data=index_df,
        index=False
    )
    # Download daily data for each index
    for _, row in index_df.iterrows():

        symbol = row['symbol'].upper()
        symbol_xq = symbol

        if market == 'hk':
            symbol_xq = f"HK{symbol_xq}"

        print(f"Downloading data for {market} index: {symbol}")
        
        # Get daily data
        try:
            df = datacenter_xq.get_us_finance_daily_trade(symbol=symbol_xq)
        except Exception as e:
            print(f"Failed to download data for {symbol}: {e}")
            continue


        if not df.empty:
            # Save daily data
            us_dir1_store_csv(
                market=market,
                dir0='index',
                dir1=symbol,
                filename='daily.csv',
                data=df,
                index=False
            )
    
    return True

if __name__ == "__main__":
    # Download data for all supported markets
    # for market in ['cn', 'hk', 'us']:
    for market in ['hk', 'us']:
        print(f"Processing {market.upper()} market...")
        print(f"Processing {market.upper()} market...")
        download_all_index_data(market)
