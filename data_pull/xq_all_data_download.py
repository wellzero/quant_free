import os
import sys
import argparse
from quant_free.dataset.xq_data_download import *
from quant_free.dataset.xq_trade_data import *
from quant_free.dataset.xq_symbol import *
from quant_free.dataset.us_equity_download import *

def download_market_data(markets, skip_daily=False, skip_finance=False, symbols=None):
    """
    Download all data for specified markets and symbols
    
    Example workflow:
    1. First download symbols and sectors:
       $ python xq_finance_download.py --markets us,cn
    2. Then download daily kline data:
       $ python xq_finance_download.py --markets us,cn
    3. Finally download financial reports:
       $ python xq_finance_download.py --markets us,cn
       
    To skip financial report download:
       $ python xq_finance_download.py --markets us,cn --skip-finance
    """
    for market in markets:
        print(f"Processing {market.upper()} market...")
        # Equity and sector download
        if symbols is None:
            xq_symbol_download(market)
            xq_sector_download(market)
        
        # Load symbols if not provided
        if symbols is None:
            symbols = symbol_load(market)
        else:
            symbols = [s.strip().upper() for s in symbols.split(',')] if symbols else []

        if len(symbols) == 0:
            print(f"No symbols provided for {market.upper()} market. Pls configure symbol rightly.")
            exit(0)
        
        if not skip_daily:
            print(f"Downloading daily kline data for {market.upper()} market...")
            xq_kline_download(market, symbols)
        
        # Finance report (optional)
        if not skip_finance:
            xq_finance_download(market, symbols)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Download XQ finance data for specified markets',
        epilog='Examples:\n'
               '  python xq_all_data_download.py  # All markets (default)\n'
               '  python xq_all_data_download.py --markets us,cn  # US and China markets\n'
               '  python xq_all_data_download.py --markets us --skip-finance  # Skip financial reports\n'
               '  python xq_all_data_download.py --markets us --symbols AAPL,MSFT  # Download specific symbols'
    )
    parser.add_argument('--markets', default='us,cn,hk',
                       help='Comma-separated list of markets to download (default: us,cn,hk)')
    parser.add_argument('--skip-finance', action='store_true',
                       help='Skip downloading financial reports')
    parser.add_argument('--skip-daily', action='store_true',
                       help='Skip downloading daily kline data')
    parser.add_argument('--symbols', default=None,
                       help='Comma-separated list of symbols to download (default: all symbols)')
    
    args = parser.parse_args()
    
    markets = [m.strip().lower() for m in args.markets.split(',')]
    valid_markets = {'us', 'cn', 'hk'}
    invalid = set(markets) - valid_markets
    if invalid:
        raise ValueError(f"Invalid markets: {invalid}. Valid options are: us, cn, hk")
    
    print(f"Starting download for markets: {', '.join(m.upper() for m in markets)}")
    download_market_data(markets, args.skip_daily, args.skip_finance, args.symbols)
    print(f"Completed download for markets: {', '.join(m.upper() for m in markets)}")