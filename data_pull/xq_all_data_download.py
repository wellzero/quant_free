import os
import sys
import argparse
from quant_free.dataset.xq_data_download import *
from quant_free.dataset.us_equity_load import *
from quant_free.dataset.us_equity_download import *

def download_market_data(market, skip_finance=False):
    """
    Download all data for specified market
    
    Example workflow:
    1. First download symbols and sectors:
       $ python xq_finance_download.py --market us
    2. Then download daily kline data:
       $ python xq_finance_download.py --market us
    3. Finally download financial reports:
       $ python xq_finance_download.py --market us
       
    For China/HK markets just replace 'us' with 'cn' or 'hk'
    
    To skip financial report download:
       $ python xq_finance_download.py --market us --skip-finance
    """
    # Equity and sector download
    xq_symbol_download(market)
    xq_sector_download(market)
    
    # Daily trade data
    symbols = us_equity_symbol_load(market)
    xq_kline_download(market, symbols)
    
    # Finance report (optional)
    if not skip_finance:
        xq_finance_download(market, symbols)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Download XQ finance data for specified market',
        epilog='Examples:\n'
               '  python xq_all_data_download.py --market us  # US market data\n'
               '  python xq_all_data_download.py --market cn  # China market data\n'
               '  python xq_all_data_download.py --market hk  # Hong Kong market data\n'
               '  python xq_all_data_download.py --market us --skip-finance  # Skip financial reports'
    )
    parser.add_argument('--market', required=True, choices=['us', 'cn', 'hk'], 
                       help='Market to download (us, cn, or hk)')
    parser.add_argument('--skip-finance', action='store_true',
                       help='Skip downloading financial reports')
    
    args = parser.parse_args()
    
    print(f"Starting download for {args.market.upper()} market...")
    download_market_data(args.market, args.skip_finance)
    print(f"Completed download for {args.market.upper()} market")