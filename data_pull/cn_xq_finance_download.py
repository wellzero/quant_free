import os
import sys


from quant_free.dataset.data_download_xq import *
from quant_free.dataset.us_equity_load import *

from quant_free.dataset.us_equity_download import *

if __name__ == "__main__":

    # us_equity_symbol_download()

    # symbols = ['.IXIC', '.DJI', '.INX']
    market = 'cn'
    # equity_xq_daily_data_download(market, symbols)

    # equity and sector download
    xq_symbol_download(market)

    xq_sector_download(market)

    # daily trade 
    symbols = us_equity_symbol_load(market)
    equity_xq_daily_data_download(market, symbols)

    # finance report
    xq_finance_download(market, symbols)
