import os
import sys
from pathlib import Path

from quant_free.dataset.data_download_xq import *
from quant_free.dataset.us_equity_load import *

from quant_free.dataset.us_equity_download import *

if __name__ == "__main__":

    # us_equity_symbol_download()

    symbols = ['.IXIC', '.DJI', '.INX']
    market = 'us'
    equity_xq_daily_data_download(market, symbols)

    # equity and sector download
    equity_xq_symbol_download(market)

    equity_xq_sector_download(market)

    # daily trade 
    symbols = us_equity_symbol_load(market = market)
    equity_xq_daily_data_download(market, symbols)

    # finance report
    equity_xq_finance_data_download(market, symbols)
