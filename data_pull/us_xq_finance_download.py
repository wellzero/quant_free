import os
import sys


from quant_free.dataset.xq_data_download import *
from quant_free.dataset.xq_trade_data import *
from quant_free.dataset.xq_symbol import *

from quant_free.dataset.us_equity_download import *

if __name__ == "__main__":

    # us_equity_symbol_download()

    symbols = ['.IXIC', '.DJI', '.INX']
    market = 'us'
    xq_kline_download(market, symbols)

    # equity and sector download
    xq_symbol_download(market)

    xq_sector_download(market)

    # daily trade 
    symbols = get_all_symbol(market = market)
    xq_kline_download(market, symbols)

    # finance report
    xq_finance_download(market, symbols)
