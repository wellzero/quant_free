import os
import sys

from quant_free.dataset.us_equity_download import *
from quant_free.dataset.xq_trade_data import *
from quant_free.dataset.xq_symbol import *

if __name__ == "__main__":
    symbols = get_all_symbol(market = 'us')
    market = 'us'
    us_equity_daily_data_download(market, symbols)


