import os
import sys

from quant_free.dataset.us_equity_download import *
from quant_free.dataset.us_equity_load import *

if __name__ == "__main__":
    symbols = us_equity_symbol_load(market = 'us')
    market = 'us'
    us_equity_daily_data_download(market, symbols)


