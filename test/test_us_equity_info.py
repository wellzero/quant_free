import os
import sys



from quant_free.dataset.us_equity_download import *
from quant_free.dataset.xq_data_load import *

if __name__ == "__main__":
    # us_equity_symbol_download()
    market = 'us'

    symbols = symbol_load(market = 'us')
    symbols = ['TEAM']
    # print(symbols.head(10))
    us_equity_info_data_download(market, symbols)