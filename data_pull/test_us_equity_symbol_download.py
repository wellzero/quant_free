import os
import sys



from quant_free.dataset.us_equity_download import *
from quant_free.dataset.xq_daily_data import *
from quant_free.dataset.xq_symbol import *

if __name__ == "__main__":
    market = 'us'
    us_equity_symbol_download(market)

    # symbols = get_all_symbol(market = 'us')
    # print(symbols.head(10))
    
#    us_equity_daily_data_download(market, symbols)

    # us_equity_efinance_finance_data_download(symbols)
    # us_equity_efinance_finance_data_download(symbols)

    # us_equity_option_data_download(symbols = symbols)
