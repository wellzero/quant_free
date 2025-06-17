import os
import sys



from quant_free.dataset.us_equity_download import *
from quant_free.dataset.equity_load import *

if __name__ == "__main__":
    market = 'us'
    us_equity_symbol_download(market)

    # symbols = symbol_load(market = 'us')
    # print(symbols.head(10))
    
#    us_equity_daily_data_download(market, symbols)

    # us_equity_efinance_finance_data_download(symbols)
    # us_equity_efinance_finance_data_download(symbols)

    # us_equity_option_data_download(symbols = symbols)
