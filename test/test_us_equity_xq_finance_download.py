import os
import sys
from pathlib import Path
_this_dir = Path(__file__).parent.parent
sys.path.append(str(_this_dir))
from quant_free.dataset.us_equity_xq_download import *
from quant_free.dataset.us_equity_load import *

if __name__ == "__main__":
    # us_equity_symbol_download()

    symbols = us_equity_symbol_load()
    # symbols = ['TEAM']
    # print(symbols.head(10))

    # finance report
    #us_equity_xq_finance_data_download(symbols)

    # equity and sector download
    us_equity_xq_euquity_data_download()
    us_equity_xq_sector_data_download()

    # daily trade 
    us_equity_xq_daily_data_download(symbols)
