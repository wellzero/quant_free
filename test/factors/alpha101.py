import sys
from pathlib import Path
_this_dir = Path(__file__).parent.parent.parent
sys.path.append(str(_this_dir))

from quant_free.dataset.us_equity_load import *
from quant_free.factor.alpha_101 import *

if __name__ == "__main__":
    # us_equity_symbol_download()

    # symbols = us_equity_symbol_load()
    symbols = ['AAPL']
    
    alpha101 = Alpha101(symbols)
    alpha101.processing('AAPL')
    # print(symbols.head(10))

    # finance report
    # us_equity_xq_finance_data_download(symbols)

    # equity and sector download
