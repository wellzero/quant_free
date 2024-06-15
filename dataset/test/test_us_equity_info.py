import os
import sys
from pathlib import Path
_this_dir = Path(__file__).parent.parent.parent
sys.path.append(str(_this_dir))
print(_this_dir)
from dataset.us_equity_download import *
from dataset.us_equity_load import *

if __name__ == "__main__":
    # us_equity_symbol_download()

    symbols = us_equity_symbol_load()
    # print(symbols.head(10))
    us_equity_info_data_download(symbols)