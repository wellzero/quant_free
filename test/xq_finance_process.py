import os
import sys

from quant_free.dataset.xq_finance_load import *

if __name__ == "__main__":

    # us_equity_symbol_download()

    xq_finance_process(market='cn')
    xq_finance_process(market='hk')
    xq_finance_process(market='us')
