import os
import sys

from quant_free.dataset.xq_data_download import *
from quant_free.dataset.xq_trade_data import *
from quant_free.dataset.xq_symbol import *

from quant_free.dataset.us_equity_download import *

if __name__ == "__main__":

    xq_fund_download(market='cn')
