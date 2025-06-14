import os
import sys

from quant_free.dataset.data_download_xq import *
from quant_free.dataset.us_equity_load import *

from quant_free.dataset.us_equity_download import *

if __name__ == "__main__":

    equity_xq_fund_download(market='cn')
