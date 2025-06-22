import os
import pandas as pd
# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from quant_free.utils.us_equity_symbol import *
from quant_free.utils.us_equity_utils import *
from quant_free.common.us_equity_common import *

def get_equity_sector_name(market = "us", symbol = "AAPL", dir_option = "xq"):
    
    sector_file = 'equity_sector.csv'
    df_sector = us_dir1_load_csv(
      market,
      dir0 = 'symbol',
      dir1 = dir_option,
      filename = sector_file)

    if dir_option == "xq":
      sectors = list(df_sector['name'].values)
    else:
      sectors = list(df_sector['Sector'].values)

    for sector in sectors:
      data_symbols = us_dir1_load_csv(
        market,
        dir0 = 'symbol',
        dir1 = dir_option,
        filename= sector +'.csv')
      if (data_symbols.empty == False):
        symbols = data_symbols['symbol'].values
        if symbol in symbols:
          return sector
    return None