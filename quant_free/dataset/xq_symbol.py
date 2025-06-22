import os
import pandas as pd
# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from quant_free.utils.us_equity_symbol import *
from quant_free.utils.us_equity_utils import *
from quant_free.common.us_equity_common import *

def symbol_load(market = 'us'):
  df = us_dir1_load_csv(market,
                        dir0 = 'symbol',
                        dir1 = 'xq',
                        filename='equity_symbol.csv',
                        dtype = str)
  return df.loc[:, 'symbol'].values

def get_sectors(market = "us", dir_option = "xq"):
    
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

    return sectors

def get_sectors_equity(market = "us", sector = "AAPL", dir_option = "xq"):

    data_symbols = us_dir1_load_csv(
        market,
        dir0 = 'symbol',
        dir1 = dir_option,
        filename= sector)
    if (data_symbols.empty == False):
        symbols = data_symbols['symbol'].values
    return symbols