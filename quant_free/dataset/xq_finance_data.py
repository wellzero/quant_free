import os
import pandas as pd
import ast
import warnings
import logging
from enum import Enum

import multitasking
from pkg_resources import _provider_factories
from tqdm.auto import tqdm
from typing import List


from quant_free.utils.us_equity_symbol import *
from quant_free.utils.us_equity_utils import *
from quant_free.common.us_equity_common import *
from quant_free.utils.xq_parse_js import *
from quant_free.dataset.xq_trade_data import *
from quant_free.dataset.xq_symbol import *
from quant_free.common.const import *
from quant_free.common.log import *

# Configure logging
logger = logging.getLogger(__name__)
class FinanceData:
    def __init__(self, market: str, provider: str = 'xq'):
        self.market = market
        self.provider = provider

    def _load_data(self, symbol: str) -> pd.DataFrame:
        """
        Load financial data for a given symbol and provider.
        """
        finance_yearly = us_dir1_load_csv(
            market=self.market,
            dir0='equity',
            dir1=symbol,
            dir2=self.provider,
            filename='finance_yearly.csv'
        )

        finance_quarterly = us_dir1_load_csv(
            market=self.market,
            dir0='equity',
            dir1=symbol,
            dir2=self.provider,
            filename='finance_quarterly.csv'
        )

        if finance_yearly.empty and finance_quarterly.empty:
            logger.warning(f"No financial data found for {symbol} in {self.market} market.")
            return None
        
        return finance_yearly, finance_quarterly
    
    def equity_finance_process(self, sector: list) -> pd.DataFrame:
        """
        Process financial data for a given symbol.
        """
        finance_yearly, finance_quarterly = self._load_data(symbol)
        
        if finance_yearly is None or finance_quarterly is None:
            return pd.DataFrame()

        # Combine yearly and quarterly data
        finance_data = pd.concat([finance_yearly, finance_quarterly], axis=0)
        finance_data.set_index('date', inplace=True)
        finance_data.sort_index(inplace=True)

        return finance_data




# --- Main Execution Block ---

if __name__ == "__main__":
    # --- US Example ---
    print("--- Loading US Data for JPM ---")
    us_data = xq_finance_data(market='us', symbol='JPM')
    if not us_data.empty:
        print(us_data.tail(10))

    # --- CN Example ---
    print("\n--- Loading CN Data for SH600519 ---")
    cn_market = 'cn'
    cn_symbol = 'SH600519'
    cn_data = xq_finance_data(market=cn_market, symbol=cn_symbol)
    if not cn_data.empty:
        print(cn_data.tail(10))

    # --- CN Example ---
    hk_market = 'hk'
    hk_symbol = '02882'
    print(f"\n--- Loading HK Data for {hk_symbol} ---")
    hk_data = xq_finance_data(market=hk_market, symbol=hk_symbol)
    if not hk_data.empty:
        print(hk_data.tail(10))