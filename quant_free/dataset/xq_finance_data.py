from operator import index
import os
from cvxpy import outer
import pandas as pd
import ast
import warnings
import logging
from enum import Enum

import multitasking
# from pkg_resources import _provider_factories
from tqdm.auto import tqdm
from typing import List


from quant_free.utils.us_equity_symbol import *
from quant_free.utils.us_equity_utils import *
from quant_free.common.us_equity_common import *
from quant_free.utils.xq_parse_js import *
from quant_free.dataset.xq_trade_data import *
from quant_free.dataset.xq_finance_parser import *
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
    
    def equity_finance_process(self, symbols: list) -> pd.DataFrame:
        """
        Process financial data for a given symbol.
        """

        financial_column = ['Net Profit',
                            'Net Profit Attributable to Parent Company Shareholders',
                            'Total Operating Revenue',
                            'Operating Profit']
        finance_yearly_column = ['market_capital']

        finance_yearly = pd.DataFrame()
        finance_quarterly = pd.DataFrame()
        combined_data_yearly = pd.DataFrame()
        combined_data_quarterly = pd.DataFrame()

        for symbol in symbols:

            finance_yearly, finance_quarterly = self._load_data(symbol)
            # finance_yearly, finance_quarterly = xq_finance_data(market=self.market, symbol=symbol)

            if finance_yearly is not None and finance_quarterly is not None:
                # Select relevant columns
                yearly_selected = finance_yearly[finance_yearly_column].copy()  # Create a copy to avoid the warning
                yearly_selected.loc[:, 'symbol'] = symbol  # Use .loc for assignment
                yearly_selected = yearly_selected.reset_index()  # Convert index to column
                yearly_selected.set_index(['symbol', 'report_name'], inplace=True)

                quarterly_selected = finance_quarterly[financial_column].copy()  # Create a copy to avoid the warning
                quarterly_selected.loc[:, 'symbol'] = symbol  # Use .loc for assignment
                quarterly_selected = quarterly_selected.reset_index()  # Convert index to column
                quarterly_selected.set_index(['symbol', 'report_name'], inplace=True)

                combined_data_yearly = pd.concat([combined_data_yearly, yearly_selected])
                combined_data_quarterly = pd.concat([combined_data_quarterly, quarterly_selected])
            else:
                logger.warning(f"Skipping {symbol} due to missing data.")

        df_yearly   = combined_data_yearly.groupby('report_name').sum()
        df_quarterly = combined_data_quarterly.groupby('report_name').sum()
        
        df_quarterly[financial_column] = df_quarterly[financial_column].rolling(4).sum()

        return df_quarterly, df_yearly



# --- Main Execution Block ---

if __name__ == "__main__":
    # --- US Example ---
    print("---  cn finance data example ---")
    symbols=['SH600519', 'SZ000995', 'SH600199']
    # symbols=['SH600199']
    fd = FinanceData(market='cn')
    data = fd.equity_finance_process(symbols=symbols)
    if not data.empty:
        print(data.tail(10))
