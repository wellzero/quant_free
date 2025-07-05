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
from typing import List, Tuple, Optional


from quant_free.utils.us_equity_symbol import *
from quant_free.utils.us_equity_utils import *
from quant_free.common.us_equity_common import *
from quant_free.utils.xq_parse_js import *
from quant_free.dataset.xq_daily_data import *
from quant_free.dataset.xq_finance_parser import *
from quant_free.dataset.xq_symbol import *
from quant_free.common.const import *
from quant_free.common.log import *

# Configure logging
logger = logging.getLogger(__name__)


"""Constants for financial column names by market"""
FinancialColumns = {
    'cn': [
        'Net Profit',
        'Net Profit Attributable to Parent Company Shareholders',
        'Total Operating Revenue',
        'Operating Profit'
    ],
    'us': [
        'Net Profit Margin',
        'Net Income Attributable to Parent Company',
        'Operating Revenue',
        'Operating Income'
    ],
    'hk': [
        'Operating Revenue',
        'Income before Tax'
        'Net Income Attributable to Parent Company',
        'Operating Income',
        # 'Net Profit Margin',
    ]
}


class FinanceData:
    def __init__(self, market: str = 'us', provider: str = 'xq'):
        self.market = market
        self.provider = provider
        self.financial_columns = ['report_date'] + FinancialColumns[self.market]
        self.roll_window = 2 if market == "hk" else 4

    def _load_financial_data(self, symbol: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Load yearly and quarterly financial data for a symbol"""
        try:
            yearly = us_dir1_load_csv(
                market=self.market,
                dir0='equity',
                dir1=symbol,
                dir2=self.provider,
                filename='finance_yearly.csv'
            )
            
            quarterly = us_dir1_load_csv(
                market=self.market,
                dir0='equity',
                dir1=symbol,
                dir2=self.provider,
                filename='finance_quarterly.csv'
            )
            
            if yearly.empty or quarterly.empty:
                logger.warning(f"Empty financial data for {symbol}")
                return None, None
                
            return yearly, quarterly
            
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            return None, None


    def merge_with_trade_data(self, symbol: str, finance_data: pd.DataFrame) -> pd.DataFrame:
        """Merge financial data with daily trade data"""
        trade_data = daily_trade_load(
            market=self.market,
            equity='equity',
            symbol=symbol,
            dir_option=self.provider,
            file_name='daily.csv'
        )
        
        if trade_data.empty or finance_data.empty:
            logger.warning(f"Empty data for {symbol}")
            return pd.DataFrame()
            
        try:
            finance_data = finance_data.reset_index().set_index('report_date')
            finance_data.index = pd.to_datetime(finance_data.index, errors='coerce')
            trade_data.index = pd.to_datetime(trade_data.index, errors='coerce')
            
            return pd.merge_asof(
                trade_data.sort_index(),
                finance_data.sort_index(),
                left_index=True,
                right_index=True,
                direction='nearest'
            )
            
        except Exception as e:
            logger.error(f"Merge error for {symbol}: {e}")
            return pd.DataFrame()

    def _process_symbol_data(self, symbol: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Process financial data for a single symbol"""
        yearly, quarterly = self._load_financial_data(symbol)
        if yearly is None or quarterly is None:
            return None, None
            
        try:
            # Process yearly data
            yearly = yearly[self.financial_columns].copy()
            yearly = self.merge_with_trade_data(symbol, yearly)
            yearly['symbol'] = symbol
            yearly = yearly.reset_index().set_index(['symbol', 'date'])
            
            # Process quarterly data
            quarterly = quarterly[self.financial_columns].copy()
            quarterly.iloc[:,1:] = quarterly.iloc[:,1:].rolling(self.roll_window).sum()
            quarterly = self.merge_with_trade_data(symbol, quarterly)
            quarterly['symbol'] = symbol
            quarterly = quarterly.reset_index().set_index(['symbol', 'date'])
            
            return yearly, quarterly
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            return None, None

    def equity_finance_process(self, symbols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process financial data for multiple symbols"""
        combined_yearly = pd.DataFrame()
        combined_quarterly = pd.DataFrame()
        
        for symbol in symbols:
            logger.info(f"Processing {symbol}")
            yearly, quarterly = self._process_symbol_data(symbol)
            
            if yearly is not None and quarterly is not None:
                combined_yearly = pd.concat([combined_yearly, yearly])
                combined_quarterly = pd.concat([combined_quarterly, quarterly])
        
        try:
            yearly_grouped = combined_yearly.groupby('date').sum()
            quarterly_grouped = combined_quarterly.groupby('date').sum()
            
            return quarterly_grouped, yearly_grouped
            
        except Exception as e:
            logger.error(f"Error combining data: {e}")
            return pd.DataFrame(), pd.DataFrame()

# --- Main Execution Block ---

if __name__ == "__main__":
    # --- US Example ---
    print("---  cn finance data example ---")
    symbols=['SH600519', 'SZ000995', 'SH600199']
    # symbols=['SH600199']
    fd = FinanceData(market='cn')
    quarter_data, year_data = fd.equity_finance_process(symbols=symbols)
    if not quarter_data.empty:
        print(quarter_data.tail(10))
