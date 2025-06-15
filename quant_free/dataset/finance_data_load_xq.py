import os
import pandas as pd
import ast
import warnings
import logging
from enum import Enum


from quant_free.utils.us_equity_symbol import *
from quant_free.utils.us_equity_utils import *
from quant_free.common.us_equity_common import *
from quant_free.utils.xq_parse_js import *

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# --- Constants ---
class Market(Enum):
    US = 'us'
    HK = 'hk'
    CN = 'cn'

MARKET_IGNORE_COLUMNS = {
    Market.US.value: 7,
    Market.HK.value: 6,
    Market.CN.value: 3,
}

# --- Core Financial Data Loading Class ---

class FinancialDataProcessor:
    """
    Handles loading, processing, and transforming financial data from XQ.
    """
    def __init__(self, market: str, symbol: str, provider: str = "xq"):
        if market not in [m.value for m in Market]:
            raise ValueError(f"Unsupported market: {market}")
        self.market = market
        self.symbol = symbol
        self.provider = provider
        self.ignore_cols = MARKET_IGNORE_COLUMNS.get(self.market, 0)
        self.xq_names = xq_js_to_dict(self.market)

    def _load_raw_statement(self, statement_type: str) -> pd.DataFrame:
        """Loads a single raw financial statement CSV."""
        equity_folder = equity_sub_folder(self.market, symbol=self.symbol, sub_dir=self.provider)
        file_path = Path(equity_folder) / f"{statement_type}.csv"
        logger.info(f"Loading {file_path} for {self.symbol}")
        try:
            df = pd.read_csv(file_path)
            return df.loc[:, ~df.columns.str.startswith('subtitle')]
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise

    def _adapt_statement(self, df: pd.DataFrame, title_map: dict) -> pd.DataFrame:
        """Adapts and cleans a financial statement DataFrame."""
        df_adapted = df.copy()
        
        # Rename columns
        title_cn = [title_map.get(key, key) for key in df_adapted.columns[self.ignore_cols:]]
        df_adapted.columns = list(df_adapted.columns[:self.ignore_cols]) + title_cn

        df_adapted['report_name'] = df_adapted['report_name'].str.replace('年', '/')
        df_adapted.set_index('report_name', inplace=True)

        # Convert string representations of lists to actual lists
        def str_to_list(cell): 
            try:
                return ast.literal_eval(cell)
            except (ValueError, SyntaxError):
                return cell
        df_adapted = df_adapted.applymap(str_to_list)

        # Extract the first value from lists
        def get_first_value(cell):
            if isinstance(cell, list) and len(cell) > 1:
                return cell[0]
            return cell
        df_adapted = df_adapted.applymap(get_first_value)

        df_adapted = df_adapted.sort_values(by='report_date', ascending=True)
        df_adapted.replace({'--': 0, '_': 0, 'None': 0}, inplace=True)
        df_adapted.fillna(0, inplace=True)
        return df_adapted.infer_objects()

    def get_full_financials(self) -> pd.DataFrame:
        """Loads, processes, and combines all financial statements."""
        try:
            # Load
            income_raw = self._load_raw_statement('income')
            cash_raw = self._load_raw_statement('cash')
            balance_raw = self._load_raw_statement('balance')
            metrics_raw = self._load_raw_statement('metrics')

            # Adapt
            income = self._adapt_statement(income_raw, self.xq_names['incomes'])
            cash = self._adapt_statement(cash_raw, self.xq_names['cashes'])
            balance = self._adapt_statement(balance_raw, self.xq_names['balances'])
            indicators = self._adapt_statement(metrics_raw, self.xq_names['indicators'])

            # Combine
            data = pd.concat([df.loc[~df.index.duplicated(keep='first')] for df in [income, cash, balance, indicators]], axis=1, join='outer')

            # Fill any NaN values that may result from the outer join
            data.fillna(0, inplace=True)

            return data.loc[:, ~data.columns.duplicated()]
        except Exception as e:
            logger.error(f"Failed to process financials for {self.symbol}: {e}")
            return pd.DataFrame()

# --- Standalone Utility Functions (Kept for compatibility or other uses) ---

def finance_data_load_xq(market='us', symbol='AAPL'):
    """Main function to load financial data for a given symbol."""
    processor = FinancialDataProcessor(market, symbol)
    return processor.get_full_financials()

# --- Main Execution Block ---

if __name__ == "__main__":
    # --- US Example ---
    print("--- Loading US Data for JPM ---")
    us_data = finance_data_load_xq(market='us', symbol='JPM')
    if not us_data.empty:
        print(us_data.head())

    # --- CN Example ---
    print("\n--- Loading CN Data for SH600519 ---")
    cn_market = 'cn'
    cn_symbol = 'SH600519'
    cn_data = finance_data_load_xq(market=cn_market, symbol=cn_symbol)
    if not cn_data.empty:
        print(cn_data.head())

    # --- CN Example ---
    print("\n--- Loading HK Data for SH600519 ---")
    hk_market = 'hk'
    hk_symbol = '02882'
    hk_data = finance_data_load_xq(market=hk_market, symbol=hk_symbol)
    if not hk_data.empty:
        print(hk_data.head())