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
    Market.US.value: 5,
    Market.HK.value: 4,
    Market.CN.value: 2,
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
        try:
            df = us_dir1_load_csv(self.market,
                 dir0 = 'equity',
                 dir1 = self.symbol,
                 dir2 = self.provider,
                 filename= statement_type)
            return df.loc[:, ~df.columns.str.startswith('subtitle')]
        except FileNotFoundError:
            logger.error(f"File not found: {statement_type}\
                         for {self.symbol} in {self.market} market.")
            raise

    def _rename_columns(self, df: pd.DataFrame, title_map: dict) -> pd.DataFrame:
        """Renames columns using the provided title_map."""
        title_cn = [title_map.get(key, key) for key in df.columns]
        df.columns = title_cn
        return df

    def _standardize_report_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes report names based on the market."""
        if self.market == 'us':
            df['report_name'] = df['report_name'].str.replace('年FY', 'Q4')
            df['report_name'] = df['report_name'].str.replace('年Q9', 'Q3')
            df['report_name'] = df['report_name'].str.replace('年Q6', 'Q2')
            df['report_name'] = df['report_name'].str.replace('年Q1', 'Q1')
        elif self.market in ['cn', 'hk']:
            df['report_name'] = df['report_name'].str.replace('年报', 'Q4')
            df['report_name'] = df['report_name'].str.replace('三季报', 'Q3')
            df['report_name'] = df['report_name'].str.replace('半年报', 'Q2')
            df['report_name'] = df['report_name'].str.replace('中报', 'Q2')
            df['report_name'] = df['report_name'].str.replace('一季报', 'Q1')
        return df

    def _convert_string_lists(self, df: pd.DataFrame) -> pd.DataFrame:
        """Converts string representations of lists to actual lists."""
        def str_to_list(cell):
            try:
                return ast.literal_eval(cell)
            except (ValueError, SyntaxError):
                return cell
        return df.applymap(str_to_list)

    def _extract_first_value(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts the first value from lists in the DataFrame."""
        def get_first_value(cell):
            if isinstance(cell, list) and len(cell) > 1:
                return cell[0]
            return cell
        return df.applymap(get_first_value)

    def _clean_and_sort(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans and sorts the DataFrame."""
        df.sort_index(inplace=True)
        df.replace({'--': 0, '_': 0, 'None': 0}, inplace=True)
        df.fillna(0, inplace=True)
        return df

    def _get_quarter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates quarter-over-quarter differences for financial statements.
        
        Args:
            df (pd.DataFrame): Input DataFrame with financial data
            
        Returns:
            pd.DataFrame: DataFrame with quarter differences calculated
            
        Notes:
            - For US market: Q1 values are kept as-is, Q2-Q4 show quarter differences
            - For HK market: Q1-Q2 values are kept as-is, Q3-Q4 show quarter differences
            - For CN market: Same as HK market
        """
        try:
            
            # Determine which quarters to keep as-is vs calculate differences
            skip_quarters = {
                'us': 'Q1',    # US: Only Q1 kept as-is
                'hk': 'Q2', # HK: Q2 kept as-is
                'cn': 'Q1'  # CN: Q1 kept as-is
            }.get(self.market, [1])  # Default to US behavior
            
            # Calculate quarter differences where needed
            for i in range(1, len(df)):
                if skip_quarters not in df.index[i]:
                    # Calculate difference from previous quarter
                    df.iloc[i, self.ignore_cols:] = (
                        df.iloc[i, self.ignore_cols:] - 
                        df.iloc[i-1, self.ignore_cols:]
                    )
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating quarter differences for {self.symbol}: {e}")
            return df  # Return original if error occurs

    def _adapt_statement(self, df: pd.DataFrame, title_map: dict) -> pd.DataFrame:
        """
        Adapts and cleans a financial statement DataFrame by renaming columns, 
        standardizing report names, and handling missing values.

        Args:
            df (pd.DataFrame): The input DataFrame containing financial statement data.
            title_map (dict): A dictionary mapping original column names to standardized names.

        Returns:
            pd.DataFrame: The adapted DataFrame with cleaned and standardized data.
        """
        df_adapted = df.copy()
        
        # Step 1: Rename columns
        df_adapted = self._rename_columns(df_adapted, title_map)
        
        # Step 2: Standardize report names based on market
        df_adapted = self._standardize_report_names(df_adapted)
        
        # Step 3: Set 'report_name' as index
        df_adapted.set_index('report_name', inplace=True)
        
        # Step 4: Convert string representations of lists to actual lists
        df_adapted = self._convert_string_lists(df_adapted)
        
        # Step 5: Extract the first value from lists
        df_adapted = self._extract_first_value(df_adapted)
        
        # Step 6: Sort by 'ctime' and clean data
        df_adapted = self._clean_and_sort(df_adapted)

        df_adapted = self._get_quarter(df_adapted)
        
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
            income = self._adapt_statement(income_raw,
                self.xq_names['incomes'])
            cash = self._adapt_statement(cash_raw,
                self.xq_names['cashes'])
            balance = self._adapt_statement(balance_raw,
                self.xq_names['balances'])
            indicators = self._adapt_statement(metrics_raw,
                self.xq_names['indicators'])

            # Combine
            data = pd.concat([df.loc[~df.index.duplicated(keep='first')] 
                for df in [income, cash, balance, indicators]],
                axis=1, join='outer')

            # Fill any NaN values that may result from the outer join
            data.fillna(0, inplace=True)

            data.sort_index(inplace=True)

            return data.loc[:, ~data.columns.duplicated()]
        except Exception as e:
            logger.error(f"Failed to process financials for {self.symbol}: {e}")
            return pd.DataFrame()

# --- Standalone Utility Functions (Kept for compatibility or other uses) ---

def xq_finance_load(market='us', symbol='AAPL'):
    """Main function to load financial data for a given symbol."""
    processor = FinancialDataProcessor(market, symbol)
    return processor.get_full_financials()

# --- Main Execution Block ---

if __name__ == "__main__":
    # --- US Example ---
    print("--- Loading US Data for JPM ---")
    us_data = xq_finance_load(market='us', symbol='JPM')
    if not us_data.empty:
        print(us_data.head())

    # --- CN Example ---
    print("\n--- Loading CN Data for SH600519 ---")
    cn_market = 'cn'
    cn_symbol = 'SH600519'
    cn_data = xq_finance_load(market=cn_market, symbol=cn_symbol)
    if not cn_data.empty:
        print(cn_data.head())

    # --- CN Example ---
    print("\n--- Loading HK Data for SH600519 ---")
    hk_market = 'hk'
    hk_symbol = '02882'
    hk_data = xq_finance_load(market=hk_market, symbol=hk_symbol)
    if not hk_data.empty:
        print(hk_data.head())