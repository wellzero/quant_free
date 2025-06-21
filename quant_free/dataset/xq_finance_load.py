import os
import pandas as pd
import ast
import warnings
import logging
from enum import Enum

import multitasking
from tqdm.auto import tqdm
from typing import List


from quant_free.utils.us_equity_symbol import *
from quant_free.utils.us_equity_utils import *
from quant_free.common.us_equity_common import *
from quant_free.utils.xq_parse_js import *
from quant_free.dataset.xq_data_load import *

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
    Market.US.value: 6,
    Market.HK.value: 5,
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
        try:
            df = us_dir1_load_csv(self.market,
                 dir0 = 'equity',
                 dir1 = self.symbol,
                 dir2 = self.provider,
                 index_col = 'report_name',
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
        """
        Standardizes report names based on the market.

        Args:
            df (pd.DataFrame): Input DataFrame with report names in the index.

        Returns:
            pd.DataFrame: DataFrame with standardized report names in the index.
        """
        # Define market-specific replacement rules
        replacements = {
            'us': {
                '年FY': 'Q4',
                '年Q9': 'Q3',
                '年Q6': 'Q2',
                '年Q1': 'Q1'
            },
            'cn': {
                '年报': 'Q4',
                '三季报': 'Q3',
                '半年报': 'Q2',
                '中报': 'Q2',
                '一季报': 'Q1'
            },
            'hk': {
                '年报': 'Q4',
                '三季报': 'Q3',
                '半年报': 'Q2',
                '中报': 'Q2',
                '一季报': 'Q1'
            }
        }

        # Get the appropriate replacements for the current market
        market_replacements = replacements.get(self.market, {})

        # Apply replacements to the index
        for old, new in market_replacements.items():
            df.index = df.index.str.replace(old, new)

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
            for i in range(len(df)-1, 0, -1):
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
              
        # Step 4: Convert string representations of lists to actual lists
        df_adapted = self._convert_string_lists(df_adapted)
        
        # Step 5: Extract the first value from lists
        df_adapted = self._extract_first_value(df_adapted)
        
        # Step 6: Sort by 'ctime' and clean data
        df_adapted = self._clean_and_sort(df_adapted)
 
        return df_adapted.infer_objects()

    def _get_finance_data(self, statement_type: str) -> pd.DataFrame:
        """
        Retrieves and processes a specific type of financial statement.

        Args:
            statement_type (str): The type of financial statement to retrieve (e.g., 'income', 'cash', 'balance').

        Returns:
            pd.DataFrame: Processed DataFrame for the specified financial statement.
        """
        try:
            raw_data = self._load_raw_statement(statement_type)
            adapted_data = self._adapt_statement(raw_data, self.xq_names[statement_type])
            return adapted_data
        except Exception as e:
            logger.error(f"Error processing {statement_type} data for {self.symbol}: {e}")
            return pd.DataFrame()
        
    def load_daily_trade_data(self) -> pd.DataFrame:
        """
        Loads daily trade data (e.g., OHLCV) for a given symbol and market.

        Returns:
            pd.DataFrame: DataFrame containing daily trade data.
        """
        try:
            df = us_dir1_load_csv(
                self.market,
                dir0='equity',
                dir1=self.symbol,
                dir2=self.provider,
                index_col='timestamp',
                filename='daily'
            )
            df.sort_index(inplace=True)
            df.fillna(0, inplace=True)
            return df
        except FileNotFoundError:
            logger.error(f"Daily trade data not found for {self.symbol} in {self.market} market.")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading daily trade data for {self.symbol}: {e}")
            return pd.DataFrame()
        
    def merge_daily_data_to_finance(self, finance) -> pd.DataFrame:
        """
        Merges financial data with daily trade data based on the financial statement dates.

        Args:
            finance (pd.DataFrame): DataFrame containing financial statements.
            trade_data (pd.DataFrame): DataFrame containing daily trade data.

        Returns:
            pd.DataFrame: Merged DataFrame with financial data aligned to trade dates.
        """
        trade_data = self.load_daily_trade_data()
        if trade_data.empty or finance.empty:
            logger.warning(f"Trade data or financial data is empty for {self.symbol}.")
            return pd.DataFrame()
        
        if 'report_name' not in finance.index.names:
            logger.error(f"'report_name' index not found in financial data for {self.symbol}.")
            return pd.DataFrame()
        
        if 'timestamp' not in trade_data.index.names:
            logger.error(f"'timestamp' index not found in trade data for {self.symbol}.")
            return pd.DataFrame()

        try:
            # Reset the index and set 'report_date' as the new index
            finance = finance.reset_index().set_index('report_date')

            # Ensure 'ctime' is in datetime format
            finance.index = pd.to_datetime(finance.index, errors='coerce')
            trade_data.index = pd.to_datetime(trade_data.index, errors='coerce')

            # Merge on the index (dates)
            merged_data = pd.merge_asof(
                finance.sort_index(),
                trade_data.sort_index(),
                left_index=True,
                right_index=True,
                direction='backward'
            )

            merged_data = merged_data.reset_index().set_index('report_name')

            return merged_data
        except Exception as e:
            logger.error(f"Error merging financial and trade data for {self.symbol}: {e}")
            return pd.DataFrame()

    def get_full_financials(self) -> pd.DataFrame:
        """Loads, processes, and combines all financial statements."""
        try:
            # Load
            income = self._get_finance_data('income')
            cash = self._get_finance_data('cash')
            balance = self._get_finance_data('balance')
            metrics = self._get_finance_data('metrics')

            if income.empty or cash.empty or balance.empty or metrics.empty:
                logger.warning(f"One or more financial statements are empty for {self.symbol}.")
                return pd.DataFrame()
            
            # Combine
            data = pd.concat([df.loc[~df.index.duplicated(keep='first')] 
                for df in [income, cash, balance, metrics]],
                axis=1, join='outer')

            # Remove duplicate columns with the same name
            data = data.loc[:, ~data.columns.duplicated()]

            # Fill any NaN values that may result from the outer join
            data.fillna(0, inplace=True)

            data.sort_index(inplace=True)

            data_yearly = data.copy()    

            data_quarterly = self._get_quarter(data)

            data_quarterly = self.merge_daily_data_to_finance(data_quarterly)

            return data_yearly, data_quarterly
        except Exception as e:
            logger.error(f"Failed to process financials for {self.symbol}: {e}")
            return pd.DataFrame()


# --- Standalone Utility Functions (Kept for compatibility or other uses) ---
def xq_finance_load(market='us', symbol='AAPL'):
    """Main function to load financial data for a given symbol."""

    logger.info(f"Loading financial data for {symbol} in {market} market...")

    processor = FinancialDataProcessor(market, symbol)

    data_yearly, data_quarterly = processor.get_full_financials()

    us_dir1_store_csv(
        market=market,
        dir0='equity',
        dir1=symbol,
        dir2='xq',
        filename='finance_yearly.csv',
        data=data_yearly,
        # index=False
    )

    us_dir1_store_csv(
        market=market,
        dir0='equity',
        dir1=symbol,
        dir2='xq',
        filename='finance_quarterly.csv',
        data=data_quarterly,
        # index=False
    )

    return data_quarterly


def parallel_calc(market='us', symbols = ['AAPL', 'GOOGL', 'MSFT']):

    @multitasking.task
    def start(symbol: str):
        xq_finance_load(market=market, symbol=symbol)
        series.append(symbol)
        pbar.update()
        pbar.set_description(f'Processing => {symbol}')

    series: List[pd.Series] = []
    pbar = tqdm(total=len(symbols))
    for symbol in symbols:
        start(symbol)
    multitasking.wait_for_tasks()


def xq_finance_process(market='us'):
    """
    Processes financial data for a given symbol and returns a DataFrame.
    
    Args:
        market (str): The market type, e.g., 'us', 'cn', or 'hk'.
        symbol (str): The stock symbol to process.
        
    Returns:
        pd.DataFrame: Processed financial data.
    """
    symbols = symbol_load(market)

    parallel_calc(market, symbols)


# --- Main Execution Block ---

if __name__ == "__main__":
    # --- US Example ---
    print("--- Loading US Data for JPM ---")
    us_data = xq_finance_load(market='us', symbol='JPM')
    if not us_data.empty:
        print(us_data.tail(10))

    # --- CN Example ---
    print("\n--- Loading CN Data for SH600519 ---")
    cn_market = 'cn'
    cn_symbol = 'SH600519'
    cn_data = xq_finance_load(market=cn_market, symbol=cn_symbol)
    if not cn_data.empty:
        print(cn_data.tail(10))

    # --- CN Example ---
    hk_market = 'hk'
    hk_symbol = '02882'
    print(f"\n--- Loading HK Data for {hk_symbol} ---")
    hk_data = xq_finance_load(market=hk_market, symbol=hk_symbol)
    if not hk_data.empty:
        print(hk_data.tail(10))