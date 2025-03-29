import pandas as pd
from quant_free.utils.us_equity_utils import us_dir1_store_csv  # Assuming this handles file storage

def us_equity_filter_and_store_by_symbol(market, df, file_name="trend"):
    """Filters a MultiIndex DataFrame by the second level index (symbol) 
    and stores each filtered DataFrame as a CSV file.

    Args:
        df: The input DataFrame with a MultiIndex. The second level 
           of the MultiIndex should be the 'symbol'.
        output_dir: The directory to store the CSV files. Defaults to "equity".
    """

    if not isinstance(df.index, pd.MultiIndex):
        raise TypeError("DataFrame index must be a MultiIndex.")

    if df.index.nlevels != 2:
      raise ValueError("The MultiIndex of the input DataFrame must be a second level.")

    if 'ticker' not in df.index.names:
        raise ValueError("The second level of the MultiIndex must be named 'symbol'.")


    for symbol in df.index.get_level_values('ticker').unique():
      
        filtered_df = df.xs(symbol, level='ticker')

        # Remove the symbol index
        # filtered_df = filtered_df.reset_index(level=0, drop=False) #Keep datetime as index
        # filtered_df.index.name = "datetime" #rename index to date

        us_dir1_store_csv(market, dir0 = 'equity', dir1 = symbol, filename = file_name + '.csv', data = filtered_df)


# Example usage (assuming your DataFrame is called 'my_multiindex_df'):
# filter_and_store_by_symbol(my_multiindex_df) 
