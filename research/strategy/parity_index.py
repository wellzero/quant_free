import pandas as pd
# import pandas_ta as ta
from quant_free.dataset.us_equity_load import *
from quant_free.utils.us_equity_utils import *
from quant_free.factor.price import *

from statsmodels.tsa.stattools import coint

def find_cointegrated_pairs(molecule: list, dataframe, threshold=0.05):
    pairs = []

    # for i in range(n):
    #     print(f"Main stock: {dataframe.columns[i]}")
    #     for j in range(i+1, n):
    for column in molecule:
      
      symbol0 = column[0]
      symbol1 = column[1]
      S1 = dataframe[symbol0]
      S2 = dataframe[symbol1]
# The coint function performs Engle-Granger cointegration test:
# 1. Regress S1 ~ S2 to get residuals ε = S1 - β*S2 - α
# 2. Test residuals ε for unit root using ADF test
# 3. Test statistic = ADF statistic from residuals
# Null hypothesis: No cointegration (residuals have unit root)

# The ADF test equation for residuals ε (from step 1) is:
# Δε_t = δ + βt + γε_{t-1} + α_1Δε_{t-2} + ... + α_{p}Δε_{t-p} + ε_t
# Test statistic: t-statistic of γ coefficient. 
# Null hypothesis: γ=0 (unit root exists → no cointegration)

      result = coint(S1, S2)
      p_value = result[1]
      print(f"Check stock: {symbol0} {symbol1} {p_value}")
      if p_value < threshold:
          pairs.append([symbol0, symbol1, p_value])
    
    return pairs

from quant_free.finml.utils.multiprocess import mp_pandas_obj


def find_cointegrated_pairs_multi_process(column_pairs, dataframe, num_threads: int = 12):

    paris= mp_pandas_obj(func=find_cointegrated_pairs,
                               pd_obj=('molecule', column_pairs),
                               dataframe=dataframe,
                               num_threads=num_threads,
                               )
    return paris

def find_index_parity(market = 'cn', start_date = '2014-01-29', end_date = '2024-01-29'):
    market='cn'
    equity='index'
    data_index = us_dir1_load_csv(
        market=market,
        dir0=equity,
        dir1='symbols',
        filename='symbols.csv',
        index_col = None
    )


    symbols = data_index['symbol'].values
    name = data_index['name'].values

    data_trade = equity_daily_data_load_within_range(market=market,
                                                    equity=equity,
                                                    dir_option = '',
                                                    symbols = symbols,
                                                    start_date = start_date,
                                                    end_date = end_date,
                                                    column_option = 'close')



    merged_data = pd.DataFrame(data_trade)
    merged_data = merged_data.dropna()

    constant_cols = [col for col in merged_data.columns 
                    if merged_data[col].nunique(dropna=False) == 1]
    merged_data = merged_data.drop(columns=constant_cols)


    from itertools import combinations
    column_pairs = [list(pair) for pair in combinations(merged_data.columns, 2)]
    print(f"pairs: {column_pairs}")

    # Find cointegrated pairs
    print(f"check the stock {merged_data.columns}")
    coint_pairs = find_cointegrated_pairs_multi_process(column_pairs, merged_data)
    
    # Save results to CSV
    # pls call us_dir1_store_csv to store coint_pairs AI!
    output_file = f"cointegrated_pairs_{start_date}_to_{end_date}.csv"
    pd.DataFrame(coint_pairs, columns=['Symbol1', 'Symbol2', 'P-Value']).to_csv(output_file, index=False)
    print(f"Saved cointegrated pairs to {output_file}")
    print("Cointegrated pairs:", coint_pairs)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Find cointegrated index pairs')
    parser.add_argument('--market', type=str, default='cn', help='Market to analyze (default: cn)')
    parser.add_argument('--start_date', type=str, required=True, help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end_date', type=str, required=True, help='End date in YYYY-MM-DD format')
    
    args = parser.parse_args()
    find_index_parity(market=args.market, start_date=args.start_date, end_date=args.end_date)

if __name__ == '__main__':
    main()
