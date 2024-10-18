
from quant_free.common.us_equity_common import *

import pandas as pd
from datetime import datetime, timedelta

# def calculate_ratio_changes(symbol, target_date, days_before, days_after):
#     # Convert the input data to a pandas DataFrame

#     df = us_equity_data_load(symbol)
#     df.index = pd.to_datetime(df.index)

#     # Find the nearest date in the data frame
#     nearest_date = df.index.sub(target_date).abs().idxmin()

#     # Filter the data around the reference date
#     start_date = nearest_date - timedelta(days=days_before)
#     end_date = nearest_date + timedelta(days=days_after)
#     filtered_data = df[(df.index >= start_date) & (df.index <= end_date)]

#     # Calculate the ratio change in closing price between the first and last day
#     close_ratio_change = filtered_data['close'].iloc[-1] / filtered_data['close'].iloc[0] - 1

#     # Calculate the ratio change in volume between the first and last day
#     volume_ratio_change = filtered_data['volume'].iloc[-1] / filtered_data['volume'].iloc[0] - 1

#     return close_ratio_change, volume_ratio_change

def finance_calculate_ratio_changes(df_factor, days_before, days_after):
    # Iterate over each symbol in the multi-index
    trade_date = pd.to_datetime(us_equity_get_trade_dates())
    for symbol, symbol_data in df_factor.groupby(level='symbol'):
        # print("symbol: ", symbol)
        try:
            df = us_equity_data_load(symbol = symbol)
            df.index = pd.to_datetime(df.index)
            date_index = df.index
            # Iterate over each row in the symbol data
            for idx, row in symbol_data.iterrows():
                target_date = idx[0]

                # Find the index of the nearest date in the DatetimeIndex
                nearest_index = abs(date_index - target_date).argmin()

                # Calculate the start and end indices based on days_before and days_after
                start_index = max(0, nearest_index - days_before)
                end_index = min(len(date_index) - 1, nearest_index + days_after)

                # Get the beginning and ending dates for the specified range
                start_date = date_index[start_index]
                end_date = date_index[end_index]

                filtered_data = df[(df.index >= start_date) & (df.index <= end_date)]

                # Calculate the ratio change in closing price between the first and last day
                close_ratio_change = (filtered_data['close'].iloc[-1] / filtered_data['close'].iloc[0]) - 1

                # Calculate the ratio change in volume between the first and last day
                volume_ratio_change = (filtered_data['volume'].iloc[-1] / filtered_data['volume'].iloc[0]) - 1

                # Attach the ratio changes to the corresponding row in the original DataFrame
                df_factor.loc[idx, f'p_{days_before}D_{days_after}D'] = close_ratio_change * 100
                # df_factor.loc[idx, f'v_{days_before}D_{days_after}D'] = volume_ratio_change * 100
        except:
            print(f"price processing error {symbol}")

    return df_factor.round(2)