import os
import pandas as pd
from quant_free.utils.us_equity_symbol import *
from quant_free.utils.us_equity_utils import *
from quant_free.common.us_equity_common import *
import efinance as ef

# Read directory from JSON file

def equity_xq_finance_data_download(market = 'us', symbols = ['AAPL'], provider="xq"):

  datacenter = ef.stock.us_finance_xq_getter(market)

  for symbol in symbols:
    try:
      # xq_symbol = datacenter.get_secucode("MMM")
      print(f"Downloading {symbol} finance data...")

      data = datacenter.get_us_finance_income(symbol = symbol)
      us_dir1_store_csv(market=market,
                        dir0='equity',
                        dir1=symbol,
                        dir2=provider,
                        filename='income',
                        data=data,
                        index=False)

      data = datacenter.get_us_finance_cash(symbol = symbol)
      us_dir1_store_csv(market=market,
                        dir0='equity',
                        dir1=symbol,
                        dir2=provider,
                        filename='cash',
                        data=data,
                        index=False)

      data = datacenter.get_us_finance_balance(symbol = symbol)
      us_dir1_store_csv(market=market,
                        dir0='equity',
                        dir1=symbol,
                        dir2=provider,
                        filename='balance',
                        data=data,
                        index=False)

      data = datacenter.get_us_finance_main_factor(symbol = symbol)
      us_dir1_store_csv(market=market,
                        dir0='equity',
                        dir1=symbol,
                        dir2=provider,
                        filename='metrics',
                        data=data,
                        index=False)
    except:
      print(f"function {__name__} error!!")

def equity_xq_daily_data_download(market = 'us', symbols = ['AAPL'], provider="xq"):
  datacenter_xq = ef.stock.us_finance_xq_getter()
  for symbol in symbols:
    try:
      
      print(f"Downloading {symbol} trade data...")
      data = datacenter_xq.get_us_finance_daily_trade(symbol = symbol)
      us_dir1_store_csv(market=market,
                        dir0='equity',
                        dir1=symbol,
                        dir2=provider,
                        filename='daily',
                        data=data,
                        index=False)
    except:
      print(f"function {__name__} {equity_xq_daily_data_download} error!!")

def equity_xq_symbol_download(market = 'us'):
  datacenter_xq = ef.stock.us_finance_xq_sector_getter(market)
  data = datacenter_xq.get_all_us_equity()
  us_dir1_store_csv(
       market,
       dir0 = 'symbol',
       dir1 = 'xq',
       filename='equity_symbol.csv',
       data = data)

  if market == 'us':
    print(f"downloading us_china")
    data = datacenter_xq.get_all_us_us_china_equity()
    us_dir1_store_csv(market,
                      dir0 = 'symbol',
                      dir1 = 'xq',
                      filename='us_china.csv',
                      data = data)

    print(f"downloading listed")
    data = datacenter_xq.get_all_us_listed_equity()
    us_dir1_store_csv(market,
                      dir0 = 'symbol',
                      dir1 = 'xq',
                      filename='listed.csv',
                      data = data)

    print(f"downloading us_star")
    data = datacenter_xq.get_all_us_star_equity()
    us_dir1_store_csv(market,
                      dir0 = 'symbol',
                      dir1 = 'xq',
                      filename='us_star.csv',
                      data = data)

def equity_xq_sector_download(market = 'us'):
  
  datacenter_xq = ef.stock.us_finance_xq_sector_getter(market)
  data = datacenter_xq.get_all_us_sector_name()
  us_dir1_store_csv(market,
                    dir0 = 'symbol',
                    dir1 = 'xq',
                    filename='equity_sector.csv',
                    data = data)

  for index, row in data.iterrows():
    print(f"downloading {row['name']}")
    data = datacenter_xq.get_all_us_equity(row['encode'])
    us_dir1_store_csv(market,
                      dir0 = 'symbol',
                      dir1 = 'xq',
                      filename=row['name'] + '.csv',
                      data = data)

def equity_xq_fund_download(market='us'):
    """Download data for each fund type."""
    datacenter_xq = ef.stock.us_finance_xq_getter(market)
    
    # Define fund types and their corresponding codes
    fund_types = {
        "分级基金": 11,
        "货币型": 12,
        "股票型": 13,
        "债券型": 14,
        "混合型": 15,
        "QDII基金": 16,
        "指数型基金": 17,
        "ETF": 18,
        "LOF": 19,
        "FOF": 20,
        "场外基金": 21,
    }

    # Download data for each fund type
    for fund_name, fund_code in fund_types.items():
        print(f"Downloading {fund_name} fund data...")
        try:
            data = datacenter_xq.get_cn_fund_list(fund_code)
            us_dir1_store_csv(
                market,
                dir0='symbol',
                dir1='xq',
                filename=f"fund_{fund_name}.csv",
                data=data
            )
            # Download daily trade data for each fund in the list
            for index, row in data.iterrows():
                fund_symbol = row['symbol']
                print(f"Downloading daily trade data for {fund_name} fund: {fund_symbol}")
                daily_data = datacenter_xq.get_us_finance_daily_trade(symbol=fund_symbol)
                us_dir1_store_csv(
                    market,
                    dir0='fund',
                    dir1='xq',
                    filename=f"{fund_symbol}.csv",
                    data=daily_data
                )

        except Exception as e:
            print(f"Error downloading {fund_name} fund data: {e}")