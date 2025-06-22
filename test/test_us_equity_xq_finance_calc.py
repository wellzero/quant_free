
import os
import sys



from quant_free.factor.xq_finance import *
from quant_free.dataset.xq_finance_parser import *
from quant_free.dataset.xq_trade_data import *
from quant_free.dataset.xq_symbol import *


if __name__ == '__main__':
  # symbols = get_all_symbol(market = 'us')
  # symbols = ['AAPL', 'NVDA']
  #symbols = ['TEAM', 'IAU', 'DKILY', 'OLCLY']
  # symbols = ['WFC', 'AXP', 'BX', 'GS']
  # symbols = 'JPM'
  # symbols = 'AAPL'
  # data = xq_finance_data(symbols)
  # print(data)

  symbols = get_all_symbol(market = 'us')
  # symbols = ['AAPL', 'NVDA']
  # symbols = ['XOM', 'WMT', 'COST']
  # symbols = ['MS', 'NOW', 'BX', 'GS', 'SCHW']
  # symbols = ['MS', 'GS', 'SCHW', 'PGR', 'BDCZ']
  us_analysis_finance = xq_finance(symbols)
  finance_factors = us_analysis_finance.finance_factors_calc()

  # us_analysis_finance = finance_factors_calc(symbols)
  # finance_factors = us_analysis_finance.finance_factors_calc()
