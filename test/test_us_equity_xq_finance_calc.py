
import os
import sys



from quant_free.factor.xq_finance import *
from quant_free.dataset.xq_finance_load import *
from quant_free.dataset.xq_data_load import *


if __name__ == '__main__':
  # symbols = symbol_load(market = 'us')
  # symbols = ['AAPL', 'NVDA']
  #symbols = ['TEAM', 'IAU', 'DKILY', 'OLCLY']
  # symbols = ['WFC', 'AXP', 'BX', 'GS']
  # symbols = 'JPM'
  # symbols = 'AAPL'
  # data = xq_finance_load(symbols)
  # print(data)

  symbols = symbol_load(market = 'us')
  # symbols = ['AAPL', 'NVDA']
  # symbols = ['XOM', 'WMT', 'COST']
  # symbols = ['MS', 'NOW', 'BX', 'GS', 'SCHW']
  # symbols = ['MS', 'GS', 'SCHW', 'PGR', 'BDCZ']
  us_analysis_finance = xq_finance(symbols)
  finance_factors = us_analysis_finance.finance_factors_calc()

  # us_analysis_finance = finance_factors_calc(symbols)
  # finance_factors = us_analysis_finance.finance_factors_calc()
