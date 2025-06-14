
import os
import sys
from pathlib import Path
_this_dir = Path(__file__).parent.parent
sys.path.append(str(_this_dir))
print(_this_dir)
from quant_free.factor.xq_finance import *
from quant_free.dataset.finance_data_load_xq import *
from quant_free.dataset.us_equity_load import *


if __name__ == '__main__':
  # symbols = us_equity_symbol_load(market = 'us')
  # symbols = ['AAPL', 'NVDA']
  #symbols = ['TEAM', 'IAU', 'DKILY', 'OLCLY']
  # symbols = ['WFC', 'AXP', 'BX', 'GS']
  # symbols = 'JPM'
  # symbols = 'AAPL'
  # data = finance_data_load_xq(symbols)
  # print(data)

  symbols = us_equity_symbol_load(market = 'us')
  # symbols = ['AAPL', 'NVDA']
  # symbols = ['XOM', 'WMT', 'COST']
  # symbols = ['MS', 'NOW', 'BX', 'GS', 'SCHW']
  # symbols = ['MS', 'GS', 'SCHW', 'PGR', 'BDCZ']
  us_analysis_finance = xq_finance(symbols)
  finance_factors = us_analysis_finance.finance_factors_calc()

  # us_analysis_finance = finance_factors_calc(symbols)
  # finance_factors = us_analysis_finance.finance_factors_calc()
