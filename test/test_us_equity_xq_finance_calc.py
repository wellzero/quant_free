
import os
import sys
from pathlib import Path
_this_dir = Path(__file__).parent.parent
sys.path.append(str(_this_dir))
print(_this_dir)
from dataset.us_equity_xq_factor import *
from dataset.us_equity_xq_load import *
from dataset.us_equity_load import *


if __name__ == '__main__':
  # symbols = us_equity_symbol_load()
  # symbols = ['AAPL', 'NVDA']
  #symbols = ['TEAM', 'IAU', 'DKILY', 'OLCLY']
  # symbols = ['WFC', 'AXP', 'BX', 'GS']
  # symbols = 'JPM'
  # symbols = 'AAPL'
  # data = us_equity_xq_finance_data_load(symbols)
  # print(data)

  symbols = us_equity_symbol_load()
  # symbols = ['AAPL', 'NVDA']
  # symbols = ['XOM', 'WMT', 'COST']
  # symbols = ['WFC', 'AXP', 'BX', 'GS']
  us_analysis_finance = us_equity_xq_factor(symbols)
  finance_factors = us_analysis_finance.finance_factors_calc()

  # us_analysis_finance = us_equity_efinance_factor(symbols)
  # finance_factors = us_analysis_finance.finance_factors_calc()
