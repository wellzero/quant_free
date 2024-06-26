
import os
import sys
from pathlib import Path
_this_dir = Path(__file__).parent.parent
sys.path.append(str(_this_dir))
print(_this_dir)
from dataset.us_equity_finance_factor import *
from dataset.us_equity_xq_load import *


if __name__ == '__main__':
  # symbols = us_equity_symbol_load()
  # symbols = ['AAPL', 'NVDA']
  #symbols = ['TEAM', 'IAU', 'DKILY', 'OLCLY']
  # symbols = ['WFC', 'AXP', 'BX', 'GS']
  symbols = 'JPM'
  symbols = 'AAPL'
  data = us_equity_xq_finance_data_load(symbols)
  print(data)


  # us_analysis_finance = us_equity_finance(symbols)
  # finance_factors = us_analysis_finance.finance_factors_calc()
