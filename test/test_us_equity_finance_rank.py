
import os
import sys
from pathlib import Path
_this_dir = Path(__file__).parent.parent
sys.path.append(str(_this_dir))
print(_this_dir)
from dataset.us_equity_finance_factor import *
from dataset.us_equity_load import *


if __name__ == '__main__':
  symbols = us_equity_symbol_load()
  symbols = ['AAPL', 'NVDA', 'MSFT','GOOGL','AMZN','META','TSM','LLY']
  # symbols = ['AAPL', 'NVDA']
  #symbols = ['TEAM', 'IAU', 'DKILY', 'OLCLY']
  # symbols = ['WFC', 'AXP', 'BX', 'GS']
  # us_analysis_finance = us_equity_finance(symbols)
  # finance_factors = us_analysis_finance.finance_factors_calc()


  # symbols = ['AAPL', 'NVDA', 'MSFT', 'AMD', 'GOOG', 'TSLA', 'AVGO', 'NXPI', 'ADI', 'QCOM', 'TSM' 'ARM']
  # symbol_list = ['000408']
  print("symbol list len ", len(symbols))
  factors = ['roe', 'roa', 'profit_revenue', 'revenue_increase_q2q_rate', 'cash_increase_q2q_rate']
  start_time = '2022-01-01'
  end_time = '2024-06-01'
  us_analysis_finance = us_equity_finance(symbols, factors,  start_time, end_time)
  result = us_analysis_finance.finance_factors_rank()
  print(result)
 
