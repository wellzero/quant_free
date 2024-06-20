
import os
import sys
from pathlib import Path
_this_dir = Path(__file__).parent.parent.parent
sys.path.append(str(_this_dir))
print(_this_dir)
from dataset.us_equity_finance_factor import *
from dataset.us_equity_load import *
from dataset.us_equity_download import *
from dataset.us_equity_load import *





if __name__ == '__main__':
  symbols = us_equity_symbol_load()
  # symbols = ['AAPL', 'NVDA']
  us_analysis_finance = us_equity_finance(symbols)
  finance_factors = us_analysis_finance.finance_factors_calc()


  # symbols = ['AAPL', 'NVDA']
  # # symbol_list = ['000408']
  # print("symbol list len ", len(symbols))
  # factors = ['roe', 'roa', 'profit_revenue', 'revenue_incr_rate', 'cash_incr_rate']
  # start_quater = '2020/Q1'
  # end_quater = '2024/Q2'
  # us_analysis_finance = us_equity_finance(symbols, factors,  start_quater, end_quater)
  # finance_factors = us_analysis_finance.finance_factors_all_stock()
  
  # analysis_path = us_equity_research_folder("finance")
  # csv_file = os.path.join(analysis_path, start_quater + '_' + end_quater + '_finance_factors.csv')
  # finance_factors.to_csv(csv_file)
  # finance_rank = us_analysis_finance.finance_factors_rank(finance_factors)
  
  # if finance_rank.empty:
  #   print("finance factors are None")
  # else:
  #   csv_file = os.path.join(analysis_path, start_quater + '_' + end_quater + '_finance_rank.csv')
  #   print('finance_rank write to ', csv_file)
  #   finance_rank.to_csv(csv_file, index=True)