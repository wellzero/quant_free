
import os
import sys



from quant_free.dataset.finance_factors_calc import *
from quant_free.dataset.xq_daily_data import *
from quant_free.dataset.xq_symbol import *


if __name__ == '__main__':
  symbols = get_all_symbol(market = 'us')
  symbols = ['AAPL', 'NVDA', 'MSFT','GOOGL','AMZN','META','TSM','LLY']
  # symbols = ['AAPL', 'NVDA']
  #symbols = ['TEAM', 'IAU', 'DKILY', 'OLCLY']
  # symbols = ['WFC', 'AXP', 'BX', 'GS']
  # us_analysis_finance = finance_factors_calc(symbols)
  # finance_factors = us_analysis_finance.finance_factors_calc()


  # symbols = ['AAPL', 'NVDA', 'MSFT', 'AMD', 'GOOG', 'TSLA', 'AVGO', 'NXPI', 'ADI', 'QCOM', 'TSM' 'ARM']
  # symbol_list = ['000408']
  print("symbol list len ", len(symbols))
  factors = ['roe', 'roa', 'profit_revenue', 'revenue_increase_q2q_rate', 'cash_increase_q2q_rate']
  start_time = '2022-01-01'
  end_time = '2024-06-01'
  us_analysis_finance = finance_factors_calc(symbols, factors,  start_time, end_time)
  result = us_analysis_finance.finance_factors_rank()
  print(result)
 
