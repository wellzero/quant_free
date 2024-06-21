# coding:utf-8

# import jqdatasdk
from datetime import datetime
import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from dataset.us_equity_load import *

import pandas as pd

def convert_to_financial_datetime(period):
    year, quarter = period.split('/')
    quarter = quarter.upper()
    
    # Mapping for the specific quarters to the end of their respective periods
    quarter_end_mapping = {
        'Q1': '03-31',
        'Q2': '06-30',
        'Q3': '09-30',
        'Q4': '12-31',
        'Q6': '06-30',
        'Q9': '09-30',
        'FY': '12-31'
    }
    
    if quarter not in quarter_end_mapping:
        raise ValueError(f"Unknown period format: {period}")
    
    date = f'{year}-{quarter_end_mapping[quarter]}'
    
    return date

def generate_Q_dates(start_quarter = '2001/Q2', end_quarter = '2023/Q1'):
    start_year, quarter = start_quarter.split('/')
    end_year, quarter = end_quarter.split('/')
    # Define a list of years from 2001 to 2022
    years = list(range(int(start_year) - 1, int(end_year) + 1))

    # Define a dictionary to map quarter numbers to their corresponding string representations
    quarter_map = {
        1: "Q1",
        2: "Q6",
        3: "Q9",
        4: "QA"
    }

    dates = []

    # Loop through each year
    for year in years:

        # Loop through each quarter (reverse order to start with Q1)
        for quarter in range(1, 5, 1):
            # Append the quarter in the format "YYYY/QN" to the dates list
            dates.append(f"{year}/{quarter_map[quarter]}")

    dates_compare = pd.DataFrame(dates).replace('FY', 'QA', regex=True)
    if 'FY' in start_quarter:
      start_quarter = start_quarter.replace('FY', 'QA')
    if 'FY' in end_quarter:
      end_quarter = end_quarter.replace('FY', 'QA')
    dates_between_quarters = [date for date in dates_compare.values if start_quarter <= date <= end_quarter]
    result = pd.DataFrame(dates_between_quarters).replace('QA', 'FY', regex=True)

    return result[0]



class us_equity_finance:

  # def __init__(self, symbols, factors = ['ROE'], start_quarter = '2001/Q2', end_quarter = '2023/Q1') -> None:
  def __init__(self, symbols, factors = None, start_quarter = None, end_quarter = None) -> None:

    self.symbols = symbols

    self.symbol_list_process = []

    self.factors = factors

    if start_quarter is None or end_quarter is None:
      self.dates = None
      self.date_times = None
    else:
      self.dates = generate_Q_dates(start_quarter, end_quarter)
      self.date_times = self.dates.apply(lambda x:convert_to_financial_datetime(x)).values


    return
  
  
  def ab_ratio_calc(self, a,b,str):
    data = a/b
    # data[np.isinf(data)]=0
    # data[np.isnan(data)]=0
    data = data * 100
    dict_data = {str:data}
    pd_data = pd.DataFrame(dict_data)
    return pd_data
  def qoq_rate_calc(self, a,str):
    data = a.pct_change(3)
    data = data * 100
    data[np.isinf(data)]=0
    # data = np.append(data,[0,0,0,0])
    dict_data = {str:data}
    pd_data = pd.DataFrame(dict_data)
    return pd_data
  def abc_ratio_calc(self, a,b,c,str,op):
    if(op=='sub'):
      data = a/(b-c)
    data[np.isinf(data)]=0
    dict_data = {str:data}
    pd_data = pd.DataFrame(dict_data)
    return pd_data

  # def nearest_date(self, date_list, input_date):
  #   return min(date_list, key=lambda x: abs(x - input_date))
  
  def nearest_date(self, date_list, input_date):
    # input_datetime = input_date
    # Sort the date list in ascending order
    date_list_sorted = sorted(date_list)
    # Do a linear search to find the nearest date
    date_objects = [datetime.strptime(date, '%Y-%m-%d') for date in date_list_sorted]

    # Given date
    given_date = datetime.strptime(input_date, '%Y-%m-%d')

    # Find the latest date before or equal to the given date
    if given_date < date_objects[0]:
      latest_date = date_objects[0]
    else:
      latest_date = max(date for date in date_objects if date <= given_date)

    # Convert the latest date back to string format
    latest_date_str = latest_date.strftime('%Y-%m-%d')
    
    return latest_date_str
  

  def get_market_value(self, symbol, df = None):
    
    if self.date_times is None:
      df_dates = df.index
      df_date_times = [convert_to_financial_datetime(date) for date in df.index]
    else:
      df_dates = self.dates
      df_date_times = self.date_times

    daily_data = us_equity_daily_data_read_csv(symbol)['close']
    daily_trade_dates = daily_data.index

    dates = []
    for get_date in df_date_times:
      # print(get_date)
      date = self.nearest_date(daily_trade_dates, get_date)
      dates.append(date)

    shares = us_equity_common_shares_load([symbol])

    # trade_data_quarter = pd.concat([trade_data_quarter, res.loc[date]], axis = 1)
    trade_data_quarter = daily_data.loc[dates]
    trade_data_quarter.index = df_dates

    return trade_data_quarter * shares

  def finance_factors_one_stock(self, symbol):

    df_finance = us_equity_efinance_finance_data_load(symbol, dates = self.dates)

    # balance
    total_equity = df_finance['total_equity']
    total_assets = df_finance['total_assets']
    goodwell = df_finance['goodwill']
    total_liabilities = df_finance['total_liabilities'] #total_liabilities
    total_liabilities_and_equity = df_finance['total_liabilities_and_equity']
    intangible_assets = df_finance['intangible_assets']
    total_non_current_assets = df_finance['total_non_current_assets']
    # income
    operating_revenue = df_finance['operating_revenue']
    operating_cost = df_finance['operating_cost']
    income_tax = df_finance['income_tax']
    profit_before_tax_from_continuing_operations = df_finance['profit_before_tax_from_continuing_operations'] #EBIT
    research_and_development_expenses = df_finance['research_and_development_expenses'] # research_and_development_expenses
    # cash
    net_cash_flow_from_operating_activities = df_finance['net_cash_flow_from_operating_activities']
    net_profit = df_finance['net_profit']
    cash_and_cash_equivalents = df_finance['cash_and_cash_equivalents']
    depreciation_and_amortization = df_finance['depreciation_and_amortization']
    # depreciation_and_amortization = df_finance['depreciationForFixedAssets']
    cash_and_cash_equivalents_at_end_of_period = df_finance['cash_and_cash_equivalents_at_end_of_period']

    noncurrent_asset = intangible_assets + goodwell + total_non_current_assets
    EBITDA = profit_before_tax_from_continuing_operations + depreciation_and_amortization
    market_cap = self.get_market_value(symbol, df_finance)
    EV = market_cap + total_liabilities - cash_and_cash_equivalents_at_end_of_period

    ##### net_profit capacity
    #roe
    pd_data = pd.DataFrame({'roe': net_profit / total_equity}) * 100
    # roe = roe.rename(columns = ['roe'])
    #roa
    roa = self.ab_ratio_calc(net_profit,total_assets,'roa')
    pd_data = pd.concat([pd_data, roa], axis=1)
    #roi
    roi = self.ab_ratio_calc(profit_before_tax_from_continuing_operations-income_tax, total_liabilities_and_equity - cash_and_cash_equivalents,'roi')
    pd_data = pd.concat([pd_data, roi], axis=1)
    #profit_revenue
    profit_revenue = self.ab_ratio_calc(net_profit,operating_revenue,'profit_revenue')
    pd_data = pd.concat([pd_data, profit_revenue], axis=1)
    #gross_profit_revenue
    profit_revenue = self.ab_ratio_calc(operating_revenue - operating_cost, operating_revenue, 'gross_profit_revenue')
    pd_data = pd.concat([pd_data, profit_revenue], axis=1)
    #profit_cost
    profit_cost = self.ab_ratio_calc(net_profit,operating_cost,'profit_cost')
    pd_data = pd.concat([pd_data, profit_cost], axis=1)
    #stackholder total_equity increase
    equity_incr_rate = self.qoq_rate_calc(total_equity,'equity_incr_rate')
    pd_data = pd.concat([pd_data, equity_incr_rate], axis=1)

    ###grow capacity
    #operating_revenue
    revenue_increase_q2q_rate = self.qoq_rate_calc(operating_revenue,'revenue_increase_q2q_rate')
    pd_data = pd.concat([pd_data, revenue_increase_q2q_rate], axis=1)
    #profit
    profit_increase_q2q_rate = self.qoq_rate_calc(net_profit,'profit_increase_q2q_rate')
    pd_data = pd.concat([pd_data, profit_increase_q2q_rate], axis=1)
    #net_cash_flow_from_operating_activities
    cash_increase_q2q_rate = self.qoq_rate_calc(net_cash_flow_from_operating_activities,'cash_increase_q2q_rate')
    pd_data = pd.concat([pd_data, cash_increase_q2q_rate], axis=1)
    #total_assets
    asset_increase_q2q_rate = self.qoq_rate_calc(total_assets,'asset_increase_q2q_rate')
    pd_data = pd.concat([pd_data, asset_increase_q2q_rate], axis=1)
    #total_liabilities
    debt_increase_q2q_rate = self.qoq_rate_calc(total_liabilities,'debt_increase_q2q_rate')
    pd_data = pd.concat([pd_data, debt_increase_q2q_rate], axis=1)

    ###total_assets struct
    #debt_asset_ratio
    debt_asset_ratio = self.ab_ratio_calc(total_liabilities,total_assets,'debt_asset_ratio')
    debt_asset_ratio = debt_asset_ratio/100
    pd_data = pd.concat([pd_data, debt_asset_ratio], axis=1)
    #debt_equity_ratio
    debt_equity_ratio = self.ab_ratio_calc(total_liabilities,total_equity,'debt_equity_ratio')
    debt_equity_ratio = debt_equity_ratio/100
    pd_data = pd.concat([pd_data, debt_equity_ratio], axis=1)
    #debt_net_asset_ratio
    debt_net_asset_ratio = self.abc_ratio_calc(total_liabilities,total_equity,intangible_assets,'debt_net_asset_ratio','sub')
    pd_data = pd.concat([pd_data, debt_net_asset_ratio], axis=1)
    #revenue_asset_ratio
    revenue_asset_ratio = self.ab_ratio_calc(operating_revenue,total_assets,'revenue_asset_ratio')
    revenue_asset_ratio = revenue_asset_ratio
    pd_data = pd.concat([pd_data, revenue_asset_ratio], axis=1)
    #goodwell_equity_ratio
    goodwell_equity_ratio = self.ab_ratio_calc(goodwell,total_equity,'goodwell_equity_ratio')
    pd_data = pd.concat([pd_data, goodwell_equity_ratio], axis=1)

    ###CFO2EV
    CFO_EV_ratio = self.ab_ratio_calc(net_cash_flow_from_operating_activities,EV,'CFO2EV')
    pd_data = pd.concat([pd_data, CFO_EV_ratio], axis=1)
    ####EBITDA2ev
    EBITDA_EV_ratio = self.ab_ratio_calc(EBITDA,EV,'EDITDA2EV')
    pd_data = pd.concat([pd_data, EBITDA_EV_ratio], axis=1)
    ###BB2P
    divedends_market_cap_ratio = self.ab_ratio_calc(depreciation_and_amortization,market_cap,'BB2P')
    pd_data = pd.concat([pd_data, divedends_market_cap_ratio], axis=1)
    ###BB2EV
    divedends_EV_ratio = self.ab_ratio_calc(depreciation_and_amortization,EV,'BB2EV')
    pd_data = pd.concat([pd_data, divedends_EV_ratio], axis=1)
    #B2P
    B2P_ratio = self.ab_ratio_calc(total_equity,market_cap,'B2P')/100
    pd_data = pd.concat([pd_data, B2P_ratio], axis=1)
    #S2EV
    S2EV_ratio = self.ab_ratio_calc(operating_revenue,EV,'S2EV')
    pd_data = pd.concat([pd_data, S2EV_ratio], axis=1)
    #equity_asset_ratio
    OL = self.ab_ratio_calc(total_equity,total_assets,'OL')
    pd_data = pd.concat([pd_data, OL], axis=1)
    #NCO2A
    NCO2A = self.ab_ratio_calc(noncurrent_asset,total_assets,'NCO2A')
    pd_data = pd.concat([pd_data, NCO2A], axis=1)
    #E2EV
    E2EV = self.ab_ratio_calc(net_profit,EV,'E2EV')
    pd_data = pd.concat([pd_data, E2EV], axis=1)
    
    return pd_data

  def finance_factors_calc(self):

    finance_factors = pd.DataFrame()
    for symbol in self.symbols:
      # print("calc symbol ", symbol)
      try:
        df = self.finance_factors_one_stock(symbol)
        us_equity_efinance_store_csv(symbol, "finance_factor", df)
      except:
        print("no finance data skip stock", symbol)
        continue


  def finance_factors_all_stock(self):

    finance_factors = pd.DataFrame()
    for symbol in self.symbols:
      # print("calc symbol ", symbol)
      try:
        pd_data = self.finance_factors_one_stock(symbol)
        [row, col] = pd_data.shape
      except:
        print("no finance data skip stock", symbol)
        continue
      
      try:
        pd_data_stock = pd_data.loc[self.dates, self.factors]
        [row, col] = pd_data_stock.shape
        
        if row == len(self.dates):
          # print("add process symbol ", symbol)
          finance_factors = pd.concat([finance_factors, pd_data.loc[self.dates, self.factors]])
          self.symbol_list_process.append(str(symbol).zfill(6))
      except:
        print('no enough finance data skip stock ', symbol)

    finance_factors.replace([np.inf, -np.inf], 0, inplace=True)

    finance_factors.index = finance_factors.index.map(lambda x: (x[0],'{:0>6s}'.format("'" + str(x[1]))))
    # finance_factors = finance_factors[~finance_factors.isin([np.nan, np.inf, -np.inf]).any(axis=1)]

    return finance_factors

  def finance_factors_rank(self, finance_factors):

    if (finance_factors.empty):
      return pd.DataFrame()

    scaler = MinMaxScaler()

    factor_scaled_sum = pd.DataFrame(0, columns=['scale_sum'], index=self.symbol_list_process)
    factor_scaled_sum.index = factor_scaled_sum.index.map(lambda x: "'" + x)
    for date in self.dates:
      factor = finance_factors.loc[(date, slice(None)), :]
      # factor_scaled = pd.DataFrame(scaler.fit_transform(factor), columns=self.factors, index=self.symbol_list_process)
      # factor_scaled_sum = factor_scaled_sum + pd.DataFrame(factor_scaled.loc[:,self.factors].sum(axis=1), index=self.symbol_list_process, columns=['scale_sum'])
      factor_scaled = factor.rank(pct = True).droplevel(0)
      factor_scaled_sum = factor_scaled_sum + factor_scaled.loc[:,self.factors].sum(axis=1).to_frame('scale_sum')

    factor_scaled_rank = factor_scaled_sum.sort_values(by = ['scale_sum'], ascending=False)
    # factor_scaled_rank.index = factor_scaled_rank.index.map(lambda x: '{:0>6s}'.format("'" + str(x)))
    # factor_scaled_rank.index = [f'{idx:06d}' for idx in factor_scaled_rank.index]
    # factor_scaled_rank.index = [f'{idx:06d}' for idx in factor_scaled_rank.index]
    return factor_scaled_rank


