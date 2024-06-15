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
    
    return pd.to_datetime(date)

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
        4: "FY"
    }

    dates = []

    # Loop through each year
    for year in years:

        # Loop through each quarter (reverse order to start with Q1)
        for quarter in range(1, 5, 1):
            # Append the quarter in the format "YYYY/QN" to the dates list
            dates.append(f"{year}/{quarter_map[quarter]}")

    print(dates)
    dates_between_quarters = [date for date in dates if start_quarter <= date <= end_quarter]

    return dates_between_quarters



class us_equity_finance:

  def __init__(self, symbols, factors = ['ROE'], start_quarter = '2001/Q2', end_quarter = '2023/Q1') -> None:

    self.symbols = symbols

    self.symbol_list_process = []

    self.factors = factors

    self.dates = generate_Q_dates(start_quarter, end_quarter)

    self.date_times = self.dates.apply(convert_to_financial_datetime)

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
    data = a.pct_change(-3)
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


  def finance_stock_financial_data(self, symbol = 'AAPL'):
    try:
      df_finance = us_equity_efinance_finance_data_load(symbol)
      # df_finance = df_finance.sort_index(ascending=False)
      return df_finance
    except:
      print(self.fetch_one_data.__name__, "input wrong!")

  # def nearest_date(self, date_list, input_date):
  #   return min(date_list, key=lambda x: abs(x - input_date))
  
  def nearest_date(self, date_list, input_date):
    input_datetime = input_date
    # Sort the date list in ascending order
    date_list_sorted = sorted(date_list)
    # Do a linear search to find the nearest date
    nearest_date = None
    for date in date_list_sorted:
        dt = date
        if dt > input_datetime:
            if (dt - input_datetime) < (input_datetime - date_list_sorted[date_list_sorted.index(date) - 1]):
                nearest_date = date
            else:
                nearest_date = date_list_sorted[date_list_sorted.index(date) - 1]
            break
    if nearest_date is None:
        nearest_date = date_list_sorted[-1]

    return nearest_date

  def get_market_value(self, symbol):
    daily_data = us_equity_daily_data_load(symbols = [symbol], start_date = self.date_times[0], end_date = self.date_times[-1])
    daily_trade_dates = daily_data.index

    dates = []
    for get_date in self.date_times:
      # print(get_date)
      date = self.nearest_date(daily_trade_dates, get_date)
      dates.append(date)

    shares = us_equity_common_shares_load(symbol)

    # trade_data_quarter = pd.concat([trade_data_quarter, res.loc[date]], axis = 1)
    trade_data_quarter = daily_data.loc[dates]

    return trade_data_quarter * shares

  def finance_factors_one_stock(self, symbol):
    df_finance = self.finance_stock_financial_data(symbol)

    equlity = df_finance['totalOwnersEquity']
    asset = df_finance['totalAssets']
    revenue = df_finance['operatingRevenue']
    cash = df_finance['netCashFlowsFromOperatingActivities']
    debt = df_finance['totalLiabilities']

    earning = df_finance['netProfit']
    tax0 = df_finance['incomeTax']
    cost = df_finance['operatingCosts']
    EBIT = df_finance['EBIT']

    equity = df_finance['totalOwnersEquity']
    debt_total = df_finance['totalLiabilitiesAndOwnersEquity']
    intangible_asset = df_finance['intangibleAssets']
    dev_cost = df_finance['developmentExpenditure']
    goodwell = df_finance['goodwill']
    fix_asset = df_finance['fixedAssets']
    noncurrent_asset = intangible_asset + goodwell + fix_asset

    depreciation = df_finance['depreciationForFixedAssets']
    amortize0 = df_finance['amortizationOfIntangibleAssets']
    amortize1 = df_finance['amortizationOfLong-termDeferredExpenses']
    excess_cash = df_finance['cashEndingBal']
    cash_quivalent = df_finance['cashBeginingBal']

    divedends = df_finance['cashPaymentsForDistrbutionOfDividendsOrProfits']
    EBITDA = EBIT + depreciation +amortize0 + amortize1
    market_cap = self.get_market_value(symbol,)
    EV = market_cap + debt - excess_cash

    ##### earning capacity
    #roe
    roe = pd.DataFrame(df_finance['ROE'])
    roe = roe.rename(columns={'ROE': 'roe'})
    # roe = roe.rename(columns = ['roe'])
    pd_data = roe
    #roa
    roa = self.ab_ratio_calc(earning,asset,'roa')
    pd_data = pd.concat([pd_data, roa], axis=1)
    #roi
    roi = self.ab_ratio_calc(EBIT-tax0,equity+debt_total-cash_quivalent,'roi')
    pd_data = pd.concat([pd_data, roi], axis=1)
    #profit_revenue
    profit_revenue = self.ab_ratio_calc(earning,revenue,'profit_revenue')
    pd_data = pd.concat([pd_data, profit_revenue], axis=1)
    #profit_cost
    profit_cost = self.ab_ratio_calc(earning,cost,'profit_cost')
    pd_data = pd.concat([pd_data, profit_cost], axis=1)
    #stackholder equity increase
    equlity_incr_rate = self.qoq_rate_calc(equity,'equlity_incr_rate')
    pd_data = pd.concat([pd_data, equlity_incr_rate], axis=1)

    ###grow capacity
    #revenue
    revenue_incr_rate = self.qoq_rate_calc(revenue,'revenue_incr_rate')
    pd_data = pd.concat([pd_data, revenue_incr_rate], axis=1)
    #profit
    profit_incr_rate = self.qoq_rate_calc(earning,'profit_incr_rate')
    pd_data = pd.concat([pd_data, profit_incr_rate], axis=1)
    #cash
    cash_incr_rate = self.qoq_rate_calc(cash,'cash_incr_rate')
    pd_data = pd.concat([pd_data, cash_incr_rate], axis=1)
    #asset
    asset_incr_rate = self.qoq_rate_calc(asset,'asset_incr_rate')
    pd_data = pd.concat([pd_data, asset_incr_rate], axis=1)
    #debt
    debt_incr_rate = self.qoq_rate_calc(debt,'debt_incr_rate')
    pd_data = pd.concat([pd_data, debt_incr_rate], axis=1)

    ###asset struct
    #debt_asset_ratio
    debt_asset_ratio = self.ab_ratio_calc(debt,asset,'debt_asset_ratio')
    debt_asset_ratio = debt_asset_ratio/100
    pd_data = pd.concat([pd_data, debt_asset_ratio], axis=1)
    #debt_equity_ratio
    debt_equity_ratio = self.ab_ratio_calc(debt,equity,'debt_equity_ratio')
    debt_equity_ratio = debt_equity_ratio/100
    pd_data = pd.concat([pd_data, debt_equity_ratio], axis=1)
    #debt_net_asset_ratio
    debt_net_asset_ratio = self.abc_ratio_calc(debt,equity,intangible_asset,'debt_net_asset_ratio','sub')
    pd_data = pd.concat([pd_data, debt_net_asset_ratio], axis=1)
    #revenue_asset_ratio
    revenue_asset_ratio = self.ab_ratio_calc(revenue,asset,'revenue_asset_ratio')
    revenue_asset_ratio = revenue_asset_ratio
    pd_data = pd.concat([pd_data, revenue_asset_ratio], axis=1)
    #goodwell_equity_ratio
    goodwell_equity_ratio = self.ab_ratio_calc(goodwell,equity,'goodwell_equity_ratio')
    pd_data = pd.concat([pd_data, goodwell_equity_ratio], axis=1)

    ###CFO2EV
    CFO_EV_ratio = self.ab_ratio_calc(cash,EV,'CFO2EV')
    pd_data = pd.concat([pd_data, CFO_EV_ratio], axis=1)
    ####EBITDA2ev
    EBITDA_EV_ratio = self.ab_ratio_calc(EBITDA,EV,'EDITDA2EV')
    pd_data = pd.concat([pd_data, EBITDA_EV_ratio], axis=1)
    ###BB2P
    divedends_market_cap_ratio = self.ab_ratio_calc(divedends,market_cap,'BB2P')
    pd_data = pd.concat([pd_data, divedends_market_cap_ratio], axis=1)
    ###BB2EV
    divedends_EV_ratio = self.ab_ratio_calc(divedends,EV,'BB2EV')
    pd_data = pd.concat([pd_data, divedends_EV_ratio], axis=1)
    #B2P
    B2P_ratio = self.ab_ratio_calc(equlity,market_cap,'B2P')/100
    pd_data = pd.concat([pd_data, B2P_ratio], axis=1)
    #S2EV
    S2EV_ratio = self.ab_ratio_calc(revenue,EV,'S2EV')
    pd_data = pd.concat([pd_data, S2EV_ratio], axis=1)
    #equity_asset_ratio
    OL = self.ab_ratio_calc(equity,asset,'OL')
    pd_data = pd.concat([pd_data, OL], axis=1)
    #NCO2A
    NCO2A = self.ab_ratio_calc(noncurrent_asset,asset,'NCO2A')
    pd_data = pd.concat([pd_data, NCO2A], axis=1)
    #E2EV
    E2EV = self.ab_ratio_calc(earning,EV,'E2EV')
    pd_data = pd.concat([pd_data, E2EV], axis=1)
    
    return pd_data

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


