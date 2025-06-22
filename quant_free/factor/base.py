
import os
import numpy as np
import pandas as pd

from sklearn import linear_model

from scipy.stats import rankdata

import numpy.linalg as la
import pandas as pd
import copy

from abc import ABC, abstractmethod
from typing import List
import inspect

import multitasking
from tqdm.auto import tqdm

from factorlab.feature_engineering.transformations import Transform as TransformLib

from quant_free.common.us_equity_common import *
from quant_free.dataset.xq_trade_data import *
from quant_free.dataset.xq_symbol import *
from quant_free.dataset.xq_sector_data import *
from quant_free.dataset.us_equity_store import *
from quant_free.utils.us_equity_utils import *

class FactorBase(ABC):

  def __init__(self, start_date, end_date, dir = 'fh', market = 'us'):
    self.start_date = start_date
    self.end_date = end_date
    self.dir = dir
    self.market = market

  @abstractmethod
  def preprocess(self, data):
    pass

  @abstractmethod
  def calc_sectors(self, sectors):
    pass

  def excute_for_multidates(self, data, func, level=0, **pramas):
    if isinstance(data.index, pd.MultiIndex):
      return data.groupby(level=level, group_keys=False).apply(func,**pramas)
    else:
      return data.apply(func,**pramas)
                                                             
  ########### indicator #################
  def get_forward_return(self, stocks_df, lags = 1, column = 'close'):
      '''计算(未来)下一个回报率
      :param stocks_df: {pd.DataFrame 或 stock_struct}
      :param column: {string}
      :return: {pd.Series}
      '''

      # ret = stocks_df[column].groupby(level=0, group_keys=False).apply(lambda x:(x/x.shift(1)-1).shift(-1))

      ret = TransformLib(stocks_df[column]).returns(lags, forward = True)

      ret.columns = ['ret_forward_' + str(lags)]
      return ret

  def get_backward_return(self, stocks_df, lags = 1, column = 'close'):
      '''计算(未来)下一个回报率
      :param stocks_df: {pd.DataFrame 或 stock_struct}
      :param column: {string}
      :return: {pd.Series}
      '''

      # ret = stocks_df[column].groupby(level=0, group_keys=False).apply(lambda x:(x/x.shift(1)-1).shift(-1))

      ret = TransformLib(stocks_df[column]).returns(lags, forward = False)

      ret.columns = ['ret_backward_' + str(lags)]
      return ret

  def get_current_return(self, stocks_df, lags = 1, column = 'close'):
      '''计算当期的回报率
      :param stocks_df: {pd.DataFrame 或 stock_struct}
      :param column: {str} --用于计算收益的列名
      :param stride: {int} --计算收益的跨度
      注意：当期回报有可能也包含未来信息。
      '''
      ret = TransformLib(stocks_df[column]).returns(lags)

      # ret.name = 'ret_' + str(lags)
      ret.columns = ['ret_' + str(lags)]
      return ret


  def neutralize(self, factor:pd.Series, data, categorical:list=None, logarithmetics:list=None):
      '''中性化：
          :param categorical：{list} --指明需要被dummy的列
          :param logarithmetics：{list}  --指明要对对数化的列
          注：被categorical的column的value必须是字符串。
          注：一般来说，顺序是 去极值->中性化->标准化
          注：单截面操作
      '''
      data.replace([np.inf, -np.inf], np.nan, inplace=True)
      factor.replace([np.inf, -np.inf], np.nan, inplace=True)

      data.ffill(inplace=True)
      data.bfill(inplace=True)
      factor.ffill(inplace=True)
      factor.bfill(inplace=True)

      if factor.index.is_monotonic_increasing == False or data.index.is_monotonic_increasing == False:
          import warnings
          warnings.warn('factor or data should be sorted, 否则有可能会造成会自变量和因变量匹配错误',UserWarning)
          
      X = data.copy()
      # 对数化
      if not logarithmetics is None:
          # X[logarithmetics] = X[logarithmetics].agg('log')
          X = X.agg('log')
      # 哑变量
      if not categorical is None:
          X = pd.get_dummies(X,columns=categorical)
          
  #     print(X)
      X = X.values.reshape(-1, 1)
      model = linear_model.LinearRegression(fit_intercept=False).fit(X, factor)
      neutralize_factor = factor - model.predict(X)

      return neutralize_factor

      

  def winsorize_by_quantile(self, obj, floor=0.025, upper=0.975, column=None, drop=True):
      """
        根据分位上下限选取数据
        :param obj:{pd.DataFrame | pd.Series} 
        :param column:{str} --当obj为DataFrame时，用来指明处理的列。
        :param drop:{bool} --分位外的数据处理方式，
                              True：删除整（行）条数据；
                              False：用临界值替换范围外的值
      """
      if isinstance(obj, pd.Series):
          qt = obj.quantile([floor,upper])
          if drop:
              return obj[(obj>=qt[floor]) & (obj<=qt[upper])]
          else:
              obj[obj < qt[floor]] = qt[floor]
              obj[obj > qt[upper]] = qt[upper]
              return obj
      
      if isinstance(obj, pd.DataFrame):
          assert column, 'COLUMN CANT be NONE when obj is dataframe'
          qt = obj[column].quantile([floor,upper])
          if drop:
              return obj[(obj[column]>=qt[floor]) & (obj[column]<=qt[upper])]
          else:
              obj.loc[obj[column] < qt[floor], column] = qt[floor]
              obj.loc[obj[column] > qt[upper], column] = qt[upper]
              return obj
      raise TypeError('obj must be series or dataframe')
      
  def winsorize_by_mad(self, obj, n=3, column=None, drop=True):
      """
        根据中位数偏离倍数选取数据
        :param obj:{pd.DataFrame | pd.Series} 
        :param n:{pd.DataFrame | pd.Series} --偏离倍数
        :param column:{str} --当obj为DataFrame时，用来指明处理的列。
        :param drop:{bool} --分位外的数据处理方式，
                              True：删除整（行）条数据；
                              False：用临界值替换范围外的值
      """
      
      if isinstance(obj, pd.Series):
          median = np.median(obj.dropna())
          mad = np.median((obj.dropna() - median).abs())
          #样本标准差的估计量(σ≈1.483)
          mad_e = 1.483*mad
          upper = median + n*mad_e
          floor = median - n*mad_e
          if drop:
              return obj[(obj>=floor) & (obj<=upper) | obj.isna()]
          else:
              obj[obj < floor] = floor
              obj[obj > upper] = upper
              return obj
      
      if isinstance(obj, pd.DataFrame):
          assert column, 'COLUMN CANT be NONE when obj is dataframe'
          median = np.median(obj[column].dropna())
          mad = np.median((obj.dropna() - median).abs())
          mad_e = 1.483*mad
          upper = median + n*mad_e
          floor = median - n*mad_e
          if drop:
              return obj[(obj[column]>=floor) & (obj[column]<=upper) | obj[column].isna()]
          else:
              obj.loc[obj[column] < floor, column] = floor
              obj.loc[obj[column] > upper, column] = upper
              return obj
      
      raise TypeError('obj must be series or dataframe')


  # 标准化
  def standardize(self, data, multi_code=False):
      if multi_code:
          return data.groupby(level=1, group_keys=False).apply(lambda x: standardize(x,multi_code=False))
      else:
          return (data - data.mean())/data.std()

  def binning(self, df, deal_column:str,box_count:int, labels=None, inplace=True):
      """
        分箱，为df增加名为"group_label"的列作为分组标签。
        :param df:{pd.DataFrame} 
        :param deal_column:{str} --要处理的列名,
        :param box_count:{int} --分几组,
        :param labels:{list} --分组的标签名，默认是分组序号（default:None）
                                默认情况下，生成的标签是反序的，既最小的值在最后的组
        :param inplace:{bool} --是否在原对象上修改,建议用true，效率高（default:True）
        :return: {pd.DataFame}
      """
      assert isinstance(df, pd.DataFrame), 'df必须为dataframe'
      if not labels is None:
          assert len(labels)==box_count, 'labels的数量必须与分箱数相等'
          labels_= labels
      else:
          labels_= np.array(range(box_count))+1
          labels_ = labels_[::-1]
      
      vals = df[deal_column]
      val_set = vals.unique()
      reality_count = len(val_set)
      
      if inplace:
          if box_count > reality_count:
              # 可能由于大量0或者nan，导致分类的数量少于分箱数量。 直接当任务失败，返回空值
              df['group_label'] = None
              return df
          else:
              vals = df[deal_column]
              val_set = vals.unique()
              bins = pd.qcut(val_set, box_count, labels=labels_, retbins=False,)
              val_bin_dic = {key:bin_val for key,bin_val in zip(val_set,bins)}
              res = list(map(lambda x: val_bin_dic[x], vals))
              
              df['group_label'] = res
              return df
      else:
          if box_count > reality_count:
              # 可能由于大量0或者nan，导致分类的数量少于分箱数量。 直接当任务失败，返回空值
              return df.assign(group_label=None)
          else:
              bins = pd.qcut(val_set, box_count, labels=labels_, retbins=False,)
              val_bin_dic = {key:bin_val for key,bin_val in zip(val_set,bins)}
              res = list(map(lambda x: val_bin_dic[x], vals))
              return df.assign(group_label=res)



  # 中性(行业中性)
  def neu_industry(self, stock_data,sector_series):
      """
      a_transform = a / sector_average(a)
      """
      #align sector_series to stock_data in case of stock_data's sparse stocks
      inner = sector_series[stock_data.columns]
      stock_data = stock_data.replace([-np.inf,np.inf],0).fillna(value=0)
      trans_matrix = pd.DataFrame()
      for stock in stock_data.columns:
          df = inner==inner[stock]
          df /= df.sum()
          trans_matrix = pd.concat([trans_matrix,df],axis=1,sort=True)
      
      result = pd.DataFrame(np.dot(stock_data.fillna(value=0),trans_matrix.fillna(value=0)),
                            index=stock_data.index,columns=stock_data.columns)    
      return (stock_data / result)

  def neutral(self, data, ind):
      # stocks = list(data.index)
      X = np.array(pd.get_dummies(ind))
      # y = data.values
      y = data
      beta_ols = la.inv(X.T.dot(X)).dot(X.T).dot(y)
      residual = y - X.dot(beta_ols)
      return residual


  def IndNeutralize(self, vwap, ind):
      vwap_ = vwap.fillna(value=0)
      for i in range(len(vwap_)):
          vwap_.iloc[i] = self.neutral(vwap_.iloc[i], ind)
      return vwap_
      # return vwap

  # 移动求和
  def ts_sum(self, df, window):
      return df.rolling(window).sum()

  # 移动平均
  def sma(self, df, window):
      return df.rolling(window).mean()

  # 移动标准差
  def stddev(self, df, window):
      return df.rolling(window).std()

  # 移动相关系数
  def correlation(self, x, y, window):
      return x.rolling(window).corr(y)

  # 移动协方差
  def covariance(self, x, y, window):
      return x.rolling(window).cov(y)

  # 在过去d天的时序排名
  def rolling_rank(self, na):
      return rankdata(na)[-1]


  def ts_rank(self, df, window):
      return df.rolling(window).apply(self.rolling_rank)

  # 过去d天的时序乘积
  def rolling_prod(self, na):
      return np.prod(na)


  def product(self, df, window):
      return df.rolling(window).apply(self.rolling_prod)

  # 过去d天最小值
  def ts_min(self, df, window):
      return df.rolling(window).min()
  # 过去d天最大值


  def ts_max(self, df, window):
      return df.rolling(window).max()

  # 当天取值减去d天前的值
  def delta(self, df, period):
      return df.diff(period)

  # d天前的值，滞后值
  def delay(self, df, period):
      return df.shift(period)

  # 截面数据排序，输出boolean值
  def rank(self, df):
      return df.rank(pct=True)
      # return df.rank(pct=True, axis=1)
  # 缩放时间序列，使其和为1
  def scale(self, df, k=1):
      return df.mul(k).div(np.abs(df).sum())

  # 过去d天最大值的位置
  def ts_argmax(self, df, window):
      return df.rolling(window).apply(np.argmax) + 1

  # 过去d天最小值的位置
  def ts_argmin(self, df, window):
      return df.rolling(window).apply(np.argmin) + 1

  # 线性衰减的移动平均加权
  def decay_linear(self, df, period):
      if df.isnull().values.any():
          # df.fillna(method='ffill', inplace=True)
          # df.fillna(method='bfill', inplace=True)
          df.ffill(inplace=True)
          df.bfill(inplace=True)
          df.fillna(value=0, inplace=True)
      na_lwma = np.zeros_like(df)  # 生成与df大小相同的零数组
  
      # na_lwma[:period, :] = df.iloc[:period, :]  # 赋前period项的值
      na_lwma[:period, ] = df.iloc[:period, ]  # 赋前period项的值
      # na_series = df.as_matrix()
      na_series = df.values
      
      # 计算加权系数
      divisor = period * (period + 1) / 2
      y = (np.arange(period) + 1) * 1.0 / divisor
      # 从第period项开始计算数值
      for row in range(period - 1, df.shape[0]):
          # x = na_series[row - period + 1: row + 1, :]
          x = na_series[row - period + 1: row + 1, ]
          # na_lwma[row, :] = (np.dot(x.T, y))
          na_lwma[row, ] = (np.dot(x.T, y))
      # return pd.DataFrame(na_lwma, index=df.index, columns=df.columns)
      return pd.Series(na_lwma, index=df.index, name=df.name)
    
  def add_forward_returns(self, df, df_stored, periods = [1, 5, 10, 15, 20]):
      """
      Adds forward returns for specified periods to the stored DataFrame.

      Parameters:
      - df: pd.DataFrame, the input DataFrame
      - periods: list of int, the forward periods to calculate
      - df_stored: pd.DataFrame, the DataFrame to append results to

      Returns:
      - pd.DataFrame: The updated DataFrame with forward returns
      """
      for period in periods:
          forward_return = self.get_forward_return(df, period)
          df_stored = pd.concat([df_stored, forward_return], axis=1)
      return df_stored
  
  def add_backward_returns(self, df, df_stored, periods = [1, 5, 10, 15, 20]):
      """
      Adds forward returns for specified periods to the stored DataFrame.

      Parameters:
      - df: pd.DataFrame, the input DataFrame
      - periods: list of int, the forward periods to calculate
      - df_stored: pd.DataFrame, the DataFrame to append results to

      Returns:
      - pd.DataFrame: The updated DataFrame with forward returns
      """
      for period in periods:
          backward_return = self.get_backward_return(df, period)
          df_stored = pd.concat([df_stored, backward_return], axis=1)
      return df_stored


  def calc_1_sector(self, sector):

    subclass_name = self.__class__.__name__
    
    print(f'preprocessing {sector}')

    df = copy.deepcopy(
        multi_sym_daily_load_multi_index(
        self.market,
        sector_name = sector,
        start_date = self.start_date,
        end_date = self.end_date,
        dir_option = self.dir)
        )
    
    if df is None:
      print(f"Skip this sector {sector}, no equity in it!")
    else:
      # Insert symbol as the first level of a MultiIndex
      if 'Trend' == subclass_name:
        # df.index = pd.MultiIndex.from_product([df.index, {'ticker': [symbol]}])
        window_size = [5, 10, 15, 20, 25, 30]
       
        df_stored = pd.DataFrame() 
        for window in window_size:
          df_preprocess = self.preprocess(df, window)
          
          for method_name in dir(self):
              if method_name.startswith("trend"):
                # Get the method by its name
                method = getattr(self, method_name)

                # Ensure that the attribute is a callable (method)
                if callable(method):
                    # Call the method
                    print(f"Calling {method_name}...")
                    result = method(df_preprocess)

                    result.columns = ['trend_' + col for col in result.columns]

                    # result.name = method_name
                    df_stored = pd.concat([df_stored, result], axis = 1)

      else:
        
        # df.index = pd.MultiIndex.from_product([[symbol], df.index])
        df_preprocess = self.preprocess(df)
        
        df_preprocess.ffill(inplace=True)
        df_preprocess.bfill(inplace=True)

        df_stored =  copy.deepcopy(df_preprocess)
        # Iterate over all attributes of the instance (methods and variables)
        for method_name in dir(self):
            # Check if the method name starts with 'alpha'
            if method_name.startswith("alpha"):
                # Get the method by its name
                method = getattr(self, method_name)

                # Ensure that the attribute is a callable (method)
                if callable(method):
                    # Call the method
                    print(f"Calling {method_name}...")
                    params = inspect.signature(getattr(self, method_name)).parameters.keys()
                    input_args = [df_preprocess[param].copy() for param in params]
                    result = method(*input_args)
                    
                    result.name = method_name
                    df_stored = pd.concat([df_stored, result], axis = 1)

      df_stored = self.add_backward_returns(df, df_stored)
      df_stored = self.add_forward_returns(df, df_stored)
      us_equity_filter_and_store_by_symbol(self.market, df_stored, subclass_name)
      return df_stored

  def parallel_calc_sectors(self, sectors):

      @multitasking.task
      def start(sector: str):
          s = self.calc_1_sector(sector)
          series.append(s)
          pbar.update()
          pbar.set_description(f'Processing => {sector}')

      series: List[pd.Series] = []
      pbar = tqdm(total=len(sectors))
      for sector in sectors:
          start(sector)
      multitasking.wait_for_tasks()

  def calc_1_symbol(self, symbol, sector_price = None):

    subclass_name = self.__class__.__name__
    
    if sector_price is None and "Trend" != subclass_name:
      sector = get_equity_sector_name(symbol, self.dir)
      sector_price = us_dir1_load_csv(self.market, dir0 = 'symbol', dir1 = self.dir, filename= "index_price.csv")
      sector_price = sector_price.loc[:, sector]
      sector_price.name = "sector_price"

    dict_data = multi_sym_daily_load(
                                      self.market,
                                      symbols = [symbol], start_date = self.start_date,
                                      end_date = self.end_date, column_option = "all", 
                                      dir_option = "xq")
    
    if(len(dict_data) == 1):
      df = pd.concat([dict_data[symbol], sector_price], axis=1)

      # Insert symbol as the first level of a MultiIndex

      print(f'preprocessing {symbol}')

      if 'Trend' == subclass_name:
        # df.index = pd.MultiIndex.from_product([df.index, {'ticker': [symbol]}])
        df_preprocess = self.preprocess(df)
        df_stored = pd.DataFrame() 
        
        for method_name in dir(self):
            if method_name.startswith("trend"):
              # Get the method by its name
              method = getattr(self, method_name)

              # Ensure that the attribute is a callable (method)
              if callable(method):
                  # Call the method
                  print(f"Calling {method_name}...")
                  result = method(df_preprocess)
                  
                  if isinstance(result.index, pd.MultiIndex):
                    result.index = result.index.get_level_values(0)
                  
                  result.columns = ['trend_' + col for col in result.columns]

                  # result.name = method_name
                  df_stored = pd.concat([df_stored, result], axis = 1)

      else:
        
        # df.index = pd.MultiIndex.from_product([[symbol], df.index])
        df_preprocess = self.preprocess(df)
        
        df_preprocess.ffill(inplace=True)
        df_preprocess.bfill(inplace=True)

        df_stored =  copy.deepcopy(df_preprocess)
        # Iterate over all attributes of the instance (methods and variables)
        for method_name in dir(self):
            # Check if the method name starts with 'alpha'
            if method_name.startswith("alpha"):
                # Get the method by its name
                method = getattr(self, method_name)

                # Ensure that the attribute is a callable (method)
                if callable(method):
                    # Call the method
                    print(f"Calling {method_name}...")
                    params = inspect.signature(getattr(self, method_name)).parameters.keys()
                    input_args = [df_preprocess[param].copy() for param in params]
                    result = method(*input_args)
                    
                    result.name = method_name
                    df_stored = pd.concat([df_stored, result], axis = 1)

      df_stored = self.add_backward_returns(df, df_stored)
      df_stored = self.add_forward_returns(df, df_stored)

      df_stored.index.name = "date"
      us_dir1_store_csv(self.market, 
                        dir0 = 'equity', 
                        dir1 = symbol, 
                        filename = subclass_name + '.csv', 
                        data = df_stored)

      return df_stored

  def parallel_calc(self, symbols, sector_price):

      @multitasking.task
      def start(symbol: str):
          s = self.calc_1_symbol(symbol, sector_price)
          series.append(s)
          pbar.update()
          pbar.set_description(f'Processing => {symbol}')

      series: List[pd.Series] = []
      pbar = tqdm(total=len(symbols))
      for symbol in symbols:
          start(symbol)
      multitasking.wait_for_tasks()

  def parallel_calc_debug(self, symbols, sector_price):

      for symbol in symbols:
          self.calc_1_symbol(symbol, sector_price)