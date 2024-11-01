
import os
import numpy as np
import pandas as pd

from sklearn import linear_model
import scipy.stats as st

from scipy.stats import rankdata

import numpy.linalg as la
import pandas as pd
from datetime import datetime
import scipy.stats as stats
import copy

from abc import ABC, abstractmethod
from typing import List
import inspect

import threading
import multitasking
from tqdm.auto import tqdm

from quant_free.dataset.us_equity_load import *
from quant_free.utils.us_equity_utils import *

class FactorBase(ABC):

  def __init__(self, start_date, end_date, dir = 'fh'):
    self.start_date = start_date
    self.end_date = end_date
    self.dir = dir
    sector_file = 'us_equity_sector.csv'
    self.sectors = list(us_dir1_load_csv(dir0 = 'symbol', dir1 = dir, filename = sector_file)['name'].values)

  @abstractmethod
  def preprocess(self, data):
    pass

  def excute_for_multidates(self, data, func, level=0, **pramas):
    return data.groupby(level=level, group_keys=False).apply(func,**pramas)
                                                             
  ########### indicator #################
  def get_forward_return(self, stocks_df,column):
      '''计算(未来)下一个回报率
      :param stocks_df: {pd.DataFrame 或 stock_struct}
      :param column: {string}
      :return: {pd.Series}
      '''
      ret = stocks_df[column].groupby(level=0, group_keys=False).apply(lambda x:(x/x.shift(1)-1).shift(-1))
      # ret = stocks_df[column].pct_change(1).shift(-1)
      ret.name = 'ret_forward'
      return ret

  def get_current_return(self, stocks_df,column,stride=1):
      '''计算当期的回报率
      :param stocks_df: {pd.DataFrame 或 stock_struct}
      :param column: {str} --用于计算收益的列名
      :param stride: {int} --计算收益的跨度
      注意：当期回报有可能也包含未来信息。
      '''
      ret = stocks_df[column].groupby(level=0, group_keys=False).apply(lambda x:(x/x.shift(stride)-1))
      # ret = stocks_df[column].pct_change(stride)
      ret.name = 'ret'
      return ret


  def neutralize(self, factor:pd.Series, data, categorical:list=None, logarithmetics:list=None):
      '''中性化：
          :param categorical：{list} --指明需要被dummy的列
          :param logarithmetics：{list}  --指明要对对数化的列
          注：被categorical的column的value必须是字符串。
          注：一般来说，顺序是 去极值->中性化->标准化
          注：单截面操作
      '''
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
          X[logarithmetics] = X[logarithmetics].agg('log')
      # 哑变量
      if not categorical is None:
          X = pd.get_dummies(X,columns=categorical)
          
  #     print(X)
          
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
      # vwap_ = vwap.fillna(value=0)
      # for i in range(len(vwap_)):
      #     vwap_.iloc[i] = neutral(vwap_.iloc[i], ind)
      # return vwap_
      return vwap

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

  def calc_1_symbol(self, symbol, sector_price_ratio):

    dict_data = us_equity_data_load_within_range(symbols = [symbol], start_date = self.start_date,
                                      end_date = self.end_date, column_option = "all", 
                                      dir_option = "xq")
    
    if(len(dict_data) == 1):
      df = pd.concat([dict_data[symbol], sector_price_ratio], axis=1)

      # Insert symbol as the first level of a MultiIndex
      df.index = pd.MultiIndex.from_product([[symbol], df.index])

      print(f'preprocessing {symbol}')

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
      
      subclass_name = self.__class__.__name__
      us_dir1_store_csv(dir0 = 'equity', dir1 = symbol, filename = subclass_name + '.csv', data = df_stored)

  def parallel_calc(self, symbols, sector_price_ratio):

      @multitasking.task
      def start(symbol: str):
          s = self.calc_1_symbol(symbol, sector_price_ratio)
          series.append(s)
          pbar.update()
          pbar.set_description(f'Processing => {symbol}')

      series: List[pd.Series] = []
      pbar = tqdm(total=len(symbols))
      for symbol in symbols:
          start(symbol)
      multitasking.wait_for_tasks()

  def calc(self):
    for sector in self.sectors:

      # sector = '互联网与直销零售'

      print(f"processing {sector} ...")

      sector_price_ratio = us_dir1_load_csv(dir0 = 'symbol', dir1 = self.dir, filename= "index_price_ratio.csv")

      if (sector in sector_price_ratio.columns):
        
        sector_price_ratio = sector_price_ratio.loc[:, sector]

        # sector_price_ratio.rename(columns = {sector:"sector_price_ratio"}, inplace=True)
        # sector_price_ratio.rename(columns={sector:"sector_price_ratio"}, inplace=True)
        sector_price_ratio.name = "sector_price_ratio"

        data_symbols = us_dir1_load_csv(dir0 = 'symbol', dir1 = self.dir, filename= sector +'.csv')
        if (data_symbols.empty == False):
          symbols = data_symbols['symbol'].values
          # symbols = ['OIS', 'FET', 'WTTR']
          self.parallel_calc(symbols, sector_price_ratio)