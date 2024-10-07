#!/usr/bin/env python
# coding: utf-8


from scipy.stats import rankdata
from dateutil import parser
import numpy as np
import numpy.linalg as la
import pandas as pd
from datetime import datetime
import scipy.stats as stats
import matplotlib.pyplot as plt
import copy

from quant_free.factor.base import FactorBase


class Alpha101(FactorBase):

  def __init__(self, start_date, end_date, dir = 'fh'):

    super().__init__(start_date, end_date, dir)

  def preprocess(self, data):
      returns = self.get_current_return(data,'close')
      returns.name = 'returns'
      ret_forward = self.get_forward_return(data,'close')
      ret_forward.name = 'ret_forward'
      # {'Open', 'cap', 'close', 'high', 'ind', 'low', 'returns', 'volume', 'vwap'}
      data = pd.concat([data, returns, ret_forward], axis=1)
      data = data.assign(vwap=data.amount/(data.volume*100))
      data.rename(columns = {"open":"Open",'market_capital':'cap','sector_price_ratio':'ind'}, inplace=True)
      data['cap']=data['cap']/data['close'] # 数据取出来的是市值


      close_ind = self.neutralize(data.close, data['ind'],categorical=['ind'])
      close_ind.name = 'close_ind'
      vwap_ind = self.neutralize(data.vwap, data['ind'],categorical=['ind'])
      vwap_ind.name = 'vwap_ind'
      high_ind = self.neutralize(data.high, data['ind'],categorical=['ind'])
      high_ind.name = 'high_ind'
      low_ind = self.neutralize(data.low, data['ind'],categorical=['ind'])
      low_ind.name = 'low_ind'
      volume_ind = self.neutralize(data.volume, data['ind'],categorical=['ind'])
      volume_ind.name = 'volume_ind'

      adv20 = self.excute_for_multidates(data.volume, lambda x:x.rolling(20).agg('mean'), level=0)
      adv20 = pd.concat([adv20,data['ind']],axis=1).dropna()
      adv20_ind = self.neutralize(adv20.volume, adv20['ind'],categorical=['ind'])
      adv20_ind.name = 'adv20_ind'

      adv40 = self.excute_for_multidates(data.volume, lambda x:x.rolling(40).agg('mean'), level=0)
      adv40 = pd.concat([adv40, data['ind']],axis=1).dropna()
      adv40_ind = self.neutralize(adv40.volume, adv40['ind'],categorical=['ind'])
      adv40_ind.name = 'adv40_ind'

      adv81 = self.excute_for_multidates(data.volume, lambda x:x.rolling(81).agg('mean'), level=0)
      adv81 = pd.concat([adv81, data['ind']],axis=1).dropna()
      adv81_ind = self.neutralize(adv81.volume, adv81['ind'],categorical=['ind'])
      adv81_ind.name = 'adv81_ind'

      co_mixed = ((data.close * 0.60733) + (data.Open * (1 - 0.60733)))
      co_mixed_ind = self.neutralize(co_mixed, data['ind'],categorical=['ind'])
      co_mixed_ind.name = 'co_mixed_ind'

      oh_mixed = ((data.Open * 0.868128) + (data.high * (1 - 0.868128)))
      oh_mixed_ind = self.neutralize(oh_mixed, data['ind'],categorical=['ind'])
      oh_mixed_ind.name = 'oh_mixed_ind'

      lv_mixed = ((data.low * 0.721001) + (data.vwap * (1 - 0.721001)))
      lv_mixed_ind = self.neutralize(lv_mixed, data['ind'],categorical=['ind'])
      lv_mixed_ind.name = 'lv_mixed_ind'

      return pd.concat([data, close_ind, vwap_ind, low_ind, high_ind, volume_ind, adv20_ind, adv40_ind, adv81_ind, co_mixed_ind, oh_mixed_ind, lv_mixed_ind], axis=1)



  # 因子函数
  def alpha1(self, close, returns):
      x =  copy.deepcopy(close)
      x[returns < 0] = self.stddev(returns, 20)
      alpha = self.rank(self.ts_argmax(x ** 2, 5))-0.5
      return alpha

  def alpha2(self, Open, close, volume):
      r1 = self.rank(self.delta(np.log(volume), 2))
      r2 = self.rank((close - Open) / Open)
      alpha = -1 * self.correlation(r1, r2, 6)
      return alpha


  def alpha3(self, Open, volume):
      r1 = self.rank(Open)
      r2 = self.rank(volume)
      alpha = -1 * self.correlation(r1, r2, 10)
      return alpha.replace([-np.inf, np.inf], 0)


  def alpha4(self, low):
      r = self.rank(low)
      alpha = -1 * self.ts_rank(r, 9)
      return alpha


  def alpha5(self, Open, vwap, close):
      alpha = (self.rank((Open - (self.ts_sum(vwap, 10) / 10)))
              * (-1 * abs(self.rank((close - vwap)))))
      return alpha


  def alpha6(self, Open, volume):
      alpha = -1 * self.correlation(Open, volume, 10)
      return alpha.replace([-np.inf, np.inf], 0)


  def alpha7(self, volume, close):
      adv20 = self.sma(volume, 20)
      alpha = -1 * self.ts_rank(abs(self.delta(close, 7)), 60) * np.sign(self.delta(close, 7))
      alpha[adv20 >= volume] = -1
      return alpha


  def alpha8(self, Open, returns):
      x1 = (self.ts_sum(Open, 5) * self.ts_sum(returns, 5))
      x2 = self.delay((self.ts_sum(Open, 5) * self.ts_sum(returns, 5)), 10)
      alpha = -1 * self.rank(x1-x2)
      return alpha


  def alpha9(self, close):
      self.delta_close = self.delta(close, 1)
      x1 = self.ts_min(self.delta_close, 5) > 0
      x2 = self.ts_max(self.delta_close, 5) < 0
      alpha = -1 * self.delta_close
      alpha[x1 | x2] = self.delta_close
      return alpha


  def alpha10(self, close):
      self.delta_close = self.delta(close, 1)
      x1 = self.ts_min(self.delta_close, 4) > 0
      x2 = self.ts_max(self.delta_close, 4) < 0
      x = -1 * self.delta_close
      # x[x1 | x2] = self.delta_close
      x.loc[x1 | x2] = self.delta_close
      alpha = self.rank(x)
      return alpha


  def alpha11(self, vwap, close, volume):
      x1 = self.rank(self.ts_max((vwap - close), 3))
      x2 = self.rank(self.ts_min((vwap - close), 3))
      x3 = self.rank(self.delta(volume, 3))
      alpha = (x1 + x2) * x3
      return alpha


  def alpha12(self, volume, close):
      alpha = np.sign(self.delta(volume, 1)) * (-1 * self.delta(close, 1))
      return alpha


  def alpha13(self, volume, close):
      alpha = -1 * self.rank(self.covariance(self.rank(close), self.rank(volume), 5))
      return alpha


  def alpha14(self, Open, volume, returns):
      x1 = self.correlation(Open, volume, 10).replace(
          [-np.inf, np.inf], 0).fillna(value=0)
      x2 = -1 * self.rank(self.delta(returns, 3))
      alpha = x1 * x2
      return alpha


  def alpha15(self, high, volume):
      x1 = self.correlation(self.rank(high), self.rank(volume), 3).replace(
          [-np.inf, np.inf], 0).fillna(value=0)
      alpha = -1 * self.ts_sum(self.rank(x1), 3)
      return alpha


  def alpha16(self, high, volume):
      alpha = -1 * self.rank(self.covariance(self.rank(high), self.rank(volume), 5))
      return alpha


  def alpha17(self, volume, close):
      adv20 = self.sma(volume, 20)
      x1 = self.rank(self.ts_rank(close, 10))
      x2 = self.rank(self.delta(self.delta(close, 1), 1))
      x3 = self.rank(self.ts_rank((volume / adv20), 5))
      alpha = -1 * (x1 * x2 * x3)
      return alpha


  def alpha18(self, close, Open):
      x = self.correlation(close, Open, 10).replace(
          [-np.inf, np.inf], 0).fillna(value=0)
      alpha = -1 * (self.rank((self.stddev(abs((close - Open)), 5) + (close - Open)) + x))
      return alpha


  def alpha19(self, close, returns):
      x1 = (-1 * np.sign((close - self.delay(close, 7)) + self.delta(close, 7)))
      x2 = (1 + self.rank(1 + self.ts_sum(returns, 250)))
      alpha = x1 * x2
      return alpha


  def alpha20(self, Open, high, close, low):
      alpha = -1 * (self.rank(Open - self.delay(high, 1)) * self.rank(Open -
                                                      self.delay(close, 1)) * self.rank(Open - self.delay(low, 1)))
      return alpha


  def alpha21(self, volume, close):
      x1 = self.sma(close, 8) + self.stddev(close, 8) < self.sma(close, 2)
      x2 = self.sma(close, 8) - self.stddev(close, 8) > self.sma(close, 2)
      x3 = self.sma(volume, 20) / volume < 1
      # alpha = pd.DataFrame(np.ones_like(
      #     close), index=close.index, columns=close.columns)
      alpha = pd.DataFrame(np.ones_like(
          close), index=close.index,columns=['alpha21'])
      alpha[x1 | x3] = -1 * alpha
      return alpha


  def alpha22(self, high, volume, close):
      x = self.correlation(high, volume, 5).replace(
          [-np.inf, np.inf], 0).fillna(value=0)
      alpha = -1 * self.delta(x, 5) * self.rank(self.stddev(close, 20))
      return alpha


  def alpha23(self, high, close):
      x = self.sma(high, 20) < high
      alpha = pd.DataFrame(np.zeros_like(close), index=close.index, columns=['alpha23'])
      a = -1 * self.delta(high, 2).fillna(value=0)
      # alpha['alpha23'][x] = a
      alpha.loc[x, 'alpha23'] = a
      return alpha


  def alpha24(self, close):
      x = self.delta(self.sma(close, 100), 100) / self.delay(close, 100) <= 0.05
      alpha = -1 * self.delta(close, 3)
      alpha[x] = -1 * (close - self.ts_min(close, 100))
      return alpha


  def alpha25(self, volume, returns, vwap, high, close):
      adv20 = self.sma(volume, 20)
      alpha = self.rank((((-1 * returns) * adv20) * vwap) * (high - close))
      return alpha


  def alpha26(self, volume, high):
      x = self.correlation(self.ts_rank(volume, 5), self.ts_rank(high, 5), 5).replace(
          [-np.inf, np.inf], 0).fillna(value=0)
      alpha = -1 * self.ts_max(x, 3)
      return alpha


  def alpha27(self, volume, vwap):
      alpha = self.rank((self.sma(self.correlation(self.rank(volume), self.rank(vwap), 6), 2) / 2.0))
      alpha[alpha > 0.5] = -1
      alpha[alpha <= 0.5] = 1
      return alpha


  def alpha28(self, volume, high, low, close):
      adv20 = self.sma(volume, 20)
      x = self.correlation(adv20, low, 5).replace(
          [-np.inf, np.inf], 0).fillna(value=0)
      alpha = self.scale(((x + ((high + low) / 2)) - close))
      return alpha


  def alpha29(self, close, returns):
      x1 = self.ts_min(self.rank(
          self.rank(self.scale(np.log(self.ts_sum(self.rank(self.rank(-1 * self.rank(self.delta((close - 1), 5)))), 2))))), 5)
      x2 = self.ts_rank(self.delay((-1 * returns), 6), 5)
      alpha = x1 + x2
      return alpha


  def alpha30(self, close, volume):
      self.delta_close = self.delta(close, 1)
      x = np.sign(self.delta_close) + np.sign(self.delay(self.delta_close, 1)) + \
          np.sign(self.delay(self.delta_close, 2))
      alpha = ((1.0 - self.rank(x)) * self.ts_sum(volume, 5)) / self.ts_sum(volume, 20)
      return alpha


  def alpha31(self, close, low, volume):
      adv20 = self.sma(volume, 20)
      x1 = self.rank(self.rank(self.rank(self.decay_linear((-1 * self.rank(self.rank(self.delta(close, 10)))), 10))))
      x2 = self.rank((-1 * self.delta(close, 3)))
      x3 = np.sign(self.scale(self.correlation(adv20, low, 12).replace(
          [-np.inf, np.inf], 0).fillna(value=0)))
      alpha = x1 + x2 + x3
      return alpha


  def alpha32(self, close, vwap):
      x = self.correlation(vwap, self.delay(close, 5), 230).replace(
          [-np.inf, np.inf], 0).fillna(value=0)
      alpha = self.scale(((self.sma(close, 7)) - close)) + 20 * self.scale(x)
      return alpha


  def alpha33(self, Open, close):
      alpha = self.rank(-1 + (Open / close))
      return alpha


  def alpha34(self, close, returns):
      x = (self.stddev(returns, 2) / self.stddev(returns, 5)).fillna(value=0)
      alpha = self.rank(2 - self.rank(x) - self.rank(self.delta(close, 1)))
      return alpha


  def alpha35(self, volume, close, high, low, returns):
      x1 = self.ts_rank(volume, 32)
      x2 = 1 - self.ts_rank(close + high - low, 16)
      x3 = 1 - self.ts_rank(returns, 32)
      alpha = (x1 * x2 * x3).fillna(value=0)
      return alpha


  def alpha36(self, Open, close, volume, returns, vwap):
      adv20 = self.sma(volume, 20)
      x1 = 2.21 * self.rank(self.correlation((close - Open), self.delay(volume, 1), 15))
      x2 = 0.7 * self.rank((Open - close))
      x3 = 0.73 * self.rank(self.ts_rank(self.delay((-1 * returns), 6), 5))
      x4 = self.rank(abs(self.correlation(vwap, adv20, 6)))
      x5 = 0.6 * self.rank((self.sma(close, 200) - Open) * (close - Open))
      alpha = x1 + x2 + x3 + x4 + x5
      return alpha


  def alpha37(self, Open, close):
      alpha = self.rank(self.correlation(self.delay(Open - close, 1),
                              close, 200)) + self.rank(Open - close)
      return alpha


  def alpha38(self, close, Open):
      x = (close / Open).replace([-np.inf, np.inf], 0).fillna(value=0)
      alpha = -1 * self.rank(self.ts_rank(Open, 10)) * self.rank(x)
      return alpha


  def alpha39(self, volume, close, returns):
      adv20 = self.sma(volume, 20)
      x = -1 * self.rank(self.delta(close, 7)) * \
          (1 - self.rank(self.decay_linear((volume / adv20), 9)))
      alpha = x * (1 + self.rank(self.ts_sum(returns, 250)))
      return alpha


  def alpha40(self, high, volume):
      alpha = -1 * self.rank(self.stddev(high, 10)) * self.correlation(high, volume, 10)
      return alpha


  def alpha41(self, high, low, vwap):
      alpha = pow((high * low), 0.5) - vwap
      return alpha


  def alpha42(self, vwap, close):
      alpha = self.rank((vwap - close)) / self.rank((vwap + close))
      return alpha


  def alpha43(self, volume, close):
      adv20 = self.sma(volume, 20)
      alpha = self.ts_rank(volume / adv20, 20) * self.ts_rank((-1 * self.delta(close, 7)), 8)
      return alpha


  def alpha44(self, high, volume):
      alpha = -1 * self.correlation(high, self.rank(volume),
                              5).replace([-np.inf, np.inf], 0)
      return alpha


  def alpha45(self, close, volume):
      x = self.correlation(close, volume, 2).replace(
          [-np.inf, np.inf], 0).fillna(value=0)
      alpha = -1 * (self.rank(self.sma(self.delay(close, 5), 20)) * x *
                    self.rank(self.correlation(self.ts_sum(close, 5), self.ts_sum(close, 20), 2)))
      return alpha


  def alpha46(self, close):
      x = ((self.delay(close, 20) - self.delay(close, 10)) / 10) - \
          ((self.delay(close, 10) - close) / 10)
      alpha = (-1 * (close - self.delay(close, 1)))
      alpha[x < 0] = 1
      alpha[x > 0.25] = -1
      return alpha


  def alpha47(self, volume, close, high, vwap):
      adv20 = self.sma(volume, 20)
      alpha = ((self.rank((1 / close)) * volume) / adv20) * ((high *
                                                        self.rank((high - close))) / self.sma(high, 5)) - self.rank((vwap - self.delay(vwap, 5)))
      return alpha


  def alpha49(self, close):
      x = (((self.delay(close, 20) - self.delay(close, 10)) / 10) -
          ((self.delay(close, 10) - close) / 10))
      alpha = (-1 * self.delta(close, 1))
      alpha[x < -0.1] = 1
      return alpha


  def alpha50(self, volume, vwap):
      alpha = -1 * self.ts_max(self.rank(self.correlation(self.rank(volume), self.rank(vwap), 5)), 5)
      return alpha


  def alpha51(self, close):
      inner = (((self.delay(close, 20) - self.delay(close, 10)) / 10) -
              ((self.delay(close, 10) - close) / 10))
      alpha = (-1 * self.delta(close, 1))
      alpha[inner < -0.05] = 1
      return alpha


  def alpha52(self, returns, volume, low):
      x = self.rank(((self.ts_sum(returns, 240) - self.ts_sum(returns, 20)) / 220))
      alpha = -1 * self.delta(self.ts_min(low, 5), 5) * x * self.ts_rank(volume, 5)
      return alpha


  def alpha53(self, close, high, low):
      alpha = -1 * self.delta((((close - low) - (high - close)) /
                          (close - low).replace(0, 0.0001)), 9)
      return alpha


  def alpha54(self, Open, close, high, low):
      x = (low - high).replace(0, -0.0001)
      alpha = -1 * (low - close) * (Open ** 5) / (x * (close ** 5))
      return alpha


  def alpha55(self, high, low, close, volume):
      x = (close - self.ts_min(low, 12)) / \
          (self.ts_max(high, 12) - self.ts_min(low, 12)).replace(0, 0.0001)
      alpha = -1 * self.correlation(self.rank(x), self.rank(volume),
                              6).replace([-np.inf, np.inf], 0)
      return alpha

  #cap 市值
  def alpha56(self, returns, cap):
      alpha = 0 - \
          (1 * (self.rank((self.sma(returns, 10) / self.sma(self.sma(returns, 2), 3))) * self.rank((returns * cap))))
      return alpha


  def alpha57(self, close, vwap):
      alpha = 0 - 1 * ((close - vwap) /
                      self.decay_linear(self.rank(self.ts_argmax(close, 30)), 2))
      return alpha


  def alpha60(self, close, high, low, volume):
      x = ((close - low) - (high - close)) * \
          volume / (high - low).replace(0, 0.0001)
      alpha = - ((2 * self.scale(self.rank(x))) - self.scale(self.rank(self.ts_argmax(close, 10))))
      return alpha


  def alpha61(self, volume, vwap):
      adv180 = self.sma(volume, 180)
      alpha = self.rank((vwap - self.ts_min(vwap, 16))
                  ) < self.rank(self.correlation(vwap, adv180, 18))
      return alpha


  def alpha62(self, volume, high, low, Open, vwap):
      adv20 = self.sma(volume, 20)
      x1 = self.rank(self.correlation(vwap, self.ts_sum(adv20, 22), 10))
      x2 = self.rank(((self.rank(Open) + self.rank(Open)) <
                (self.rank(((high + low) / 2)) + self.rank(high))))
      alpha = x1 < x2
      return alpha*-1


  def alpha64(self, high, low, Open, volume, vwap):
      adv120 = self.sma(volume, 120)
      x1 = self.rank(self.correlation(self.ts_sum(
          ((Open * 0.178404) + (low * (1 - 0.178404))), 13), self.ts_sum(adv120, 13), 17))
      x2 = self.rank(self.delta(((((high + low) / 2) * 0.178404) +
                      (vwap * (1 - 0.178404))), 3.69741))
      alpha = x1 < x2
      return alpha*-1


  def alpha65(self, volume, vwap, Open):
      adv60 = self.sma(volume, 60)
      x1 = self.rank(self.correlation(
          ((Open * 0.00817205) + (vwap * (1 - 0.00817205))), self.ts_sum(adv60, 9), 6))
      x2 = self.rank((Open - self.ts_min(Open, 14)))
      alpha = x1 < x2
      return alpha*-1


  def alpha66(self, vwap, low, Open, high):
      x1 = self.rank(self.decay_linear(self.delta(vwap, 4), 7))
      x2 = (((low * 0.96633) + (low * (1 - 0.96633))) - vwap) / \
          (Open - ((high + low) / 2))
      alpha = (x1 + self.ts_rank(self.decay_linear(x2, 11), 7)) * -1
      return alpha


  def alpha68(self, volume, high, close, low):
      adv15 = self.sma(volume, 15)
      x1 = self.ts_rank(self.correlation(self.rank(high), self.rank(adv15), 9), 14)
      x2 = self.rank(self.delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))
      alpha = x1 < x2
      return alpha*-1


  def alpha71(self, volume, close, low, Open, vwap):
      adv180 = self.sma(volume, 180)
      x1 = self.ts_rank(self.decay_linear(self.correlation(
          self.ts_rank(close, 3), self.ts_rank(adv180, 12), 18), 4), 16)
      x2 = self.ts_rank(self.decay_linear(
          (self.rank(((low + Open) - (vwap + vwap))).pow(2)), 16), 4)
      alpha = x1
      alpha[x1 < x2] = x2
      return alpha


  def alpha72(self, volume, high, low, vwap):
      adv40 = self.sma(volume, 40)
      x1 = self.rank(self.decay_linear(self.correlation(((high + low) / 2), adv40, 9), 10))
      x2 = self.rank(self.decay_linear(self.correlation(
          self.ts_rank(vwap, 4), self.ts_rank(volume, 19), 7), 3))
      alpha = (x1 / x2.replace(0, 0.0001))
      return alpha


  def alpha73(self, vwap, Open, low):
      x1 = self.rank(self.decay_linear(self.delta(vwap, 5), 3))
      x2 = self.delta(((Open * 0.147155) + (low * (1 - 0.147155))), 2) / \
          ((Open * 0.147155) + (low * (1 - 0.147155)))
      x3 = self.ts_rank(self.decay_linear((x2 * -1), 3), 17)
      alpha = x1
      alpha[x1 < x3] = x3
      return -1 * alpha


  def alpha74(self, volume, close, high, vwap):
      adv30 = self.sma(volume, 30)
      x1 = self.rank(self.correlation(close, self.ts_sum(adv30, 37), 15))
      x2 = self.rank(self.correlation(
          self.rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), self.rank(volume), 11))
      alpha = x1 < x2
      return alpha*-1


  def alpha75(self, volume, vwap, low):
      adv50 = self.sma(volume, 50)
      alpha = self.rank(self.correlation(vwap, volume, 4)) < self.rank(
          self.correlation(self.rank(low), self.rank(adv50), 12))
      return alpha


  def alpha77(self, volume, high, low, vwap):
      adv40 = self.sma(volume, 40)
      x1 = self.rank(self.decay_linear(((((high + low) / 2) + high) - (vwap + high)), 20))
      x2 = self.rank(self.decay_linear(self.correlation(((high + low) / 2), adv40, 3), 6))
      alpha = x1
      alpha[x1 > x2] = x2
      return alpha


  def alpha78(self, volume, low, vwap):
      adv40 = self.sma(volume, 40)
      x1 = self.rank(self.correlation(
          self.ts_sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 20), self.ts_sum(adv40, 20), 7))
      x2 = self.rank(self.correlation(self.rank(vwap), self.rank(volume), 6))
      alpha = x1.pow(x2)
      return alpha


  def alpha81(self, volume, vwap):
      adv10 = self.sma(volume, 10)
      x1 = self.rank(np.log(
          self.product(self.rank((self.rank(self.correlation(vwap, self.ts_sum(adv10, 50), 8)).pow(4))), 15)))
      x2 = self.rank(self.correlation(self.rank(vwap), self.rank(volume), 5))
      alpha = x1 < x2
      return alpha*-1


  def alpha83(self, high, low, close, volume, vwap):
      x = self.rank(self.delay(((high - low) / (self.ts_sum(close, 5) / 5)), 2)) * \
          self.rank(self.rank(volume))
      alpha = x / (((high - low) / (self.ts_sum(close, 5) / 5)) / (vwap - close))
      return alpha


  def alpha84(self, vwap, close):
      alpha = pow(self.ts_rank((vwap - self.ts_max(vwap, 15)), 21), self.delta(close, 5))
      return alpha


  def alpha85(self, volume, high, close, low):
      adv30 = self.sma(volume, 30)
      x1 = self.rank(self.correlation(
          ((high * 0.876703) + (close * (1 - 0.876703))), adv30, 10))
      alpha = x1.pow(
          self.rank(self.correlation(self.ts_rank(((high + low) / 2), 4), self.ts_rank(volume, 10), 7)))
      return alpha


  def alpha86(self, volume, close, Open, vwap):
      adv20 = self.sma(volume, 20)
      x1 = self.ts_rank(self.correlation(close, self.sma(adv20, 15), 6), 20)
      x2 = self.rank(((Open + close) - (vwap + Open)))
      alpha = x1 < x2
      return alpha*-1


  def alpha88(self, volume, Open, low, high, close):
      adv60 = self.sma(volume, 60)
      x1 = self.rank(self.decay_linear(
          ((self.rank(Open) + self.rank(low)) - (self.rank(high) + self.rank(close))), 8))
      x2 = self.ts_rank(self.decay_linear(self.correlation(
          self.ts_rank(close, 8), self.ts_rank(adv60, 21), 8), 7), 3)
      alpha = x1
      alpha[x1 > x2] = x2
      return alpha


  def alpha92(self, volume, high, low, close, Open):
      adv30 = self.sma(volume, 30)
      x1 = self.ts_rank(self.decay_linear(
          ((((high + low) / 2) + close) < (low + Open)), 15), 19)
      x2 = self.ts_rank(self.decay_linear(self.correlation(self.rank(low), self.rank(adv30), 8), 7), 7)
      alpha = x1
      alpha[x1 > x2] = x2
      return alpha


  def alpha94(self, volume, vwap):
      adv60 = self.sma(volume, 60)
      x = self.rank((vwap - self.ts_min(vwap, 12)))
      alpha = x.pow(
          self.ts_rank(self.correlation(self.ts_rank(vwap, 20), self.ts_rank(adv60, 4), 18), 3)) * -1
      return alpha


  def alpha95(self, volume, high, low, Open):
      adv40 = self.sma(volume, 40)
      x = self.ts_rank(
          (self.rank(self.correlation(self.sma(((high + low) / 2), 19), self.sma(adv40, 19), 13)).pow(5)), 12)
      alpha = self.rank((Open - self.ts_min(Open, 12))) < x
      return alpha


  def alpha96(self, volume, vwap, close):
      adv60 = self.sma(volume, 60)
      x1 = self.ts_rank(self.decay_linear(self.correlation(self.rank(vwap), self.rank(volume), 4), 4), 8)
      x2 = self.ts_rank(self.decay_linear(self.ts_argmax(self.correlation(
          self.ts_rank(close, 7), self.ts_rank(adv60, 4), 4), 13), 14), 13)
      alpha = x1
      alpha[x1 < x2] = x2
      return alpha


  def alpha98(self, volume, Open, vwap):
      adv5 = self.sma(volume, 5)
      adv15 = self.sma(volume, 15)
      x1 = self.rank(self.decay_linear(self.correlation(vwap, self.sma(adv5, 26), 5), 7))
      alpha = x1 - \
          self.rank(self.decay_linear(
              self.ts_rank(self.ts_argmin(self.correlation(self.rank(Open), self.rank(adv15), 21), 9), 7), 8))
      return alpha


  def alpha99(self, volume, high, low):
      adv60 = self.sma(volume, 60)
      x1 = self.rank(self.correlation(self.ts_sum(((high + low) / 2), 20), self.ts_sum(adv60, 20), 9))
      x2 = self.rank(self.correlation(low, volume, 6))
      alpha = x1 < x2
      return alpha*-1


  def alpha101(self, close, Open, high, low):
      alpha = (close - Open) / ((high - low) + 0.001)
      return alpha


  def alpha48(self, close_ind, close, ind):
      r1 = (self.correlation(self.delta(close_ind, 1), self.delta(self.delay(close_ind, 1), 1), 250)
            * self.delta(close_ind, 1)) / close
      r2 = self.ts_sum((pow((self.delta(close, 1) / self.delay(close, 1)), 2)), 250)
      # alpha = IndNeutralize(r1, ind) / r2
      alpha = r1 / r2
      return alpha


  def alpha58(self, vwap_ind, volume, ind):
      x = vwap_ind
      # x = IndNeutralize(vwap, ind)
      alpha = -1 * self.ts_rank(self.decay_linear(self.correlation(x, volume, 4), 8), 6)
      return alpha


  def alpha59(self, vwap_ind, volume, ind):
      x = vwap_ind
      # x = IndNeutralize(((vwap * 0.728317) + (vwap * (1 - 0.728317))), ind)
      alpha = -1 * self.ts_rank(self.decay_linear(self.correlation(x, volume, 4), 16), 8)
      return alpha


  def alpha63(self, volume, close_ind, vwap, Open, ind):
      adv180 = self.sma(volume, 180).fillna(value=0)
      # r1 = self.rank(self.decay_linear(self.delta(IndNeutralize(close, ind), 2), 8))
      r1 = self.rank(self.decay_linear(self.delta(close_ind, 2), 8))
      r2 = self.rank(self.decay_linear(self.correlation(
          ((vwap * 0.318108) + (Open * (1 - 0.318108))), self.ts_sum(adv180, 37), 14), 12))
      alpha = -1 * (r1 - r2)
      return alpha


  def alpha67(self, volume, vwap_ind, adv20_ind, high, ind):
      # adv20 = self.sma(volume, 20)
      # r = self.rank(self.correlation(IndNeutralize(vwap, ind), IndNeutralize(adv20, ind), 6))
      r = self.rank(self.correlation(vwap_ind, adv20_ind, 6))
      alpha = pow(self.rank(high - self.ts_min(high, 2)), r) * -1
      return alpha


  def alpha69(self, volume, vwap, vwap_ind, ind, close):
      adv20 = self.sma(volume, 20)
      # r1 = self.rank(self.ts_max(self.delta(IndNeutralize(vwap, ind), 3), 5))
      r1 = self.rank(self.ts_max(self.delta(vwap_ind, 3), 5))
      r2 = self.ts_rank(self.correlation(
          ((close * 0.490655) + (vwap * (1 - 0.490655))), adv20, 5), 9)
      alpha = pow(r1, r2) * -1
      return alpha


  def alpha70(self, volume, close_ind, ind, vwap):
      adv50 = self.sma(volume, 50).fillna(value=0)
      # r = self.ts_rank(self.correlation(IndNeutralize(close, ind), adv50, 18), 18)
      r = self.ts_rank(self.correlation(close_ind, adv50, 18), 18)
      
      alpha = pow(self.rank(self.delta(vwap, 1)), r) * -1
      return alpha


  def alpha76(self, volume, vwap, low_ind, ind):
      adv81 = self.sma(volume, 81).fillna(value=0)
      r1 = self.rank(self.decay_linear(self.delta(vwap, 1), 12))
      # r2 = self.ts_rank(self.decay_linear(self.ts_rank(self.correlation(IndNeutralize(low, ind), adv81, 8), 20), 17), 19)
      r2 = self.ts_rank(self.decay_linear(self.ts_rank(self.correlation(low_ind, adv81, 8), 20), 17), 19)
      alpha = r1
      alpha[r1 < r2] = r2
      return alpha


  def alpha79(self, volume, close, Open, co_mixed_ind, ind, vwap):
      adv150 = self.sma(volume, 150).fillna(value=0)
      # r1 = self.rank(self.delta(IndNeutralize(((close * 0.60733) + (Open * (1 - 0.60733))), ind), 1))
      r1 = self.rank(self.delta(co_mixed_ind , 1))
      r2 = self.rank(self.correlation(self.ts_rank(vwap, 4), self.ts_rank(adv150, 9), 15))
      alpha = (r1 < r2) * -1
      return alpha


  def alpha80(self, Open, high, volume, oh_mixed_ind, ind):
      adv10 = self.sma(volume, 10)
      # r1 = self.rank(np.sign(self.delta(IndNeutralize(((Open * 0.868128) + (high * (1 - 0.868128))), ind), 4)))
      r1 = self.rank(np.sign(self.delta(oh_mixed_ind, 4)))
      r2 = self.ts_rank(self.correlation(high, adv10, 5), 6)
      alpha = pow(r1, r2) * -1
      return alpha


  def alpha82(self, Open, volume, volume_ind, ind):
      r1 = self.rank(self.decay_linear(self.delta(Open, 1), 15))
      # r2 = self.ts_rank(self.decay_linear(self.correlation(IndNeutralize(volume, ind),
      #                                       ((Open * 0.634196) + (Open * (1 - 0.634196))), 17), 7), 13)
      r2 = self.ts_rank(self.decay_linear(self.correlation(volume_ind,
                                        ((Open * 0.634196) + (Open * (1 - 0.634196))), 17), 7), 13)
      alpha = r1
      alpha[r1 > r2] = r2
      return -1 * alpha


  def alpha87(self, volume, close, vwap, adv81_ind, ind):
      # adv81 = self.sma(volume, 81).fillna(value=0)
      r1 = self.rank(self.decay_linear(
          self.delta(((close * 0.369701) + (vwap * (1 - 0.369701))), 2), 3))
      # r2 = self.ts_rank(self.decay_linear(
      #     abs(self.correlation(IndNeutralize(adv81, ind), close, 13)), 5), 14)
      r2 = self.ts_rank(self.decay_linear(
          abs(self.correlation(adv81_ind, close, 13)), 5), 14)
      alpha = r1
      alpha[r1 < r2] = r2
      return -1 * alpha


  def alpha89(self, low, vwap, vwap_ind, ind, volume):
      adv10 = self.sma(volume, 10)
      r1 = self.ts_rank(self.decay_linear(self.correlation(
          ((low * 0.967285) + (low * (1 - 0.967285))), adv10, 7), 6), 4)
      # r2 = self.ts_rank(self.decay_linear(self.delta(IndNeutralize(vwap, ind), 3), 10), 15)
      r2 = self.ts_rank(self.decay_linear(self.delta(vwap_ind, 3), 10), 15)
      alpha = r1 - r2
      return alpha


  def alpha90(self, volume, close, adv40_ind, ind, low):
      # adv40 = self.sma(volume, 40).fillna(value=0)
      r1 = self.rank((close - self.ts_max(close, 5)))
      r2 = self.ts_rank(self.correlation(adv40_ind, low, 5), 3)
      alpha = pow(r1, r2) * -1
      return alpha


  def alpha91(self, close, close_ind, ind, volume, vwap):
      adv30 = self.sma(volume, 30)
      # r1 = self.ts_rank(self.decay_linear(self.decay_linear(self.correlation(
      #     IndNeutralize(close, ind), volume, 10), 16), 4), 5)
      r1 = self.ts_rank(self.decay_linear(self.decay_linear(self.correlation(
          close_ind, volume, 10), 16), 4), 5)
      r2 = self.rank(self.decay_linear(self.correlation(vwap, adv30, 4), 3))
      alpha = (r1 - r2) * -1
      return alpha


  def alpha93(self, vwap, vwap_ind, ind, volume, close):
      adv81 = self.sma(volume, 81).fillna(value=0)
      r1 = self.ts_rank(self.decay_linear(self.correlation(
          vwap_ind, adv81, 17), 20), 8)
      r2 = self.rank(self.decay_linear(
          self.delta(((close * 0.524434) + (vwap * (1 - 0.524434))), 3), 16))
      alpha = r1 / r2
      return alpha


  def alpha97(self, volume, low, lv_mixed_ind, vwap, ind):
      adv60 = self.sma(volume, 60).fillna(value=0)
      # r1 = self.rank(self.decay_linear(self.delta(IndNeutralize(((low * 0.721001) + (vwap * (1 - 0.721001))), ind), 3), 20))
      r1 = self.rank(self.decay_linear(self.delta(lv_mixed_ind, 3), 20))
      r2 = self.ts_rank(self.decay_linear(
          self.ts_rank(self.correlation(self.ts_rank(low, 8), self.ts_rank(adv60, 17), 5), 19), 16), 7)
      alpha = (r1 - r2) * -1
      return alpha


  def alpha100(self, volume_ind, close_ind, low_ind, high_ind, ind):
      adv20 = self.sma(volume_ind, 20)
      # r1 = IndNeutralize(self.rank(((((close - low) - (high - close)) / (high - low)) * volume)), ind)
      # r2 = 1.5 * self.scale(IndNeutralize(r1, ind))
      # r3 = self.scale(IndNeutralize((self.correlation(close, self.rank(adv20), 5) - self.rank(self.ts_argmin(close, 30))), ind))
      r1 = self.rank(((((close_ind - low_ind) - (high_ind - close_ind)) / (high_ind - low_ind)) * volume_ind))
      r2 = 1.5 * self.scale(r1)
      r3 = self.scale((self.correlation(close_ind, self.rank(adv20), 5) - self.rank(self.ts_argmin(close_ind, 30))))

      alpha = -1 * (r2 - r3) * (volume_ind / adv20)
      return alpha

