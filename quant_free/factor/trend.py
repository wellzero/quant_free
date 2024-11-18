
from factorlab.feature_engineering.factors.trend import Trend as TrendLib

from quant_free.factor.base import FactorBase


class Trend(FactorBase):

  def __init__(self, start_date, end_date, dir = 'fh'):

    super().__init__(start_date, end_date, dir)

  def preprocess(self, data):
    return TrendLib(data, vwap='vwap', log='log')

  def trend_breakout(self, df_trend):
    return df_trend.breakout()

  def trend_price_mom(self, df_trend):
    return df_trend.price_mom()
    
  def trend_ewma(self, df_trend):
    return df_trend.ewma()

  def trend_divergence(self, df_trend):
    return df_trend.divergence()

  def trend_time_trend(self, df_trend):
    return df_trend.time_trend()

  def trend_time_price_acc(self, df_trend):
    return df_trend.price_acc()

  def trend_alpha_mom(self, df_trend):
    return df_trend.alpha_mom()

  def trend_rsi(self, df_trend):
    return df_trend.rsi()

  def trend_stochastic(self, df_trend):
    return df_trend.stochastic()

  def trend_intensity(self, df_trend):
    return df_trend.intensity()

  def trend_mw_diff(self, df_trend):
    return df_trend.mw_diff()

  def trend_ewma_diff(self, df_trend):
    return df_trend.ewma_diff()

  def trend_energy(self, df_trend):
    return df_trend.energy()

  def trend_snr(self, df_trend):
    return df_trend.snr()

  def trend_adx(self, df_trend):
    return df_trend.adx()