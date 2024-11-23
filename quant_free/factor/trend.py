
from factorlab.feature_engineering.factors.trend import Trend as TrendLib

from quant_free.factor.base import FactorBase


class Trend(FactorBase):

  def __init__(self, start_date, end_date, dir = 'fh'):

    super().__init__(start_date, end_date, dir)

  def preprocess(self, data):
    return TrendLib(data, vwap='vwap', log='log')

  def trend_breakout(self, df_trend):
    """
      Computes the breakout trend factor.

    Parameters
    ----------
    method: str, {'min-max', 'percentile', 'norm'}, default 'min-max'
        Method to use to normalize price series between 0 and 1.

      Returns
      -------
    breakout: pd.DataFrame
        DataFrame with DatetimeIndex and breakout signal values (cols).
    """
    return df_trend.breakout()

  def trend_price_mom(self, df_trend):
    """
    Computes the price momentum trend factor. df_trend.diff(self.windows)/df_trend.disperse()

    Returns
    -------
    mom: pd.DataFrame
        DataFrame with DatetimeIndex and price momentum values (cols).
    """
    return df_trend.price_mom()
    
  def trend_ewma(self, df_trend):
    """
    Computes the price momentum trend factor. (df_trend.diff(lags)/df_trend.disperse()).ewma(self.windows).mean()

    Returns
    -------
    mom: pd.DataFrame
        DataFrame with DatetimeIndex and price momentum values (cols).
    """
    return df_trend.ewma()

  def trend_divergence(self, df_trend):
    """
    np.sign(self.price.diff()).smooth()
    """
    return df_trend.divergence()

  def trend_time_trend(self, df_trend):
    '''
    it's OLS model trend parameters
    formula price = alpha + beta1 * t
    beta1 is the acc output
    '''
    return df_trend.time_trend()

  def trend_time_price_acc(self, df_trend):
    '''
    it's OLS model acc trend parameters
    
    formula price = alpha + beta1 * t + beta2 * tt
    beta2 is the acc output
    
    '''
    return df_trend.price_acc()

  def trend_alpha_mom(self, df_trend):
    """
    Constant term/coefficient (alpha) from fitting an OLS linear regression of price on the market portfolio (beta,
    i.e. cross-sectional average of returns).
    
    formula r = beta * r_market + alpha

    Returns
    -------
    alpha: pd.DataFrame
        DataFrame with DatetimeIndex and alpha values (cols).
    """
    return df_trend.alpha_mom()

  def trend_rsi(self, df_trend):
    """
    Computes the RSI indicator.

    formula RS = Average Gain / Average Loss
            RSI = 100 - (100 / (1 + RS))
            (RSI - 50) / 50
    Parameters
    ----------
    signal: bool, default True
        Converts RSI to a signal between -1 and 1.
        Typically, RSI is normalized to between 0 and 100.

    Returns
    -------
    rsi: pd.DataFrame - MultiIndex
        DataFrame with DatetimeIndex and RSI indicator (cols).
    """
    return df_trend.rsi()

  def trend_stochastic(self, df_trend):
    """
    Computes the stochastic indicator K and D.

    k = (self.df.close - self.df.low.rolling(self.window_size).min()) / \
         (self.df.high.rolling(self.window_size).max() - self.df.low.rolling(self.window_size).min())
    d = k.smooth

    Parameters
    ----------
    stochastic: str, {'k', 'd', 'all'}, default 'd'
        Stochastic to return.
    signal: bool, default True
        Converts stochastic to a signal between -1 and 1.

    Returns
    -------
    stochastic k, d: pd.Series or pd.DataFrame - MultiIndex
        DataFrame with DatetimeIndex and Stochastic indicator.
    """
    return df_trend.stochastic()

  def trend_intensity(self, df_trend):
    """
    Computes intraday intensity trend factor.
    
    tr = max(today.high - today.low, today.high - yesterday.close, today.low - yesterday.close)
    chg = (today.close - today.open)
    smooth (chg / tr)

    Returns
    -------
    intensity: pd.DataFrame
        DataFrame with DatetimeIndex and intensity values (cols).
    """
    return df_trend.intensity()

  def trend_mw_diff(self, df_trend):
    """
    Computes the moving window difference trend factor.

    Returns
    -------
    mw_diff: pd.Series or pd.DataFrame - MultiIndex
        Series with DatetimeIndex (level 0), ticker (level 1) and
          moving window difference trend factor values (cols).
    """
    return df_trend.mw_diff()

  def trend_ewma_diff(self, df_trend):
    """
    Computes the exponentially weighted moving average (EWMA) crossover trend factor.

    A CTA-momentum signal, based on the cross-over of multiple exponentially weighted moving averages (EWMA) with
    different half-lives.

    Computed as described in Dissecting Investment Strategies in the Cross-Section and Time Series:
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2695101

    Parameters
    ----------
    s_k: list of int, default [2, 4, 8]
        Represents n for short window where halflife is given by log(0.5)/log(1 − 1/n).
    l_k: list of int, default [6, 12, 24]
        Represents n for long window where halflife is given by log(0.5)/log(1 − 1/n).
    signal: bool, False
        Converts normalized ewma crossover values to signal between [-1,1].

    Returns
    -------
    ewma_xover: pd.Series or pd.DataFrame
        Series or DataFrame with DatetimeIndex (level 0), tickers (level 1) and
        ewma crossover trend factor values (cols).
    """
    return df_trend.ewma_diff()

  def trend_energy(self, df_trend):
    return df_trend.energy()

  def trend_snr(self, df_trend):
    '''
    chg = self.price.diff(self.window_size)
    abs_roll_chg = np.abs(self.price.diff()).rolling(self.window_size).sum()
    # compute snr
    self.trend = chg / abs_roll_chg
    '''
    return df_trend.snr()

  def trend_adx(self, df_trend):
    '''
    If High(i) - High(i-1) > 0  dm_plus(i) = High[(i) - High(i-1), otherwise dm_plus(i) = 0.
    If Low(i-1) - Low(i) > 0  dm_minus(i) = Low(i-1) - Low(i), otherwise dm_minus(i) = 0.
    tr(i) = Max(ABS(High(i) - High(i-1)), ABS(High(i) - Close(i-1)), ABS(Low(i) - Close(i-1)))
    ATR(i) = SMMA(tr, Period_ADX,i)

    Plus_D(i) = SMMA(dm_plus, Period_ADX,i)/ATR(i)*100

    Minus_D(i) = SMMA(dm_minus, Period_ADX,i)/ATR(i)*100
    DX(i) = ABS(Plus_D(i) - Minus_D(i))/(Plus_D(i) + Minus_D(i)) * 100
    ADX(i) = SMMA(DX, Period_ADX, i)
    '''

    return df_trend.adx()