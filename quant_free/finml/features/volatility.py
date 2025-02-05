import numpy as np
import pandas as pd
import arch

def daily_volatility(close: pd.Series, lookback: int = 100) -> pd.Series:
    """
    Calculates daily volatility using exponentially weighted moving standard deviation.
    
    The function works as follows:
    
    1. df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
       - Finds the index positions of the previous trading day for each date
       - Uses searchsorted to efficiently find where each date minus 1 day would fit in the index
       
    2. df0 = df0[df0 > 0]
       - Filters out any indices where the previous day is not available (index 0)
       
    3. df0 = (pd.Series(close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0]:]))
       - Creates a mapping between current dates and their previous trading days
       - The index contains current dates, values contain corresponding previous dates
       
    4. df0 = close.loc[df0.index] / close.loc[df0].values - 1
       - Calculates daily returns as percentage change from previous day
       - close.loc[df0.index] gets current prices
       - close.loc[df0].values gets previous day prices
       - Division and subtraction calculates percentage change
       
    5. df0 = df0.ewm(span = lookback).std()
       - Computes exponentially weighted moving standard deviation of returns
       - Uses a span of 'lookback' periods (default 100)
       - This gives more weight to recent observations
       
    Parameters:
        close (pd.Series): Series of closing prices with datetime index
        lookback (int): Span for the exponential moving window (default 100)
        
    Returns:
        pd.Series: Series of daily volatility values
    """
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = (pd.Series(close.index[df0 - 1],
                     index=close.index[close.shape[0] - df0.shape[0]:]))
    df0 = close.loc[df0.index] / close.loc[df0].values - 1  # daily rets
    df0 = df0.ewm(span = lookback).std()
    return df0
def parkinson_volatility(high: pd.Series, low : pd.Series, window : int = 20) -> pd.Series :
    ret = np.log(high / low)
    estimator = 1 / (4 * np.log(2)) * (ret ** 2)
    return np.sqrt(estimator.rolling(window = window).mean())

"""
def garman_class_volatility(open : pd.Series,
                            high : pd.Series,
                            low : pd.Series,
                            close : pd.Series,
                            window : int = 20) -> pd.Series :
    ret = np.log(high / low)
    close_open_ret = np.log(close / open)
    estimator = 0.5 * ret ** 2 - (2 * np.log(2) - 1) * close_open_ret ** 2
    return np.sqrt(estimator.rolling(window = window).mean())

def yang_zhang_volatility(open : pd.Series,
                          high : pd.Series,
                          low : pd.Series,
                          close : pd.Series,
                          window : int = 20) -> pd.Series :
    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    open_prev_close_ret = np.log(open / close.shift(1))
    close_prev_open_ret = np.log(close / open.shift(1))

    high_close_ret = np.log(high / close)
    high_open_ret = np.log(high / open)
    low_close_ret = np.log(low / close)
    low_open_ret = np.log(low / open)

    sigma_open_sq = 1 / (window - 1) * (open_prev_close_ret ** 2).rolling(window=window).sum()
    sigma_close_sq = 1 / (window - 1) * (close_prev_open_ret ** 2).rolling(window=window).sum()
    sigma_rs_sq = 1 / (window - 1) * (high_close_ret * high_open_ret + low_close_ret * low_open_ret).rolling(
        window=window).sum()

    return np.sqrt(sigma_open_sq + k * sigma_close_sq + (1 - k) * sigma_rs_sq)
"""

def garman_klass_volatility(series, window=21):
    """
    Function to calculate Garman-Klass volatility
    """
    a = (np.log(series['High'] / series['Low']) ** 2).rolling(window=window).mean() * 0.5
    b = (2 * np.log(2) - 1) * (np.log(series['Close'] / series['Open']) ** 2).rolling(window=window).mean()
    return np.sqrt(a - b)


def rogers_satchell_volatility(series, window=21):
    """
    Function to calculate Rogers-Satchell volatility
    """
    a = (np.log(series['High'] / series['Close']) * np.log(series['High'] / series['Open'])).rolling(
        window=window).mean()
    b = (np.log(series['Low'] / series['Close']) * np.log(series['Low'] / series['Open'])).rolling(window=window).mean()
    return np.sqrt(a + b)

def yang_zhang_volatility(series, window=21):
    """
    Function to calculate Yang-Zhang volatility
    """
    a = (np.log(series['Open'] / series['Close'].shift(1))).rolling(window=window).mean()
    vol_open = ((np.log(series['Open'] / series['Close'].shift(1)) - a) ** 2).rolling(window=window).mean()
    b = (np.log(series['Close'] / series['Open'])).rolling(window=window).mean()
    vol_close = ((np.log(series['Close'] / series['Open']) - b) ** 2).rolling(window=window).mean()
    vol_rogers_satchell = rogers_satchell_volatility(series, window)
    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    yz_volatility = np.sqrt(
        vol_open + k * vol_close + (1 - k) * (vol_rogers_satchell ** 2)
    )

    return yz_volatility

def intrinsic_entropy(series, total_volume, window=21):
    # Intrinsic Entropy Calculation
    # The intrinsic entropy measures the uncertainty in the price movement and volume.
    # It is calculated using the formula:
    #
    # h_co = -E[(log(Open/Close_{t-1}) * (Volume_t / Total_Volume) * log(Volume_{t-1} / Total_Volume))]
    # h_oc = -E[(log(Close/Open) * (Volume_t / Total_Volume) * log(Volume_t / Total_Volume))]
    # h_ohlc = -E[((log(Open/High) * log(High/Close)) + (log(Low/Open) * log(Low/Close))) * (Volume_t / Total_Volume) * log(Volume_t / Total_Volume)]
    #
    # Where E[] denotes the rolling expectation over the given window.
    # The final intrinsic entropy h is calculated as:
    # h = |h_co + k * h_oc + (1 - k) * h_ohlc|
    # Where k is a scaling factor defined as 0.34 / (1.34 + (window + 1) / (window - 1))

    h_co = - (
            np.log(series['Open'] / series['Close'].shift(1)) *
            (series['Volume'] / total_volume) *
            np.log(series['Volume'].shift(1) / total_volume)
    ).rolling(window=window).mean()

    h_oc = - (
            np.log(series['Close'] / series['Open']) *
            (series['Volume'] / total_volume) *
            np.log(series['Volume'] / total_volume)
    ).rolling(window=window).mean()

    h_ohlc = - (
            (
                    (np.log(series['Open'] / series['High']) * np.log(series['High'] / series['Close'])) +
                    (np.log(series['Low'] / series['Open']) * np.log(series['Low'] / series['Close']))
            ) * (series['Volume'] / total_volume) * np.log(series['Volume'] / total_volume)
    ).rolling(window=window).mean()

    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    h = np.abs(h_co + k * h_oc + (1 - k) * h_ohlc)
    return h

class HeteroscedasticityModels:
    def __init__(self, close : pd.Series, vol : str = 'original'):
        if vol == 'original' :
            self.ret = np.log(close / close.shift(1)).dropna()
        elif vol == 'return' :
            self.ret = close
        else :
            raise ValueError('Only [original, return] can choose')
    def arch(self, p : int = 1,
             mean : str = 'Constant',
             dist : str = 'normal') :
        model = arch.arch_model(self.ret, vol = 'ARCH', p = p, mean = mean, dist = dist)
        result = model.fit()
        return result
    def garch(self, p : int = 1,
              q : int = 1,
              mean : str = 'Constant',
              dist : str = 'normal'):
        model = arch.arch_model(self.ret, vol = 'GARCH', p = p, q = q, mean = mean, dist = dist)
        result = model.fit()
        return result
    def egarch(self, p : int = 1,
               q : int = 1,
               mean : str = 'Constant',
               dist : str = 'normal'):
        model = arch.arch_model(self.ret, vol = 'EGARCH', p = p, q = q, mean = mean, dist = dist)
        result = model.fit()
        return result
    def garchm(self, p : int = 1,
               q : int = 1,
               mean : str = 'constant',
               dist : str = 'normal'):
        model = arch.arch_model(self.ret, vol = 'GARCH', p = p, q = q, mean = mean, dist = dist)
        result = model.fit()
        return result
    def figarch(self,
                p : int = 1,
                q : int = 1,
                mean : str = 'constant',
                dist : str = 'normal'):
        model = arch.arch_model(self.ret, vol = 'FIGARCH', p = p, q = q, mean = mean, dist = dist)
        result = model.fit()
        return result