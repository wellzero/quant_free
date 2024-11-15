
from factorlab.feature_engineering.factors.trend import Trend

from quant_free.factor.base import FactorBase


class Trend(FactorBase):

  def __init__(self, start_date, end_date, dir = 'fh'):

    super().__init__(start_date, end_date, dir)