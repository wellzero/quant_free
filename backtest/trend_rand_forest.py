import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import ta
from autots import AutoTS
from lumibot.backtesting import PandasDataBacktesting
from lumibot.brokers import Alpaca
from lumibot.entities import Asset, Data
from lumibot.entities.asset import Asset
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from sklearn.ensemble import RandomForestRegressor


from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from credentials import AlpacaConfig

import sys
from pathlib import Path
_this_dir = Path(__file__).parent.parent
sys.path.append(str(_this_dir))
from quant_free.utils.us_equity_utils import *
from quant_free.dataset.us_equity_load import *



class Trend(Strategy):
    """Parameters:

    symbol (str, optional): The symbol that we want to trade. Defaults to "SRNE".
    compute_frequency (int, optional): The time (in minutes) that we should retrain our model.
    lookback_period (int, optional): The amount of data (in minutes) that we get from our data source to use in the model.
    pct_portfolio_per_trade (float, optional): The size that each trade will be (in percent of the total portfolio).
    price_change_threshold_up (float, optional): The difference between predicted price and the current price that will trigger a buy order (in percentage change).
    price_change_threshold_down (float, optional): The difference between predicted price and the current price that will trigger a sell order (in percentage change).
    max_pct_portfolio (float, optional): The maximum that the strategy will buy or sell as a percentage of the portfolio (eg. if this is 0.8 - or 80% - and our portfolio is worth $100k, then we will stop buying when we own $80k worth of the symbol)
    take_profit_factor: Where you place your limit order based on the prediction
    stop_loss_factor: Where you place your stop order based on the prediction
    """

    parameters = {
        # "symbol": "QCOM",
        "symbol": "TSM",
        # "symbol": "INTC",
        # "symbol": "AAPL",
        "quantity": 10,
        "forward_period": 5,
        "factor_name": 'Trend',
        "model": 'rand_forest',
        'training_start_date': get_json_config_value("training_start_date"),
        'training_end_date': get_json_config_value("training_end_date"),
        'test_start_date': get_json_config_value("test_start_date"),
        'test_end_date': get_json_config_value("test_end_date")
    }

    def initialize(self):
      # Set the initial variables or constants

      self.sleeptime = "1D"
      

      # Variable initial states
      self.last_compute = None
      self.prediction = None
      self.last_price = None
      self.asset_value = None
      self.shares_owned = None
      self.cache_df = None
      self.market = 'us'

      self.load_factor_model_train(self.parameters["symbol"])

    def factor_filter(self, factor):
      like1 = 'trend'
      like2 = 'ret_backward_'
      factor = factor.replace({True: 1, False: 0})
      factor = factor.loc[:, (factor != 0).any(axis=0)]
      filtered_1 = factor.filter(like=like1)
      filtered_2 = factor.filter(like=like2)
      trnsX = pd.concat([filtered_1, filtered_2], axis=1)
      return trnsX

    def load_factor_model_train(self, symbol):


      factor = equity_tradedata_load_bt_dates(
          self.market,
          symbols = [symbol],
          start_date = self.parameters["training_start_date"],
          end_date = self.parameters["training_end_date"],
          column_option = "all",
          # dir_option = 'xq',
          file_name = self.__class__.__name__ + '.csv')[symbol]

      factor = factor.replace({True: 1, False: 0})
      factor = factor.loc[:, (factor != 0).any(axis=0)]

      trnsX = self.factor_filter(factor)

      y_data = factor.loc[:, f'ret_forward_{self.parameters['forward_period']}']
      cont = pd.DataFrame(y_data.map(lambda x: 1 if x > 0 else 0 if x < 0 else 0))
      cont = pd.concat([cont, y_data], axis = 1)
      cont.columns = ['bin', 'price_ratio']
      cont['t1'] = cont.index

      forest = RandomForestClassifier(
          criterion = 'log_loss',
          class_weight = 'balanced_subsample',
          min_weight_fraction_leaf = 0.0,
          random_state = 42,
          n_estimators = 1000,
          max_features = 1,
          oob_score = True,
          n_jobs = 1)
      self.fit = forest.fit(X = trnsX, y = cont['bin'])

      print(f"oob score {self.fit.oob_score_}")

      test_factors = equity_tradedata_load_bt_dates(
          self.market,
          symbols = [symbol],
          start_date = self.parameters["test_start_date"],
          end_date = self.parameters["test_end_date"],
          column_option = "all",
          # dir_option = 'xq',
          file_name = self.__class__.__name__ + '.csv')[symbol]
      self.test_factors = trnsX = self.factor_filter(test_factors)

    def on_trading_iteration(self):
        # Get parameters for this iteration
        dt = pd.to_datetime(self.get_datetime().date())

        factor = self.test_factors.loc[[dt]] 
        predict = self.fit.predict(factor)

        symbol = self.parameters["symbol"]

        if predict == 1:
          quantity = self.parameters["quantity"]
          main_order = self.create_order(
              symbol, quantity, "buy", quote=self.quote_asset
          )
          self.submit_order(main_order)
        elif predict == 0:
          
        #   positions = self.get_positions()
        #   for position in positions:
        #       if position.asset == Asset(symbol=symbol, asset_type="stock"):
        #         quantity = self.parameters["quantity"]
        #         # quantity = position.quantity
        #         main_order = self.create_order(
        #             position.asset, quantity, "sell", quote=self.quote_asset
        #         )
        #         self.submit_order(main_order)

          quantity = self.parameters["quantity"]
          # quantity = position.quantity
          main_order = self.create_order(
              symbol, quantity, "sell", quote=self.quote_asset
          )
          self.submit_order(main_order)

        else:
            print(f"not exsist pridict {predict}")

    def on_abrupt_closing(self):
        self.sell_all()

if __name__ == "__main__":
    is_live = False

    # Get symbol from command line if provided
    if len(sys.argv) > 1:
        symbol = sys.argv[1].upper()
        Trend.parameters["symbol"] = symbol

    if is_live:
        ####
        # Run the strategy
        ####

        ac = AlpacaConfig(False)

        broker = Alpaca(ac)

        strategy = Trend(
            broker=broker,
        )

        trader = Trader()
        trader.add_strategy(strategy)
        trader.run_all()

    else:
        ####
        # Backtest
        ####

        backtesting_start = pd.to_datetime(Trend.parameters["test_start_date"])
        backtesting_end = pd.to_datetime(Trend.parameters["test_end_date"])

        ####
        # Get and Organize Data
        ####
        symbol = Trend.parameters["symbol"] # "AAPL"
        asset = Asset(symbol=symbol, asset_type="stock")

        df = equity_tradedata_load_bt_dates(
            market = 'us',
            symbols = [symbol],
            start_date = backtesting_start,
            end_date = backtesting_end,
            column_option = "all",
            dir_option = 'xq')[symbol]
        
        # df = pd.read_csv(f"data/{asset}_1min.csv")
        # df = df.set_index("time")
        # df.index = pd.to_datetime(df.index)

        pandas_data = dict()
        pandas_data[asset] = Data(
            asset=asset,
            df=df,
            timestep="day",
        )

        Trend.backtest(
            PandasDataBacktesting,
            backtesting_start,
            backtesting_end,
            pandas_data=pandas_data,
            benchmark_asset=symbol,
            sleeptime = '1D',
            parameters={
                "asset": asset,
            },
        )
        from quant_free.utils.backtest_utill import *
        backtest_store_result(Path(__file__).parent, Trend.parameters)
