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


from quant_free.utils.us_equity_utils import *
from quant_free.dataset.xq_data_load import *



class factors_classifier(Strategy):
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
        "factor_name":"Alpha101",
        # "symbol": "QCOM",
        "symbol": "TSM",
        # "symbol": "INTC",
        # "symbol": "AAPL",
        "model": "LDA", # SVM or QDA LDA
        "quantity": 10,
        "forward_period": 5,
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
      like1 = self.parameters["factor_name"].lower()[:4]
      like2 = 'ret_backward_'
      factor = factor.replace({True: 1, False: 0})
      factor = factor.loc[:, (factor != 0).any(axis=0)]

      factor.columns = factor.columns.str.lower()
      filtered_1 = factor.filter(like=like1)
      filtered_2 = factor.filter(like=like2)
      trnsX = pd.concat([filtered_1, filtered_2], axis=1)
      return trnsX

    def load_factor_model_train(self, symbol):


      factor = multi_sym_daily_load(
          self.market,
          symbols = [symbol],
          start_date = self.parameters["training_start_date"],
          end_date = self.parameters["training_end_date"],
          column_option = "all",
          # dir_option = 'xq',
          file_name = self.parameters["factor_name"] + '.csv')[symbol]

      trnsX = self.factor_filter(factor)

      print(f"used training factor: {trnsX.columns}")

      y_data = factor.loc[:, f'ret_forward_{self.parameters['forward_period']}']
      cont = pd.DataFrame(y_data.map(lambda x: 1 if x > 0 else 0 if x < 0 else 0))
      cont = pd.concat([cont, y_data], axis = 1)
      cont.columns = ['bin', 'price_ratio']
      cont['t1'] = cont.index


      if self.parameters["model"] == "SVM":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        
        # Create pipeline with scaling and SVM
        self.fit = make_pipeline(
            StandardScaler(),
            # SVC(
            #     kernel='poly',
            #     class_weight='balanced',
            #     probability=True,
            #     random_state=42,
            #     gamma='scale'
            # )
            SVC(
                kernel='linear', C=0.5, class_weight='balanced'
            )
        ).fit(X = trnsX, y = cont['bin'])

      elif self.parameters["model"] == "QDA":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

        # Create pipeline with scaling and QDA
        self.fit = make_pipeline(
            StandardScaler(),
            QuadraticDiscriminantAnalysis()
        ).fit(X = trnsX, y = cont['bin'])
      elif self.parameters["model"] == "LDA":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        # Create pipeline with scaling and LDA
        self.fit = make_pipeline(
            StandardScaler(),
            LinearDiscriminantAnalysis()
        ).fit(X = trnsX, y = cont['bin'])
      elif self.parameters["model"] == "RAND_FOREST":

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
      else:
         print("pls configure model correctly:", self.parameters["model"])

      # Calculate accuracy score
      train_pred = self.fit.predict(trnsX)
      accuracy = accuracy_score(cont['bin'], train_pred)
      print(f"Training accuracy: {accuracy}")

      test_factors = multi_sym_daily_load(
          self.market,
          symbols = [symbol],
          start_date = self.parameters["test_start_date"],
          end_date = self.parameters["test_end_date"],
          column_option = "all",
          # dir_option = 'xq',
          file_name = self.parameters["factor_name"] + '.csv')[symbol]
      self.test_factors = trnsX = self.factor_filter(test_factors)

    #   print(f"used test factor: {trnsX.columns}")

    # Calculate accuracy score
      y_data = test_factors.loc[:, f'ret_forward_{self.parameters['forward_period']}']
      cont = pd.DataFrame(y_data.map(lambda x: 1 if x > 0 else 0 if x < 0 else 0))
      cont = pd.concat([cont, y_data], axis = 1)
      cont.columns = ['bin', 'price_ratio']
      cont['t1'] = cont.index

      test_pred = self.fit.predict(trnsX)
      accuracy = accuracy_score(cont['bin'], test_pred)
      print(f"Test accuracy: {accuracy}")

      if (accuracy < 0.6):
         sys.exit(0)

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
          positions = self.get_positions()
          # for position in positions:
          # if position.asset == Asset(symbol=symbol, asset_type="stock"):
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

    # Show help if requested or no args
    if len(sys.argv) == 1 or sys.argv[1].lower() in ['-h', '--help']:

        print("""
            Usage: python classifier.py [SYMBOL] [MODEL] [factor_name]

            Parameters:
            SYMBOL          Stock ticker symbol to trade (default: TSM)
            MODEL           Classification model to use: SVM, QDA LDA, rand_forest (default: LDA)
            factor_name     used factor option: Alpha101, Trend (default: Alpha101)

            Examples:
            python classifier.py AAPL LDA Alpha101
            python classifier.py QCOM SVM Alpha101
            python classifier.py QCOM rand_forest Trend
            """)
        sys.exit(0)

    # Get symbol from command line if provided
    if len(sys.argv) > 1:
        symbol = sys.argv[1].upper()
        factors_classifier.parameters["symbol"] = symbol
    if len(sys.argv) > 2:
        model = sys.argv[2].upper()
        factors_classifier.parameters["model"] = model
    if len(sys.argv) > 3:
        factor_name = sys.argv[3]
        factors_classifier.parameters["factor_name"] = factor_name

    print(f"symbol: {symbol} model: {model}")
    if is_live:
        ####
        # Run the strategy
        ####

        ac = AlpacaConfig(False)

        broker = Alpaca(ac)

        strategy = factors_classifier(
            broker=broker,
        )

        trader = Trader()
        trader.add_strategy(strategy)
        trader.run_all()

    else:
        ####
        # Backtest
        ####

        backtesting_start = pd.to_datetime(factors_classifier.parameters["test_start_date"])
        backtesting_end = pd.to_datetime(factors_classifier.parameters["test_end_date"])

        ####
        # Get and Organize Data
        ####
        symbol = factors_classifier.parameters["symbol"] # "AAPL"
        asset = Asset(symbol=symbol, asset_type="stock")

        df = multi_sym_daily_load(
            market = 'us'
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

        factors_classifier.backtest(
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
        backtest_store_result(os.getenv("QUANT_FREE_ROOT"), factors_classifier.parameters)
