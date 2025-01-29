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

from quant_free.finml.labeling.labeling import *
from quant_free.finml.features.volatility import daily_volatility


#pls write a sort function AI!

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
# pls clean parameters are not used in following code AI!
    parameters = {
        # "symbol": "INTC",
        "symbol": "AAPL",
        "take_profit_price": 405,
        "stop_loss_price": 395,
        "quantity": 10,
        "forward_period": 5,
        "factor_window": 5,
        "factors": ['trend_divergence','trend_snr','trend_breakout','trend_time_trend','trend_price_mom'],
        'training_start_date': get_json_config_value("training_start_date"),
        'training_end_date': get_json_config_value("training_end_date"),
        'test_start_date': get_json_config_value("test_start_date"),
        'test_end_date': get_json_config_value("test_end_date"),
        "lookback": 60
    }

    def initialize(self):
      # Set the initial variables or constants

      self.sleeptime = "1D"
      
      self.factor_names = [f"{factor_name}_{self.parameters['factor_window']}" for factor_name in self.parameters['factors']]

      # Variable initial states
      self.last_compute = None
      self.prediction = None
      self.last_price = None
      self.asset_value = None
      self.shares_owned = None
      self.cache_df = None

      self.load_labeling(self.parameters["symbol"])

    def load_labeling(self, symbol):
       
        data = us_equity_data_load_within_range(
            symbols = [symbol],
            start_date = backtesting_start,
            end_date = backtesting_end,
            column_option = "all",
            dir_option = 'xq')[symbol]
        
        vertical_barrier = add_vertical_barrier(
            data.index, 
            data['close'], 
            num_days = 7 # expariation limit
        )

        volatility = daily_volatility(
            data['close'], 
            lookback = self.parameters['lookback'] # moving average span
        )

        triple_barrier_events = get_events(
            close = data['close'],
            t_events = data.index[2:],
            pt_sl = [1, 1], # profit taking 2, stopping loss 1
            target = volatility, # dynamic threshold
            min_ret = 0.01, # minimum position return
            num_threads = 1, # number of multi-thread 
            vertical_barrier_times = vertical_barrier, # add vertical barrier
            side_prediction = None # betting side prediction (primary model)
        )

        self.labels = meta_labeling(
            triple_barrier_events, 
            data['close'])['bin']
        self.labels.index = pd.to_datetime(self.labels.index)


    def on_trading_iteration(self):
        # Get parameters for this iteration
        dt = pd.to_datetime(self.get_datetime().date())

        
        if dt not in self.labels.index or dt < self.labels.index[self.parameters['lookback']]:
            return
            
        predict = self.labels.loc[[dt]] 

        symbol = self.parameters["symbol"]

        if predict.iloc[0] == -1:
          quantity = self.parameters["quantity"]
          main_order = self.create_order(
              symbol, quantity, "buy", quote=self.quote_asset
          )
          self.submit_order(main_order)
        elif predict.iloc[0] == 1:
          positions = self.get_positions()
          for position in positions:
              if position.asset == Asset(symbol=symbol, asset_type="stock"):
                quantity = self.parameters["quantity"]
                # quantity = position.quantity
                main_order = self.create_order(
                    position.asset, quantity, "sell", quote=self.quote_asset
                )
                self.submit_order(main_order)

        else:
            print(f"not exsist pridict {predict}")

    def on_abrupt_closing(self):
        self.sell_all()

if __name__ == "__main__":
    is_live = False

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

        df = us_equity_data_load_within_range(
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
