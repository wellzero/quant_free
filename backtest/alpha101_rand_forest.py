import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from lumibot.backtesting import PandasDataBacktesting
from lumibot.brokers import Alpaca
from lumibot.entities import Asset, Data
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight


import sys
from pathlib import Path
_this_dir = Path(__file__).parent.parent
sys.path.append(str(_this_dir))
from credentials import AlpacaConfig
from quant_free.dataset.us_equity_load import *
from quant_free.utils.us_equity_utils import *
from quant_free.utils.backtest_utill import backtest_store_result

class Alpha101(Strategy):
    """Alpha101 Strategy using Random Forest Classifier for trading decisions."""

    parameters = {
        "model": "RandomForestClassifier",
        "factor_name": "Alpha101",
        "symbol": "INTC",
        "quantity": 10,
        "forward_period": 5,
        "training_start_date": get_json_config_value("training_start_date"),
        "training_end_date": get_json_config_value("training_end_date"),
        "test_start_date": get_json_config_value("test_start_date"),
        "test_end_date": get_json_config_value("test_end_date"),
    }

    def initialize(self):
        """Initialize strategy variables and load initial model."""
        self.sleeptime = "1D"
        self.iteration_count = 0
        self.training_threshold = 50
        self.market = 'us'
        self.load_factor_model_train(self.parameters["symbol"])

    def factor_filter(self, factor):
        """Filter factor data to include only relevant columns."""
        like1, like2 = 'alpha', 'ret_backward_'
        factor = factor.replace({True: 1, False: 0})
        factor = factor.loc[:, (factor != 0).any(axis=0)]
        filtered_1 = factor.filter(like=like1)
        filtered_2 = factor.filter(like=like2)
        return pd.concat([filtered_1, filtered_2], axis=1)

    def create_cont_dataframe(self, factor, forward_period):
        """Create a DataFrame with binary classification and price ratio."""
        y_data = factor.loc[:, f'ret_forward_{forward_period}']
        cont = pd.DataFrame(y_data.map(lambda x: 1 if x > 0 else 0 if x < 0 else 0))
        cont = pd.concat([cont, y_data], axis=1)
        cont.columns = ['bin', 'price_ratio']
        cont['t1'] = cont.index
        return cont

    def load_factor_model_train(self, symbol):
        """Load and train the Random Forest model."""
        factor = equity_tradedata_load(
            self.market,
            symbols=[symbol],
            start_date=self.parameters["training_start_date"],
            end_date=self.parameters["training_end_date"],
            column_option="all",
            file_name=self.parameters['factor_name'] + '.csv'
        )[symbol]

        factor = factor.replace({True: 1, False: 0})
        factor = factor.loc[:, (factor != 0).any(axis=0)]

        trnsX = self.factor_filter(factor)
        cont = self.create_cont_dataframe(factor, self.parameters['forward_period'])

        classes = np.unique(cont['bin'])
        class_weights = compute_class_weight('balanced', classes=classes, y=cont['bin'])
        class_weight_dict = dict(zip(classes, class_weights))

        forest = RandomForestClassifier(
            criterion='log_loss',
            class_weight=class_weight_dict,
            min_weight_fraction_leaf=0.0,
            random_state=42,
            n_estimators=50,
            max_features=5,
            oob_score=True,
            n_jobs=1,
            warm_start=True
        )
        self.fit = forest.fit(X=trnsX, y=cont['bin'])
        print(f"oob score {self.fit.oob_score_}")

        self.load_test_data(symbol)

    def load_test_data(self, symbol):
        """Load test data for evaluation."""
        test_factors = equity_tradedata_load(
            self.market,
            symbols=[symbol],
            start_date=self.parameters["test_start_date"],
            end_date=self.parameters["test_end_date"],
            column_option="all",
            file_name=self.parameters['factor_name'] + '.csv'
        )[symbol]
        self.test_factors = self.factor_filter(test_factors)
        self.test_cont = self.create_cont_dataframe(test_factors, self.parameters['forward_period'])

    def on_trading_iteration(self):
        """Execute trading logic for each iteration."""
        dt = pd.to_datetime(self.get_datetime().date())
        factor = self.test_factors.loc[[dt]]
        predict = self.fit.predict(factor)
        symbol = self.parameters["symbol"]

        if predict == 1:
            self.submit_order(self.create_order(symbol, self.parameters["quantity"], "buy", quote=self.quote_asset))
        elif predict == 0:
            self.submit_order(self.create_order(symbol, self.parameters["quantity"], "sell", quote=self.quote_asset))
        else:
            print(f"Invalid prediction {predict}")

        self.iteration_count += 1
        if self.iteration_count >= self.training_threshold:
            self.retrain_model(dt)

    def retrain_model(self, dt):
        """Retrain the model with the latest data."""
        self.iteration_count = 0
        start_idx = max(0, self.test_factors.index.get_loc(dt) - self.training_threshold)
        X_train = self.test_factors.iloc[start_idx:self.test_factors.index.get_loc(dt) + 1]
        y_train = self.test_cont.iloc[start_idx:self.test_factors.index.get_loc(dt) + 1]['bin']
        self.fit.fit(X=X_train, y=y_train)

    def on_abrupt_closing(self):
        """Sell all positions on abrupt closing."""
        self.sell_all()


if __name__ == "__main__":
    is_live = False

    if len(sys.argv) > 1:
        Alpha101.parameters["symbol"] = sys.argv[1].upper()

    if is_live:
        broker = Alpaca(AlpacaConfig(False))
        strategy = Alpha101(broker=broker)
        trader = Trader()
        trader.add_strategy(strategy)
        trader.run_all()
    else:
        backtesting_start = pd.to_datetime(Alpha101.parameters["test_start_date"])
        backtesting_end = pd.to_datetime(Alpha101.parameters["test_end_date"])
        symbol = Alpha101.parameters["symbol"]
        asset = Asset(symbol=symbol, asset_type="stock")

        df = equity_tradedata_load(
            market='us',
            symbols=[symbol],
            start_date=backtesting_start,
            end_date=backtesting_end,
            column_option="all",
            dir_option='xq'
        )[symbol]

        pandas_data = {asset: Data(asset=asset, df=df, timestep="day")}

        Alpha101.backtest(
            PandasDataBacktesting,
            backtesting_start,
            backtesting_end,
            pandas_data=pandas_data,
            benchmark_asset=symbol,
            sleeptime='1D',
            parameters={"asset": asset},
        )

        backtest_store_result(Path(__file__).parent, Alpha101.parameters)
