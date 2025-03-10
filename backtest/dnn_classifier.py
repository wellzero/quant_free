import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

from lumibot.backtesting import PandasDataBacktesting
from lumibot.brokers import Alpaca
from lumibot.entities import Asset, Data
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader

from credentials import AlpacaConfig
_this_dir = Path(__file__).parent.parent
sys.path.append(str(_this_dir))
from quant_free.utils.us_equity_utils import *
from quant_free.dataset.us_equity_load import *


class DNNClassifier(Strategy):
    """Parameters:
    symbol (str): Stock ticker symbol to trade
    dnn_units: Number of units in each DNN layer
    learning_rate: Learning rate for Adam optimizer
    dropout_rate: Dropout rate for regularization
    epochs: Number of training epochs
    batch_size: Training batch size
    """

    parameters = {
        "factor_name": "Alpha101",
        "symbol": "TSM",
        "model": "DNN",
        "quantity": 10,
        "forward_period": 5,
        "dnn_units": [256, 128, 64],
        "learning_rate": 1e-3,
        "dropout_rate": 0.3,
        "epochs": 100,
        "batch_size": 32,
        'training_start_date': get_json_config_value("training_start_date"),
        'training_end_date': get_json_config_value("training_end_date"),
        'test_start_date': get_json_config_value("test_start_date"),
        'test_end_date': get_json_config_value("test_end_date")
    }

    def initialize(self):
        self.sleeptime = "1D"
        self.last_compute = None
        self.prediction = None
        self.scaler = StandardScaler()
        self.load_factor_model_train(self.parameters["symbol"])

    def transformer_encoder(self, inputs):
        # Transformer-like attention mechanism
        x = MultiHeadAttention(num_heads=2, key_dim=2)(inputs, inputs)
        x = Dropout(self.parameters["dropout_rate"])(x)
        x = LayerNormalization(epsilon=1e-6)(x + inputs)
        return x
    
    # The DNN (Deep Neural Network) model consists of several key components:
    # 1. Input Layer: Accepts the feature data with a shape defined by the number of features.
    # 2. Dense Layers: Fully connected layers where each neuron is connected to every neuron in the previous layer.
    #    - The first dense layer has a number of units specified by the first element in the 'dnn_units' list and uses ReLU activation.
    #    - Subsequent dense layers have units specified by the remaining elements in the 'dnn_units' list, also using ReLU activation.
    # 3. Dropout Layers: Regularization layers that randomly set a fraction of input units to 0 at each update during training time, which helps prevent overfitting.
    #    - Dropout rate is controlled by the 'dropout_rate' parameter.
    # 4. Transformer Encoder Block: Incorporates a multi-head self-attention mechanism followed by a feed-forward neural network, similar to the architecture used in transformers.
    #    - It includes a multi-head attention layer, dropout, and layer normalization.
    # 5. Output Layer: A single neuron with a sigmoid activation function, suitable for binary classification tasks.
    # 6. Compilation: The model is compiled with the Adam optimizer, binary cross-entropy loss, and accuracy as a metric.

    def build_dnn_model(self, input_shape):
        model = Sequential()
        input_layer = Input(shape=input_shape)
        x = Dense(self.parameters["dnn_units"][0], activation='relu')(input_layer)
        
        # Transformer encoder block with proper input handling
        x = self.transformer_encoder(x)
        
        # Add dense layers
        for units in self.parameters["dnn_units"][1:]:
            x = Dense(units, activation='relu')(x)
            x = Dropout(self.parameters["dropout_rate"])(x)
            
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(optimizer=Adam(learning_rate=self.parameters["learning_rate"]),
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        return model

    def factor_filter(self, factor):
        like1 = self.parameters["factor_name"].lower()[:4]
        like2 = 'ret_backward_'
        factor = factor.replace({True: 1, False: 0})
        factor = factor.loc[:, (factor != 0).any(axis=0)]
        factor.columns = factor.columns.str.lower()
        filtered_1 = factor.filter(like=like1)
        filtered_2 = factor.filter(like=like2)
        return pd.concat([filtered_1, filtered_2], axis=1)

    def load_factor_model_train(self, symbol):
        # Load and preprocess data
        factor = us_equity_data_load_within_range(
            symbols=[symbol],
            start_date=self.parameters["training_start_date"],
            end_date=self.parameters["training_end_date"],
            column_option="all",
            file_name=self.parameters["factor_name"] + '.csv'
        )[symbol]

        trnsX = self.factor_filter(factor)
        self.scaler.fit(trnsX)
        X_train = self.scaler.transform(trnsX)
        
        y_train = factor.loc[:, f'ret_forward_{self.parameters["forward_period"]}']
        y_train = y_train.map(lambda x: 1 if x > 0 else 0).values

        # Build and train DNN model
        self.model = self.build_dnn_model((X_train.shape[1],))
        
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        self.model.fit(X_train, y_train,
                      epochs=self.parameters["epochs"],
                      batch_size=self.parameters["batch_size"],
                      validation_split=0.2,
                      callbacks=[early_stop],
                      verbose=1)

        # Load test data
        test_factors = us_equity_data_load_within_range(
            symbols=[symbol],
            start_date=self.parameters["test_start_date"],
            end_date=self.parameters["test_end_date"],
            column_option="all",
            file_name=self.parameters["factor_name"] + '.csv'
        )[symbol]
        self.test_factors = self.factor_filter(test_factors)

    def on_trading_iteration(self):
        dt = pd.to_datetime(self.get_datetime().date())
        factor = self.test_factors.loc[[dt]]
        X = self.scaler.transform(factor)
        
        prediction = self.model.predict(X, verbose=0)[0][0]
        action = 1 if prediction > 0.5 else 0
        
        symbol = self.parameters["symbol"]
        quantity = self.parameters["quantity"]
        
        if action == 1:
            order = self.create_order(symbol, quantity, "buy")
            self.submit_order(order)
        else:
            positions = self.get_positions()
            if any(pos.asset.symbol == symbol for pos in positions):
                order = self.create_order(symbol, quantity, "sell")
                self.submit_order(order)

    def on_abrupt_closing(self):
        self.sell_all()

if __name__ == "__main__":
    is_live = False

    if len(sys.argv) == 1 or sys.argv[1].lower() in ['-h', '--help']:
        print("""
        Usage: python dnn_classifier.py [SYMBOL] [DNN_UNITS] [LEARNING_RATE] [DROPOUT_RATE]
        
        Parameters:
        SYMBOL          Stock ticker symbol (default: TSM)
        DNN_UNITS       Comma-separated list of layer units (e.g., 256,128,64)
        LEARNING_RATE   Learning rate for Adam optimizer (default: 0.001)
        DROPOUT_RATE    Dropout rate (default: 0.3)
        
        Examples:
        python dnn_classifier.py AAPL 256,128,64
        python dnn_classifier.py QCOM 512,256,128 0.0005 0.2
        """)
        sys.exit(0)

    # Parse command line arguments
    if len(sys.argv) > 1:
        DNNClassifier.parameters["symbol"] = sys.argv[1].upper()
    if len(sys.argv) > 2:
        DNNClassifier.parameters["dnn_units"] = list(map(int, sys.argv[2].split(',')))
    if len(sys.argv) > 3:
        DNNClassifier.parameters["learning_rate"] = float(sys.argv[3])
    if len(sys.argv) > 4:
        DNNClassifier.parameters["dropout_rate"] = float(sys.argv[4])

    # Backtesting setup
    backtesting_start = pd.to_datetime(DNNClassifier.parameters["test_start_date"])
    backtesting_end = pd.to_datetime(DNNClassifier.parameters["test_end_date"])
    symbol = DNNClassifier.parameters["symbol"]
    
    df = us_equity_data_load_within_range(
        symbols=[symbol],
        start_date=backtesting_start,
        end_date=backtesting_end,
        column_option="all",
        dir_option='xq'
    )[symbol]

    pandas_data = {
        Asset(symbol=symbol, asset_type="stock"): Data(
            asset=Asset(symbol=symbol, asset_type="stock"),
            df=df,
            timestep="day",
        )
    }

    DNNClassifier.backtest(
        PandasDataBacktesting,
        backtesting_start,
        backtesting_end,
        pandas_data=pandas_data,
        benchmark_asset=symbol,
        sleeptime='1D',
        parameters={"asset": Asset(symbol=symbol, asset_type="stock")}
    )

    from quant_free.utils.backtest_utill import *
    backtest_store_result(Path(__file__).parent, DNNClassifier.parameters)
