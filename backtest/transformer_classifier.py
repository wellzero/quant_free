import os
import sys
from datetime import datetime, timedelta


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

from quant_free.utils.us_equity_utils import *
from quant_free.dataset.equity_load import *


import torch
import torch.nn as nn
import torch.optim as optim

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
 
    def forward(self, Q, K, V):
        Q = self.W_Q(Q)
        K = self.W_K(K)
        V = self.W_V(V)

        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        attention_weights = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.depth, dtype=torch.float32))
        attention_weights = torch.softmax(attention_weights, dim=-1)

        output = torch.matmul(attention_weights, V)
        output = self._combine_heads(output)

        output = self.W_O(output)
        return output
 
    def _split_heads(self, tensor):
        tensor = tensor.view(tensor.size(0), -1, self.num_heads, self.depth)
        return tensor.transpose(1, 2)
 
    def _combine_heads(self, tensor):
        tensor = tensor.transpose(1, 2).contiguous()
        return tensor.view(tensor.size(0), -1, self.num_heads * self.depth)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
 
    def forward(self, x):
        attention_output = self.attention(x, x, x)
        attention_output = self.norm1(x + attention_output)

        feedforward_output = self.feedforward(attention_output)
        output = self.norm2(attention_output + feedforward_output)
        return output


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.encoder_attention = MultiHeadAttention(d_model, num_heads)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
 
    def forward(self, x, encoder_output):
        self_attention_output = self.self_attention(x, x, x)
        self_attention_output = self.norm1(x + self_attention_output)

        encoder_attention_output = self.encoder_attention(self_attention_output, encoder_output, encoder_output)
        encoder_attention_output = self.norm2(self_attention_output + encoder_attention_output)

        feedforward_output = self.feedforward(encoder_attention_output)
        output = self.norm3(encoder_attention_output + feedforward_output)
        return output

class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, output_dim):
        super(Transformer, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.encoder_layers = nn.ModuleList([EncoderLayer(hidden_dim, num_heads) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(hidden_dim, num_heads) for _ in range(num_layers)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
 
    def forward(self, x):
        x = self.input_layer(x)
        x = x.unsqueeze(1)  # Add sequence dimension (batch, 1, hidden_dim)

        encoder_output = x.transpose(0, 1)  # (1, batch, hidden_dim)

        #encoder
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output)
        # Take the first token (only one here)
        encoder_output = encoder_output[0, :, :]
        #output
        output = self.output_layer(encoder_output)

        #encoder
        # for layer in self.encoder_layers:
        #     encoder_output = layer(encoder_output)
        # decoder_output = encoder_output
        # #decoder
        # for layer in self.decoder_layers:
        #     decoder_output = layer(decoder_output, encoder_output)
        # decoder_output = decoder_output[-1, :, :]
        # #output
        # output = self.output_layer(decoder_output)

        return output

class TransformerClassifier(Strategy):
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
        "model": "transformer_classifier",
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
        self.market = 'us'
        self.load_factor_model_train(self.parameters["symbol"])

    def transformer_encoder(self, inputs):
        # Transformer-like attention mechanism
        x = MultiHeadAttention(num_heads=4, key_dim=2)(inputs, inputs)
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
        input_dim = input_shape  # Fixed subscript error by removing [0]
        hidden_dim = self.parameters["dnn_units"][0]
        num_heads = 4
        num_layers = 2
        output_dim = 1
        model = Transformer(input_dim, hidden_dim, num_heads, num_layers, output_dim)
        return model

    # def build_dnn_model(self, input_shape):
    #     """Hybrid DNN-Transformer Architecture for Stock Prediction                                                                                                
                                                                                                                                                            
    #     Architecture Components:                                                                                                                                   
    #     1. Input Layer:                                                                                                                                            
    #     - Accepts feature vectors of shape (input_shape,)                                                                                                       
    #     - Initializes the neural network input                                                                                                                  
                                                                                                                                                            
    #     2. Initial Dense Layer:                                                                                                                                    
    #     - 128 units with ReLU activation                                                                                                                        
    #     - 30% dropout for regularization                                                                                                                        
    #     - Reduces dimensionality while introducing non-linearity                                                                                                
                                                                                                                                                            
    #     3. Transformer Encoder Blocks (2 layers):                                                                                                                  
    #     - MultiHeadAttention (4 heads) to capture feature interactions                                                                                          
    #     - Residual connections + LayerNormalization for stability                                                                                               
    #     - Dropout (30%) to prevent overfitting                                                                                                                  
    #     - Enables attention-based feature weighting                                                                                                             
                                                                                                                                                            
    #     4. Regularized Dense Layers:                                                                                                                               
    #     - 64 units with ReLU and L2 regularization (kernel_regularizer='l2')                                                                                    
    #     - 32 units with ReLU and L2 regularization                                                                                                              
    #     - Additional 30% dropout between layers                                                                                                                 
    #     - Balances complexity with regularization                                                                                                               
                                                                                                                                                            
    #     5. Output Layer:                                                                                                                                           
    #     - Single sigmoid unit for binary classification (buy/sell)                                                                                              
    #     - Maps model outputs to [0,1] probability range                                                                                                         
                                                                                                                                                            
    #     Compilation:                                                                                                                                               
    #     - Optimizer: Adam (1e-3 learning rate)                                                                                                                  
    #     - Loss: Binary cross-entropy (for classification)                                                                                                       
    #     - Metrics: Accuracy tracking                                                                                                                            
    #     - EarlyStopping for training stability                                                                                                                  
                                                                                                                                                            
    #     Key Design Choices:                                                                                                                                        
    #     - Combines dense layers' pattern recognition with transformer's attention                                                                                  
    #     - Regularization (dropout + L2) prevents overfitting on financial time-series                                                                              
    #     - Progressive dimensionality reduction (128 → 64 → 32) maintains efficiency                                                                                
    #     - Attention mechanisms help capture temporal dependencies in features                                                                                      
    #     """ 
    #     model = Sequential()
    #     input_layer = Input(shape=input_shape)
        
    #     # Balanced architecture
    #     x = Dense(128, activation='relu')(input_layer)
    #     x = Dropout(0.3)(x)
        
    #     # Add one transformer layer back
    #     x = self.transformer_encoder(x)
    #     x = self.transformer_encoder(x)
        
    #     # Two dense layers with moderate regularization
    #     x = Dense(64, activation='relu', kernel_regularizer='l2')(x)
    #     x = Dropout(0.3)(x)
    #     x = Dense(32, activation='relu', kernel_regularizer='l2')(x)
        
    #     model.add(Dense(1, activation='sigmoid'))
        
    #     # Balanced learning rate
    #     model.compile(optimizer=Adam(learning_rate=1e-3),
    #                 loss='binary_crossentropy',
    #                 metrics=['accuracy'])
    #     return model

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
        factor = multi_sym_daily_trade_load(
            symbols=[symbol],
            start_date=self.parameters["training_start_date"],
            end_date=self.parameters["training_end_date"],
            column_option="all",
            file_name=self.parameters["factor_name"] + '.csv'
        )[symbol]

        trnsX = self.factor_filter(factor)

        print("factor name: ", trnsX.columns)

        self.scaler.fit(trnsX)
        X_train = self.scaler.transform(trnsX)
        
        y_train = factor.loc[:, f'ret_forward_{self.parameters["forward_period"]}']
        y_train = y_train.map(lambda x: 1 if x > 0 else 0).values

        # Build and train Transformer model
        input_shape = X_train.shape[1]
        self.model = self.build_dnn_model(input_shape)

        # Full PyTorch training implementation
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                    lr=self.parameters["learning_rate"])

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

        self.model.train()  # Set training mode

        for epoch in range(self.parameters["epochs"]):
            optimizer.zero_grad()
            outputs = self.model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{self.parameters["epochs"]}, Loss: {loss.item():.4f}')

        print("Training complete.")

        # Load test data
        test_factors = multi_sym_daily_trade_load(
          self.market,
            symbols=[symbol],
            start_date=self.parameters["test_start_date"],
            end_date=self.parameters["test_end_date"],
            column_option="all",
            file_name=self.parameters["factor_name"] + '.csv'
        )[symbol]

        # Compute test labels
        self.y_test = test_factors.loc[:, f'ret_forward_{self.parameters["forward_period"]}']
        self.y_test = self.y_test.map(lambda x: 1 if x > 0 else 0).values

        self.test_factors = self.factor_filter(test_factors)

        # Evaluate model on test set
        self.model.eval()
        X_test = self.scaler.transform(self.test_factors)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(X_test_tensor)
            probabilities = torch.sigmoid(outputs)
        y_pred = (probabilities.numpy() > 0.5).astype(int)
        test_acc = accuracy_score(self.y_test, y_pred)
        print(f"Test Accuracy: {test_acc:.4f}")

    def on_trading_iteration(self):
        dt = pd.to_datetime(self.get_datetime().date())
        factor = self.test_factors.loc[[dt]]
        X = self.scaler.transform(factor)
        
        # Convert to tensor and add batch dimension
        X_tensor = torch.tensor(X, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            output = self.model(X_tensor)
            prediction = torch.sigmoid(output).item()
        
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
        Usage: python transformer_classifier.py [SYMBOL] [DNN_UNITS] [LEARNING_RATE] [DROPOUT_RATE]
        
        Parameters:
        SYMBOL          Stock ticker symbol (default: TSM)
        DNN_UNITS       Comma-separated list of layer units (e.g., 256,128,64)
        LEARNING_RATE   Learning rate for Adam optimizer (default: 0.001)
        DROPOUT_RATE    Dropout rate (default: 0.3)
        
        Examples:
        python transformer_classifier.py AAPL 256,128,64
        python transformer_classifier.py QCOM 512,256,128 0.0005 0.2
        """)
        sys.exit(0)

    # Parse command line arguments
    if len(sys.argv) > 1:
        TransformerClassifier.parameters["symbol"] = sys.argv[1].upper()
    if len(sys.argv) > 2:
        TransformerClassifier.parameters["factor_name"] = sys.argv[2]
    if len(sys.argv) > 3:
        TransformerClassifier.parameters["dnn_units"] = list(map(int, sys.argv[3].split(',')))
    if len(sys.argv) > 4:
        TransformerClassifier.parameters["learning_rate"] = float(sys.argv[4])
    if len(sys.argv) > 5:
        TransformerClassifier.parameters["dropout_rate"] = float(sys.argv[5])

    # Backtesting setup
    backtesting_start = pd.to_datetime(TransformerClassifier.parameters["test_start_date"])
    backtesting_end = pd.to_datetime(TransformerClassifier.parameters["test_end_date"])
    symbol = TransformerClassifier.parameters["symbol"]
    
    df = multi_sym_daily_trade_load(
        market = 'us',
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

    TransformerClassifier.backtest(
        PandasDataBacktesting,
        backtesting_start,
        backtesting_end,
        pandas_data=pandas_data,
        benchmark_asset=symbol,
        sleeptime='1D',
        parameters={"asset": Asset(symbol=symbol, asset_type="stock")}
    )

    from quant_free.utils.backtest_utill import *
    backtest_store_result(os.getenv("QUANT_FREE_ROOT"), TransformerClassifier.parameters)
