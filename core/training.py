import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from sklearn.preprocessing import StandardScaler

def build_hybrid_LSTM_model(input_shape, units=64, dropout=0.2):
    model = Sequential()
    model.add(Input(shape=input_shape))  # (lookback, features)
    model.add(LSTM(units, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(units))
    model.add(Dropout(dropout))
    model.add(Dense(1))  # Predict EOD price
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def load_model_from_file(path='btc_lstm_model.keras'):
    return load_model(path)

def prepare_eod_training_data_hybrid(df, lookback=120):
    df = df.copy()
    df['log_return'] = np.log(df['close'].shift(-1) / df['close'])
    df['target'] = df['log_return'] * 100  # <- Scale return

    # Feature engineering
    df['ma_20'] = df['close'].rolling(20).mean()
    df['ma_ratio'] = df['close'] / df['ma_20']
    df['volatility_20'] = df['close'].rolling(20).std()
    df['daily_range'] = (df['high'] - df['low']) / df['close']
    df['body_size'] = abs(df['close'] - df['open']) / df['close']

    feature_cols = [
        'close', 'rsi', 'bb_width', 'volume_scaled',
        'macd_line', 'macd_signal', 'macd_hist',
        'obv', 'stoch_k', 'stoch_d', 'adx', 'cci',
        'ma_ratio', 'volatility_20', 'daily_range', 'body_size'
    ]

    df = df.dropna(subset=feature_cols + ['target'])

    X, y = [], []
    for i in range(lookback, len(df)):
        X.append(df.iloc[i - lookback:i][feature_cols].values)
        y.append(df.iloc[i]['target'])

    X = np.array(X)
    y = np.array(y)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_scaled = np.array([x_scaler.fit_transform(x) for x in X])
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

    return X_scaled, y_scaled, x_scaler, y_scaler
