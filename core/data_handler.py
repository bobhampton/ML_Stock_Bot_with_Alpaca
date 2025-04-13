import pandas as pd
import numpy as np
import os
from config.env_loader import init_env
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
import ta

# Load API creds once
api_key, secret_key, _, _ = init_env()

def crypto_bars(symbol, start_date, end_date, limit, timeframe):
    try:
        crypto_data_client = CryptoHistoricalDataClient(api_key, secret_key)
        request = CryptoBarsRequest(
            symbol_or_symbols=symbol,
            start=start_date,
            end=end_date,
            limit=limit,
            timeframe=timeframe
        )
        bar_data = crypto_data_client.get_crypto_bars(request_params=request)
        return bar_data.df
    except Exception as e:
        print(f"[crypto_bars] Error fetching bars for {symbol}: {e}")
        return pd.DataFrame()

def describe_data(df):
    print("Statistical summary of raw BTC data:")
    print(df.describe())

def check_missing_and_outliers(df):
    print("\nMissing values:")
    print(df.isnull().sum())

    # Outlier check for 'close' using IQR
    Q1 = df['close'].quantile(0.25)
    Q3 = df['close'].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df['close'] < Q1 - 1.5 * IQR) | (df['close'] > Q3 + 1.5 * IQR)]
    print(f"\nDetected {len(outliers)} potential outliers in closing prices.")

def add_technical_indicators(df):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

    # Bollinger Band Width
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_width'] = bb.bollinger_wband()

    # Volume Scaled
    if 'volume' in df.columns:
        df['volume_scaled'] = (df['volume'] - df['volume'].mean()) / df['volume'].std()
    else:
        df['volume_scaled'] = 0

    # MACD
    macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd_line'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()

    # On-Balance Volume
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(
        close=df['close'],
        volume=df.get('volume', pd.Series(0))
    ).on_balance_volume()

    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=14,
        smooth_window=3
    )
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()

    # ADX
    df['adx'] = ta.trend.ADXIndicator(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=14
    ).adx()

    # CCI
    df['cci'] = ta.trend.CCIIndicator(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=20
    ).cci()

    return df

def extract_high_volatility_window(df, window_size=20, top_pct=0.3):
    df = df.copy()
    df['rolling_vol'] = df['close'].rolling(window=window_size).std()
    df = df.dropna()

    threshold = df['rolling_vol'].quantile(1 - top_pct)
    high_vol_df = df[df['rolling_vol'] >= threshold]

    return high_vol_df

