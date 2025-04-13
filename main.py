from datetime import datetime, timedelta, timezone
import random
import time
import os
import numpy as np
import logging
import pandas as pd
import matplotlib.pyplot as plt
import run_test
import optuna
from statistics import mean
import ta
import tensorflow as tf
import json
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, Input
from keras.callbacks import EarlyStopping
from keras import backend as K
import gc
import alpaca_trade_api as tradeapi
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import (
    CryptoBarsRequest, CryptoLatestOrderbookRequest, CryptoSnapshotRequest, CryptoTradesRequest, 
    CryptoQuoteRequest, StockBarsRequest, StockQuotesRequest, StockLatestBarRequest, NewsRequest
)
from alpaca.data.timeframe import TimeFrame
from alpaca.data.historical.news import NewsClient
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest, OrderRequest
from alpaca.trading.enums import AssetClass, OrderSide, OrderType, OrderClass, TimeInForce
from sklearn.metrics import mean_absolute_error, mean_squared_error
from collections import Counter
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()
BASE_URL = "https://paper-api.alpaca.markets"
api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")
api = tradeapi.REST(key_id=api_key, secret_key=secret_key, base_url=BASE_URL, api_version='v2')
if not api_key or not secret_key:
    raise EnvironmentError("Missing ALPACA_API_KEY or ALPACA_SECRET_KEY in environment variables.")

def set_random_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
def analyze_trade_wins(trade_log, csv_path="trade_performance_summary.csv", plot_dir="analytics"):
    import pandas as pd
    import matplotlib.pyplot as plt
    from statistics import mean
    from datetime import datetime
    import os

    os.makedirs(plot_dir, exist_ok=True)

    if not trade_log:
        print("No trades to analyze.")
        return {}

    trades = pd.DataFrame(trade_log)
    trades['date'] = pd.to_datetime(trades['date'])
    trades = trades.sort_values('date')

    # Track open and closed trades
    holding = 0
    entry_price = 0
    positions = []
    pnl_list = []

    for _, row in trades.iterrows():
        if row['decision'] == 'BUY':
            holding += 1
            entry_price += row['actual_eod']
        elif row['decision'] == 'SELL' and holding > 0:
            avg_entry = entry_price / holding
            pnl = row['actual_eod'] - avg_entry
            pnl_list.append({
                'date': row['date'],
                'pnl': pnl
            })
            positions.append(pnl)
            holding -= 1
            entry_price -= avg_entry

    if not pnl_list:
        print("No closed positions (BUY followed by SELL).")
        return {}

    pnl_df = pd.DataFrame(pnl_list)
    pnl_df['cumulative_pnl'] = pnl_df['pnl'].cumsum()

    wins = pnl_df[pnl_df['pnl'] > 0]['pnl'].tolist()
    losses = pnl_df[pnl_df['pnl'] <= 0]['pnl'].tolist()

    stats = {
        'Total Trades': len(pnl_df),
        'Win Rate (%)': round(len(wins) / len(pnl_df) * 100, 2),
        'Avg Win': round(mean(wins), 2) if wins else 0.0,
        'Avg Loss': round(mean(losses), 2) if losses else 0.0,
        'Best Trade': round(max(wins), 2) if wins else 0.0,
        'Worst Trade': round(min(losses), 2) if losses else 0.0,
        'Net Profit': round(sum(pnl_df['pnl']), 2)
    }

    print("\n=== TRADE PERFORMANCE SUMMARY ===")
    for k, v in stats.items():
        print(f"{k}: {v}")

    # ave CSV
    pd.DataFrame([stats]).to_csv(csv_path, index=False)
    print(f"[✓] Performance summary saved to: {csv_path}")

    # P&L Histogram
    plt.figure(figsize=(10, 5))
    plt.hist(wins, bins=15, alpha=0.6, label='Wins', color='green')
    plt.hist(losses, bins=15, alpha=0.6, label='Losses', color='red')
    plt.axvline(0, color='black', linestyle='--')
    plt.title("Distribution of Trade P&L")
    plt.xlabel("Profit/Loss ($)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    hist_path = os.path.join(plot_dir, "pnl_histogram.png")
    plt.savefig(hist_path)
    print(f"[✓] Histogram saved to: {hist_path}")
    plt.close()

    # Cumulative P&L Chart
    plt.figure(figsize=(10, 5))
    plt.plot(pnl_df['date'], pnl_df['cumulative_pnl'], marker='o', color='blue')
    plt.title("Cumulative Profit Over Time")
    plt.xlabel("date")
    plt.ylabel("Cumulative P&L ($)")
    plt.grid(True)
    plt.tight_layout()
    cum_path = os.path.join(plot_dir, "cumulative_profit.png")
    plt.savefig(cum_path)
    print(f"[✓] Cumulative profit chart saved to: {cum_path}")
    plt.close()

    # Cumulative Loss Only Chart
    pnl_df['loss_only'] = pnl_df['pnl'].apply(lambda x: x if x < 0 else 0)
    pnl_df['cumulative_loss'] = pnl_df['loss_only'].cumsum()

    plt.figure(figsize=(10, 5))
    plt.plot(pnl_df['date'], pnl_df['cumulative_loss'], marker='o', color='red')
    plt.title("Cumulative Loss Over Time")
    plt.xlabel("date")
    plt.ylabel("Cumulative Loss ($)")
    plt.grid(True)
    plt.tight_layout()

    loss_path = os.path.join(plot_dir, "cumulative_loss.png")
    plt.savefig(loss_path)
    print(f"[✓] Cumulative loss chart saved to: {loss_path}")
    plt.close()

    return stats

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
        return pd.DataFrame()  # Empty fallback

def describe_data(df):
    print("Statistical summary of raw BTC data:")
    print(df.describe())

def check_missing_and_outliers(df):
    print("\nMissing values:")
    print(df.isnull().sum())

    # Outlier check for 'close' using IQR
    # <https://careerfoundry.com/en/blog/data-analytics/how-to-find-outliers/>
    Q1 = df['close'].quantile(0.25)
    Q3 = df['close'].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df['close'] < Q1 - 1.5 * IQR) | (df['close'] > Q3 + 1.5 * IQR)]
    print(f"\nDetected {len(outliers)} potential outliers in closing prices.")

# Fast = batch prediction
# Slow = sequential prediction
def predict_eod_with_uncertainty(model, scaler, input_sequence, n_simulations=30, mode = 'fast'):
    if mode == 'fast':
        input_seq_reshaped = input_sequence.reshape(1, *input_sequence.shape)
        input_seq_batch = np.repeat(input_seq_reshaped, n_simulations, axis=0)  # shape: (30, lookback, features)
        
        predictions_scaled = model(input_seq_batch, training=True).numpy()
        predictions = scaler.inverse_transform(predictions_scaled)
        
        mean_pred = predictions.mean()
        std_pred = predictions.std()
        
        return mean_pred, std_pred
    elif mode == 'slow':
        preds = []
        input_seq_reshaped = input_sequence.reshape(1, *input_sequence.shape)
        for _ in range(n_simulations):
            predicted_scaled = model(input_seq_reshaped, training=True)
            predicted = scaler.inverse_transform(predicted_scaled)[0][0]
            preds.append(predicted)
        return np.mean(preds), np.std(preds)

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

def extract_high_volatility_window(df, window_size=20, top_pct=0.3):
    df = df.copy()
    df['rolling_vol'] = df['close'].rolling(window=window_size).std()
    df = df.dropna()

    threshold = df['rolling_vol'].quantile(1 - top_pct)
    high_vol_df = df[df['rolling_vol'] >= threshold]
    
    return high_vol_df

def add_technical_indicators(df):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

    # Bollinger Band Width
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_width'] = bb.bollinger_wband()

    # Volume Scaled (fallback to 0 if not present)
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
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(close=df['close'], volume=df.get('volume', pd.Series(0))).on_balance_volume()

    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()

    # ADX
    df['adx'] = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14).adx()

    # CCI
    df['cci'] = ta.trend.CCIIndicator(high=df['high'], low=df['low'], close=df['close'], window=20).cci()

    return df

def simulate_eod_trading_on_holdout(df, lookback=120, initial_cash=10000, qty=0.01, 
                                    cooldown_days=2, lookahead_days=3, model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)
    df = add_technical_indicators(df)
    df['date'] = df['timestamp'].dt.date

    # Feature engineering
    df['log_return'] = np.log(df['close'].shift(-1) / df['close'])
    df['target'] = df['log_return'] * 100

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
    all_dates = sorted(df['date'].unique())

    split_idx = int(len(all_dates) * 0.8)
    train_dates = all_dates[:split_idx]
    test_dates = all_dates[split_idx:]

    train_df = df[df['date'].isin(train_dates)]
    train_df = extract_high_volatility_window(train_df)

    test_df = df[df['date'].isin(test_dates)]

    # Volatility stats
    train_eod = train_df.groupby('date')['close'].last()
    daily_deltas = train_eod.diff().dropna()
    delta_mean = daily_deltas.abs().mean()
    delta_std = daily_deltas.abs().std()
    print(f"Estimated EOD Delta Mean: {delta_mean:.2f} | Std Dev: {delta_std:.2f}")

    # Save/load model
    model_name = f"hybrid_{train_dates[0]}_{train_dates[-1]}.keras"
    model_path = os.path.join(model_dir, model_name)
    X_train, y_train, x_scaler, y_scaler = prepare_eod_training_data_hybrid(train_df, lookback)
    input_shape = (X_train.shape[1], X_train.shape[2])

    if os.path.exists(model_path):
        print(f"[MODEL] Loading cached model from {model_path}")
        model = load_model_from_file(model_path)
    else:
        print(f"[MODEL] Training new model -> {model_path}")
        model = build_hybrid_LSTM_model(input_shape)
        model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=0)
        
        y_pred_train = model.predict(X_train, verbose=0)
        y_true_inv = y_scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
        y_pred_inv = y_scaler.inverse_transform(y_pred_train).flatten()

        mae = mean_absolute_error(y_true_inv, y_pred_inv)
        rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))

        print(f"[TRAIN METRICS] MAE: {mae:.4f} | RMSE: {rmse:.4f}")

        model.save(model_path)
        print(f"[MODEL] Saved to {model_path}")

    # Always calculate training error for consistency
    y_pred_train = model.predict(X_train, verbose=0)
    y_true_inv = y_scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    y_pred_inv = y_scaler.inverse_transform(y_pred_train).flatten()

    cash, btc_available, btc_locked_hodl = initial_cash, 0.0, 0.0
    active_trades = []
    trade_log, portfolio_values = [], []
    last_buy_date = None
    std_history = []
    rolling_window = 5
    dynamic_std_limit_min = 2000
    unlock_log = []
    initial_portfolio_value = initial_cash
    unlock_threshold = initial_portfolio_value * 1.10  # 10% increase threshold
    profit_pool = 0.0  # Tracks locked-in profits
    profit_lock_threshold = initial_cash * 1.10
    low_cash_threshold = initial_cash * 0.10
    min_cash_restore_target = initial_cash * 0.25

    for i in range(1, len(test_dates)):
        prev_day = test_dates[i - 1]
        curr_day = test_dates[i]

        prev_data = df[df['date'] == prev_day]
        curr_data = df[df['date'] == curr_day]
        hist = df[df['timestamp'] < curr_data['timestamp'].iloc[0]].tail(lookback)

        if len(hist) < lookback or len(prev_data) < 24 or len(curr_data) < 24:
            continue

        actual_prev_eod = prev_data['close'].iloc[-1]
        actual_today_eod = curr_data['close'].iloc[-1]

        input_seq = hist[feature_cols].values
        input_scaled = x_scaler.transform(input_seq).reshape(1, lookback, len(feature_cols))
        mean_pred, std_pred = predict_eod_with_uncertainty(model, y_scaler, input_seq, n_simulations=30)

        predicted_log_return = mean_pred / 100
        # predicted_price = actual_today_eod * np.exp(predicted_log_return)
        predicted_price = actual_prev_eod * np.exp(predicted_log_return)

        # Calculate the current portfolio value
        total_btc = btc_available + btc_locked_hodl
        total_value = cash + total_btc * actual_today_eod

        last_unlock_value = initial_cash
        position_multiplier = 1.0

        if total_value >= profit_lock_threshold and btc_locked_hodl > 0:
            unlocked_qty = btc_locked_hodl * 0.5
            if unlocked_qty < 1e-6:
                continue
            btc_locked_hodl -= unlocked_qty
            btc_available += unlocked_qty

            # Calculate how many 10% jumps we've done
            growth_factor = total_value / last_unlock_value
            position_multiplier = min(1.0 + (growth_factor - 1) * 2, 2.0)  # Cap at 2.0x

            print(f"*** RELEASED LOCKED BTC: +{unlocked_qty:.4f} BTC due to 10% portfolio growth.")
            print(f"*** NEW POSITION MULTIPLIER: x{position_multiplier:.2f}")

            last_unlock_value = total_value
            profit_lock_threshold = total_value * 1.10

        std_history.append(std_pred)
        # dynamic_std_limit = max(np.mean(std_history[-rolling_window:]) * 1.25, dynamic_std_limit_min) \
        #     if len(std_history) >= rolling_window else float('inf')

        if len(std_history) < rolling_window:
            dynamic_std_limit = dynamic_std_limit_min
        else:
            recent_std = np.mean(std_history[-rolling_window:])
            dynamic_std_limit = max(recent_std * 1.25, dynamic_std_limit_min)

        predicted_cum = [predicted_price]
        future_hist = hist.copy()
        future_dates = test_dates[i+1:i+1+lookahead_days]

        for f_day in future_dates:
            f_day_data = df[df['date'] == f_day]
            if len(f_day_data) < 24:
                continue
            future_hist = pd.concat([future_hist, f_day_data]).tail(lookback)
            if len(future_hist) < lookback:
                continue
            f_input = future_hist[feature_cols].values
            f_input_scaled = x_scaler.transform(f_input).reshape(1, lookback, len(feature_cols))
            f_pred, _ = predict_eod_with_uncertainty(model, y_scaler, f_input, n_simulations=30)
            f_price = actual_today_eod * np.exp(f_pred / 100)
            predicted_cum.append(f_price)

        # predicted_cum_delta = predicted_cum[-1] - actual_today_eod
        predicted_cum_delta = predicted_cum[-1] - actual_prev_eod

        # Dynamic thresholds
        buy_threshold = delta_mean * 0.005
        sell_threshold = -delta_mean * 0.5

        decision = "HOLD"
        can_trade = not last_buy_date or (curr_day - last_buy_date).days >= cooldown_days

        print(f"BUY CHECK: Δ={predicted_cum_delta:.2f} vs Threshold={buy_threshold:.2f}")
        print(f"Cash: ${cash:,.2f}, Required: ${actual_today_eod * qty:,.2f}")
        print(f"Can Trade: {can_trade}")

        unlock_days = 2  # Min holding period
        min_profit_pct = 0.02  # At least 2% profit
        release_fraction = 0.5  # Release 50% of locked HODL when conditions are met

        for trade in active_trades:

            if trade['sold']:
                continue

            days_held = (curr_day - trade['buy_date']).days
            profit_target_price = trade['buy_price'] * (1 + min_profit_pct)

            # This causes the final value of the portfolio to go down from 12689.75 to 11283.21
            # if days_held >= unlock_days and actual_today_eod >= profit_target_price:
            #     unlock_qty = trade['buy_qty'] * release_fraction

            #     if btc_locked_hodl >= unlock_qty:
            #         btc_locked_hodl -= unlock_qty
            #         btc_available += unlock_qty

            #         trade['sold'] = True  # OR flag as 'partially_unlocked'

            #         print(f"\nUNLOCKED HODL! -> +{unlock_qty:.4f} BTC back to tradable pool @ ${actual_today_eod:.2f}")

            #         unlock_log.append({
            #             'date': str(curr_day),
            #             'action': 'UNLOCK',
            #             'unlocked_qty': unlock_qty,
            #             'price': actual_today_eod,
            #             'held_days': days_held
            #         })

            #     if actual_today_eod > trade['buy_price']:
            #         sell_qty = trade['buy_qty'] * 0.5
            #         if btc_available >= sell_qty:
            #             btc_available -= sell_qty
            #             cash += actual_today_eod * sell_qty
            #             trade['sold'] = True
            #             decision = "PARTIAL SELL"


        # Emergency Liquidity Unlock Logic
        # without this available cash stops at 444.10 and no trades can be made, but portfolio value is 12839.80
        if cash < low_cash_threshold and btc_available > 0:
            btc_to_sell = min(btc_available, (min_cash_restore_target - cash) / actual_today_eod)
            if btc_to_sell > 1e-6:
                btc_available -= btc_to_sell
                cash += btc_to_sell * actual_today_eod
                decision = "LIQUIDATE BTC"
                print(f"\n!!! LOW CASH: Sold {btc_to_sell:.4f} BTC to restore liquidity. Cash: ${cash:,.2f}")

        if cash < low_cash_threshold and btc_available < 0.01 and btc_locked_hodl > 0.01:
            emergency_unlock = btc_locked_hodl * 0.25
            btc_locked_hodl -= emergency_unlock
            btc_available += emergency_unlock
            print(f"\n!!! Emergency unlock: {emergency_unlock:.4f} BTC released from HODL to liquidity pool.")


        if std_pred <= dynamic_std_limit and can_trade:
            # if predicted_cum_delta >= buy_threshold and cash >= actual_today_eod * qty:
            #     btc_available += qty * 0.5
            #     btc_locked_hodl += qty * 0.5
            #     cash -= actual_today_eod * qty
            #     decision = "BUY"
            #     last_buy_date = curr_day
            #     active_trades.append({
            #         'buy_date': curr_day,
            #         'buy_price': actual_today_eod,
            #         'buy_qty': qty,
            #         'sold': False
            #     })

            # Dynamic Position Sizing
            base_qty = 0.01
            min_qty = 0.001

            confidence_score = 1 / (1 + std_pred)
            # adjusted_qty = np.clip(base_qty * confidence_score, min_qty, base_qty)
            adjusted_qty = np.clip(base_qty * confidence_score * position_multiplier, min_qty, base_qty * position_multiplier)

            required_cash = actual_today_eod * adjusted_qty

            if predicted_cum_delta >= buy_threshold and cash >= required_cash:
                btc_available += adjusted_qty * 0.5
                btc_locked_hodl += adjusted_qty * 0.5
                cash -= required_cash

                decision = "BUY"
                last_buy_date = curr_day
                active_trades.append({
                    'buy_date': curr_day,
                    'buy_price': actual_today_eod,
                    'buy_qty': adjusted_qty,
                    'sold': False
                })

            elif predicted_cum_delta <= sell_threshold and btc_available >= qty:
                # btc_available -= qty
                # cash += actual_today_eod * qty
                btc_available -= adjusted_qty
                cash += actual_today_eod * adjusted_qty
                decision = "SELL"

        total_btc = btc_available + btc_locked_hodl
        total_value = cash + total_btc * actual_today_eod
        trade_log.append({
            'date': str(curr_day),
            'actual_eod': actual_today_eod,
            'predicted_eod': predicted_price,
            'pred_std': std_pred,
            'prev_eod': actual_prev_eod,
            'decision': decision,
            'cash': cash,
            'btc_available': btc_available,
            'btc_locked': btc_locked_hodl,
            'btc_total': total_btc,
            'portfolio_value': total_value
        })

        portfolio_values.append(total_value)

        print(f"\nDate: {curr_day}")
        print(f"  Prediction: {predicted_price:.2f} ± {std_pred:.2f}")
        print(f"  Previous EOD: {actual_prev_eod:.2f}")
        print(f"  Cumulative Predicted Δ (next {lookahead_days} days): {predicted_cum_delta:.2f}")
        print(f"  Decision: {decision}")
        print(f"  Cash: ${cash:,.2f} | BTC Available: {btc_available:.4f} | HODL BTC: {btc_locked_hodl:.4f} | Portfolio Value: ${total_value:,.2f}")
        if decision == "HOLD" and std_pred > dynamic_std_limit:
            print(f"  Skipping trade: std {std_pred:.2f} > limit {dynamic_std_limit:.2f}")

    print("\nSimulation Complete")
    print(f"Final Portfolio Value: ${portfolio_values[-1]:,.2f}")
    print(f"Total Trades: {sum(1 for t in trade_log if t['decision'] != 'HOLD')}")

    pd.DataFrame(trade_log).to_csv('trade_log.csv', index=False)
    print("Trade log saved to trade_log.csv")

    mae = mean_absolute_error(y_true_inv, y_pred_inv)

    # Plot actual vs predicted EOD prices
    plt.figure(figsize=(12, 6))
    plt.plot(pd.to_datetime([t['date'] for t in trade_log]), [t['actual_eod'] for t in trade_log], label="Actual EOD", color='red', marker='o')
    plt.plot(pd.to_datetime([t['date'] for t in trade_log]), [t['predicted_eod'] for t in trade_log], label="Predicted EOD", color='blue', marker='o')
    plt.fill_between(
        pd.to_datetime([t['date'] for t in trade_log]),
        [t['predicted_eod'] - t['pred_std'] for t in trade_log],
        [t['predicted_eod'] + t['pred_std'] for t in trade_log],
        color='blue', alpha=0.2,
        label="95% Confidence Interval"
    )
    plt.title("BTC EOD Prediction with Hybrid LSTM")
    plt.gcf().text(0.5, 0.90, f"Final Portfolio Value: ${portfolio_values[-1]:,.2f}\nMAE: {mae:.2f}", fontsize=10, ha='center', va='top',
                   bbox=dict(facecolor='white', edgecolor='black', alpha=0.8))
    plt.xlabel("Date")
    plt.ylabel("BTC Price (USD)")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("eod_prediction.png")
    plt.show(block=False)
    plt.pause(3)
    plt.close('all')

    # Plot portfolio value over time
    plot_dates = pd.to_datetime([t['date'] for t in trade_log])
    portfolio_vals = [t['portfolio_value'] for t in trade_log]  # Always in sync

    plt.figure(figsize=(12, 6))
    plt.plot(plot_dates, portfolio_vals, label="Portfolio Value", color='blue')
    plt.title("Portfolio Value Over Time")
    plt.gcf().text(0.5, 0.90, f"Final Portfolio Value: ${portfolio_values[-1]:,.2f}", fontsize=10, ha='center', va='top',
                   bbox=dict(facecolor='white', edgecolor='black', alpha=0.8))
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (USD)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("portfolio_value_over_time.png")
    plt.show(block=False)
    plt.pause(3)
    plt.close('all')

    # Analyze trade effectiveness after simulation
    results = analyze_trade_wins(trade_log)

    return trade_log

def main():

    set_random_seed(42)

    # Load historical BTC data
    df = crypto_bars(
        'BTC/USD',
        "2021-01-01",
        '2025-04-04',
        # '2025-04-12',
        None,
        TimeFrame.Hour
    )

    print(df.head(5))

    df = df.reset_index()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['close'] = df[[col for col in df.columns if 'close' in col.lower()]].squeeze()

    describe_data(df)
    check_missing_and_outliers(df)
    simulate_eod_trading_on_holdout(df)

if __name__ == "__main__":
    main()