from datetime import datetime, timedelta, timezone
import random
import time
import hashlib
import os
import numpy as np
import logging
import pandas as pd
import matplotlib.pyplot as plt
import run_test
import optuna
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
    # Optional: Force single-thread mode for more determinism
    # tf.config.threading.set_intra_op_parallelism_threads(1)
    # tf.config.threading.set_inter_op_parallelism_threads(1)

def account_details():
    # Initialize the TradingClient.
    trading_client = TradingClient(
        api_key,
        secret_key,
        paper=True
    )

    # Grab the account
    account = trading_client.get_account()
    return account

# Get a list of all of our positions
def portfolio_details():
    portfolio = api.list_positions()
    return portfolio
    
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

def crypto_quotes(symbol, start_date, end_date, limit, timeframe):
    # Initialize the CryptoHistoricalDataClient.
    crypto_data_client = CryptoHistoricalDataClient(
        api_key,
        secret_key
    )

    # Define request
    request = CryptoQuoteRequest(
        symbol_or_symbols=symbol,
        start=start_date,
        end=end_date,
        limit=limit,
        timeframe=timeframe
    )

    # Get the data.
    data = crypto_data_client.get_crypto_quotes(request_params=request)

    # Return the data as a dataframe.
    return data.df  

# Makes a market crypto order
def make_crypto_order(symbol, qty, time_in_force):
    trading_client = TradingClient(
        api_key,
        secret_key,
        paper=True
    )

    # Define a new order request.
    order_request = OrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        type=OrderType.MARKET,
        order_class=OrderClass.SIMPLE,
        time_in_force=time_in_force
    )

    order_submission_response = trading_client.submit_order(order_data=order_request)
    return order_submission_response

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

def build_model_LSTM(input_shape, units=50, dropout=0.2):
    regressor = Sequential()
    regressor.add(Input(shape=input_shape))
    regressor.add(LSTM(units, return_sequences=True))
    regressor.add(Dropout(dropout))
    regressor.add(LSTM(units, return_sequences=True))
    regressor.add(Dropout(dropout))
    regressor.add(LSTM(units, return_sequences=True))
    regressor.add(Dropout(dropout))
    regressor.add(LSTM(units))
    regressor.add(Dropout(dropout))
    regressor.add(Dense(units=1))
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    return regressor

def prepare_LSTM_training_data(df, scaler=None):
    if 'close' not in df.columns:
        raise ValueError("'close' column is required in the dataframe.")

    data = df['close'].values.reshape(-1, 1)

    if scaler is None:
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)
    else:
        data_scaled = scaler.transform(data)

    X, y = [], []
    for i in range(60, len(data_scaled)):
        X.append(data_scaled[i-60:i, 0])
        y.append(data_scaled[i, 0])

    X = np.array(X).reshape(-1, 60, 1)
    y = np.array(y)

    return X, y, scaler

# Modified code from <https://www.simplilearn.com/tutorials/machine-learning-tutorial/stock-price-prediction-using-machine-learning>
# def train_model_LSTM(X, y, input_shape=(60, 1), validation_split=0.1):
def train_model_LSTM(X, y, input_shape=(120, 1), validation_split=0.1):
    model = build_model_LSTM(input_shape)

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X, y,
        epochs=100,
        batch_size=32,
        validation_split=validation_split,
        verbose=1,
        callbacks=[early_stop]
    )

    return model, history

def save_model(model, path='btc_lstm_model.keras'):
    model.save(path)

def load_model_from_file(path='btc_lstm_model.keras'):
    
    return load_model(path)

# Function to test and visualize the LSTM model's loss and predicted vs actual prices and
def run_LSTM_test():
    # Get historical BTC/USD bars
    df = crypto_bars('BTC/USD', "2024-01-01", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), None, TimeFrame.Hour)

    if df.empty:
        print("No data fetched. Exiting.")
        return

    df = df.reset_index()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['close'] = df[[col for col in df.columns if 'close' in col.lower()]].squeeze()

    describe_data(df)
    check_missing_and_outliers(df)

    # Chronological 80/20 split
    total_len = len(df)
    train_len = int(total_len * 0.8)

    # Split the data
    #train_df = df.iloc[:train_len] # need to exclude 60-step back for training sequences
    train_df = df.iloc[:train_len - 60]  # exclude 60-step back for training sequences
    test_df = df.iloc[train_len - 60:]  # include 60-step back for test sequences

    # Prepare training/test sets
    X_train, y_train, scaler = prepare_LSTM_training_data(train_df)
    X_test, y_test, _ = prepare_LSTM_training_data(test_df, scaler=scaler)

    # Train the model
    model, history = train_model_LSTM(X_train, y_train)

    # Save model
    #save_model(model)

    # Plot training and validation loss
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(3)  # show it briefly
    plt.savefig("training_loss.png")
    plt.close('all')

    # Predict on test set
    predicted_scaled = model.predict(X_test)
    predicted = scaler.inverse_transform(predicted_scaled)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    timestamps = test_df['timestamp'].iloc[60:]  # align with X_test

    # Eval metrics
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    metrics_text = f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}'

    # Plot actual vs predicted prices
    plt.figure(figsize=(14, 5))
    plt.grid(True)
    plt.plot(timestamps, actual, color='red', label='Actual BTC Price')
    plt.plot(timestamps, predicted, color='blue', label='Predicted BTC Price')
    plt.title('LSTM Prediction on Test Set (20%)')
    plt.xlabel('Time')
    plt.ylabel('BTC Price (USD)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.gcf().text(0.5, 0.90, metrics_text, fontsize=10, ha='center', va='top',
                   bbox=dict(facecolor='white', edgecolor='black', alpha=0.8))
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(3)  # show it briefly
    plt.savefig("LSTM_Prediction_on_Test_Set.png")
    plt.close('all')

    return df, model, scaler

def predict_next_LSTM_price(df, model, scaler, steps_ahead=24, symbol='BTC/USD', qty=0.001, buy_threshold=500, sell_threshold=-500):
    recent_close = df['close'].values[-60:].reshape(-1, 1)
    recent_scaled = scaler.transform(recent_close)

    predictions = []
    input_seq = recent_scaled.copy()

    for _ in range(steps_ahead):
        input_seq_reshaped = input_seq.reshape(1, 60, 1)
        predicted_scaled = model.predict(input_seq_reshaped)
        predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]
        predictions.append(predicted_price)
        input_seq = np.append(input_seq[1:], predicted_scaled).reshape(60, 1)

    last_timestamp = df['timestamp'].iloc[-1]
    future_times = [last_timestamp + timedelta(hours=i + 1) for i in range(steps_ahead)]

    # Plot forecast
    plt.figure(figsize=(12, 5))
    plt.plot(future_times, predictions, marker='o', label="Forecasted BTC Price", color='blue')
    plt.title("24-Hour BTC Price Forecast (LSTM)")
    plt.xlabel("Time")
    plt.ylabel("Predicted BTC Price (USD)")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(3)  # show it briefly
    plt.savefig("forecasted_btc_price.png")
    plt.close('all')

    # Analyze changes and make trades
    for i in range(1, len(predictions)):
        delta = predictions[i] - predictions[i - 1]
        if delta >= buy_threshold:
            print(f"BUY SIGNAL: +${delta:.2f} @ Hour {i}")
            execute_trade("BUY", symbol, qty)
        elif delta <= sell_threshold:
            print(f"SELL SIGNAL: -${abs(delta):.2f} @ Hour {i}")
            execute_trade("SELL", symbol, qty)
        else:
            print(f"No significant change: ${delta:.2f} @ Hour {i}")

    return predictions

def objective(trial):
    units = trial.suggest_categorical("units", [32, 50, 64, 100])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    epochs = trial.suggest_int("epochs", 30, 100)

    # Re-fetch and split dataset inside for stateless tuning
    df = crypto_bars('BTC/USD', "2024-01-01", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), None, TimeFrame.Hour)
    if df.empty:
        raise ValueError("Data fetch failed")

    df = df.reset_index()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['close'] = df[[col for col in df.columns if 'close' in col.lower()]].squeeze()

    train_len = int(len(df) * 0.8)
    train_df = df.iloc[:train_len]

    X_train, y_train, scaler = prepare_LSTM_training_data(train_df)

    model = build_model_LSTM((60, 1), units=units, dropout=dropout)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[early_stop]
    )

    return min(history.history['val_loss'])  # best val loss

def optimize_lstm_hyperparameters():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    print("Best hyperparameters:", study.best_params)
    return study.best_params

def execute_trade(signal: str, symbol: str, qty: float, time_in_force="gtc"):
    print(f"Executing {signal} order for {qty} {symbol}")
    if signal == "BUY":
        return make_crypto_order(symbol, qty, time_in_force)
    elif signal == "SELL":
        return make_crypto_order(symbol, -qty, time_in_force)
    else:
        print("ERROR: Invalid trade signal")

def save_best_params(best_params, path='best_lstm_params.json'):
    with open(path, 'w') as f:
        json.dump(best_params, f)
    print(f"Saved best parameters to {path}")

def load_best_params(path='best_lstm_params.json'):
    if not os.path.exists(path):
        raise FileNotFoundError(f"ERROR: {path} not found. Run hyperparameter optimization first.")
    with open(path, 'r') as f:
        best_params = json.load(f)
    print(f"Loaded best parameters from {path}")
    return best_params

def safe_simulate_trading(
    df,
    window_size=1000,
    steps_ahead=6,
    initial_cash=10000,
    btc_per_trade=0.001,
    plot_equity=True,
    use_percent_change=True,
    threshold_multiplier=0.05,
    plot_path="safe_equity_curve.png",
    best_params=None,
    retrain_interval=12  # retrain every N steps
):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    tf.get_logger().setLevel("ERROR")

    print(f"window_size: {window_size}, steps_ahead: {steps_ahead}, threshold_multiplier: {threshold_multiplier}")
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['close'] = df['close'].astype(float)

    df['abs_diff'] = df['close'].diff().abs()
    avg_move = df['abs_diff'].mean()
    std_move = df['abs_diff'].std()
    print(f"[VOLATILITY] Avg hourly move: ${avg_move:.2f}, Std: ${std_move:.2f}")

    equity_curve = []
    trade_log = []
    cash = initial_cash
    btc = 0
    portfolio_value = []
    scaler = MinMaxScaler()
    start_index = window_size
    end_index = len(df) - steps_ahead

    loop_durations = []
    total_loops = (end_index - start_index) // steps_ahead
    model = None
    recent_model_time = None

    for t_idx, t in enumerate(range(start_index, end_index, steps_ahead), 1):
        loop_start_time = time.time()

        # Retrain model at intervals
        if recent_model_time is None or (t - recent_model_time) >= retrain_interval:
            log.info(f"Retraining model at index {t} ({df['timestamp'].iloc[t]})")

            window_df = df.iloc[t - window_size:t]
            future_df = df.iloc[t:t + steps_ahead]

            try:
                X_train, y_train, scaler = prepare_LSTM_training_data(window_df, scaler=None)
            except Exception as e:
                print(f"Training data prep error at index {t}: {e}")
                continue
            
            # Check if model exists and delete it to free up memory
            if model:
                del model
                K.clear_session()
                gc.collect()

            units = min(best_params.get("units", 50), 32) if best_params else 32
            dropout = best_params.get("dropout", 0.2) if best_params else 0.2
            epochs = min(best_params.get("epochs", 10), 5) if best_params else 5
            batch_size = best_params.get("batch_size", 32) if best_params else 32

            model = build_model_LSTM((60, 1), units=units, dropout=dropout)
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

            recent_model_time = t

        # Prediction batch
        recent_close = df['close'].values[t - 60:t].reshape(-1, 1)
        recent_scaled = scaler.transform(recent_close)
        input_sequences = []
        temp_input = recent_scaled.copy()

        for _ in range(steps_ahead):
            input_sequences.append(temp_input.reshape(1, 60, 1))
            temp_input = np.append(temp_input[1:], temp_input[-1]).reshape(60, 1)

        batch_input = np.concatenate(input_sequences, axis=0)
        predicted_scaled = model.predict(batch_input, verbose=0)
        predictions = scaler.inverse_transform(predicted_scaled).flatten().tolist()
        future_df = df.iloc[t:t + steps_ahead]

        for i in range(1, len(predictions)):
            timestamp = future_df['timestamp'].iloc[i]
            price = df['close'].iloc[t + i]
            prev_pred = predictions[i - 1]
            curr_pred = predictions[i]

            delta = ((curr_pred - prev_pred) / prev_pred) if use_percent_change else (curr_pred - prev_pred)
            threshold = (0.003 if use_percent_change else (avg_move + std_move * threshold_multiplier))

            if delta >= threshold and cash >= price * btc_per_trade:
                print(f"BUY SIGNAL: +{delta:.4f} @ {timestamp}")
                btc += btc_per_trade
                cash -= price * btc_per_trade
                trade_log.append({
                    'timestamp': timestamp,
                    'action': 'BUY',
                    'price': price,
                    'btc': btc_per_trade,
                    'cash': cash
                })
            elif delta <= -threshold and btc >= btc_per_trade:
                print(f"SELL SIGNAL: {delta:.4f} @ {timestamp}")
                btc -= btc_per_trade
                cash += price * btc_per_trade
                trade_log.append({
                    'timestamp': timestamp,
                    'action': 'SELL',
                    'price': price,
                    'btc': btc_per_trade,
                    'cash': cash
                })

            total_value = cash + btc * price
            equity_curve.append((timestamp, total_value))
            portfolio_value.append(total_value)

        # Loop timing & ETA
        # loop_time = time.time() - loop_start_time
        # loop_durations.append(loop_time)

        # avg_loop = np.mean(loop_durations)
        # loops_remaining = total_loops - t_idx
        # eta_seconds = int(avg_loop * loops_remaining)
        # eta_str = str(timedelta(seconds=eta_seconds))

        # log.info(f"Loop {t_idx}/{total_loops} | Time: {loop_time:.2f}s | ETA remaining: {eta_str}")
        log.info(f"Loop {t_idx}/{total_loops}")

    # Final output
    print("\nSimulation Complete:")
    print(f"Final Portfolio Value: ${portfolio_value[-1]:.2f}")
    print(f"Cash: ${cash:.2f}, BTC: {btc:.6f} (~${btc * df['close'].iloc[-1]:.2f})")
    print(f"Trades Executed: {len(trade_log)}")

    # if trade_log:
    #     daily_trades = Counter(pd.to_datetime([t['timestamp'] for t in trade_log]).dt.date)
    #     print("\nTrades per day:")
    #     for day, count in sorted(daily_trades.items()):
    #         print(f"{day}: {count}")

    if plot_equity and equity_curve:
        timestamps, values = zip(*equity_curve)
        plt.figure(figsize=(12, 5))
        plt.plot(timestamps, values, label='Equity Curve', color='green')
        plt.title("Simulated Trading Equity Curve")
        plt.xlabel("Time")
        plt.ylabel("Portfolio Value (USD)")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.legend()
        plt.savefig(plot_path)
        plt.close('all')

    return {
        'final_value': portfolio_value[-1] if portfolio_value else initial_cash,
        'trade_log': trade_log,
        'equity_curve': equity_curve
    }

# Calculate average hourly price movement
def analyze_volatility(df):
    df = df.copy()
    df['returns'] = df['close'].pct_change() * 100
    df['abs_diff'] = df['close'].diff().abs()
    
    mean_change = df['abs_diff'].mean()
    std_change = df['abs_diff'].std()
    print(f"Average hourly move: ${mean_change:.2f}, Std Dev: ${std_change:.2f}")
    
    return mean_change, std_change

def run_backtest1(best_params=None):
        # Fetch historical data
    df = crypto_bars(
        symbol='BTC/USD',
        start_date="2024-01-01",
        end_date=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        limit=None,
        timeframe=TimeFrame.Hour
    )

    if df.empty:
        print("Data fetch failed.")
        return

    # Prep DataFrame
    df = df.reset_index()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['close'] = df[[col for col in df.columns if 'close' in col.lower()]].squeeze()

    # Run simulation - ALL - (rolling window retrain + trading logic)
    result = safe_simulate_trading(
    df=df,
    # 1000 hours took a long time to run, trying 500
    window_size=500,     
    # 6 is 4x predictions/day
    # stepping down to 3 to see if it helps with performance
    steps_ahead=3,
    retrain_interval=12,            
    initial_cash=10000,
    btc_per_trade=0.001,
    plot_equity=True,
    use_percent_change=True,        # Predict % price move
    threshold_multiplier=0.05,      # High sensitivity for more trades
    best_params=best_params    
)

    # Display Summary
    print("\nTrading simulation completed.")
    print(f"Final portfolio value: ${result['final_value']:.2f}")
    print(f"Total trades executed: {len(result['trade_log'])}")
    print(f"Equity curve points: {len(result['equity_curve'])}")

    # Optional: Save trade log to CSV
    pd.DataFrame(result['trade_log']).to_csv("trade_log.csv", index=False)
    print("Trade log saved to trade_log.csv")

def load_trade_log(path='trade_log.csv'):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return []
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df.to_dict(orient='records')

def analyze_trade_wins(trade_log):
    if not trade_log:
        print("No trades to analyze.")
        return

    from statistics import mean, median

    trades = pd.DataFrame(trade_log)
    trades['timestamp'] = pd.to_datetime(trades['timestamp'])
    trades = trades.sort_values('timestamp')

    positions = []
    holding = 0
    entry_price = 0
    wins = []
    losses = []

    for _, row in trades.iterrows():
        if row['action'] == 'BUY':
            holding += row['btc']
            entry_price += row['price'] * row['btc']
        elif row['action'] == 'SELL' and holding > 0:
            avg_entry = entry_price / holding
            pnl = (row['price'] - avg_entry) * row['btc']
            positions.append({
                'timestamp': row['timestamp'],
                'entry_price': avg_entry,
                'exit_price': row['price'],
                'btc': row['btc'],
                'pnl': pnl
            })
            if pnl >= 0:
                wins.append(pnl)
            else:
                losses.append(pnl)
            # Reduce holding + entry_price for multi-partial trades
            holding -= row['btc']
            entry_price -= avg_entry * row['btc']

    if not positions:
        print("No closed positions (BUY followed by SELL).")
        return

    # Summary Stats
    total = len(positions)
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = win_count / total * 100
    avg_win = mean(wins) if wins else 0
    avg_loss = mean(losses) if losses else 0
    best = max(wins) if wins else 0
    worst = min(losses) if losses else 0
    net_profit = sum(wins) + sum(losses)

    print(f"\nTRADE WIN ANALYSIS:")
    print(f"Closed Trades: {total}")
    print(f"Wins: {win_count} | Losses: {loss_count}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Average Win: ${avg_win:.2f}")
    print(f"Average Loss: ${avg_loss:.2f}")
    print(f"Best Trade: +${best:.2f}")
    print(f"Worst Trade: ${worst:.2f}")
    print(f"Net Profit (Closed Trades): ${net_profit:.2f}")

# Predict multiple steps ahead without using real data
def predict_multi_step(model, scaler, initial_sequence, steps_ahead):
    predictions = []
    input_seq = initial_sequence.copy()  # Start with the initial sequence

    for _ in range(steps_ahead):
        # Reshape input sequence for the model
        input_seq_reshaped = input_seq.reshape(1, 60, 1)

        # Predict the next step
        predicted_scaled = model.predict(input_seq_reshaped, verbose=0)
        predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]

        # Append the prediction to the results
        predictions.append(predicted_price)

        # Update the input sequence: Remove the oldest value and add the new prediction
        input_seq = np.append(input_seq[1:], predicted_scaled).reshape(60, 1)

    print("Multi-step predictions completed.")

    return predictions

def run_multi_step_test1(start_date="2024-01-01", steps_ahead = 24):
    # Get historical BTC/USD bars
    df = crypto_bars('BTC/USD', start_date, datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), None, TimeFrame.Hour)

    if df.empty:
        print("No data fetched. Exiting.")
        return

    df = df.reset_index()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['close'] = df[[col for col in df.columns if 'close' in col.lower()]].squeeze()

    describe_data(df)
    check_missing_and_outliers(df)

    # Chronological 80/20 split
    total_len = len(df)
    train_len = int(total_len * 0.8)

    # Split the data
    train_df = df.iloc[:train_len - 60]  # Exclude the last 60 rows from training
    test_df = df.iloc[train_len - 60:]  # Include 60 rows before the test set for lookback

    # Prepare training/test sets
    X_train, y_train, scaler = prepare_LSTM_training_data(train_df)
    X_test, y_test, _ = prepare_LSTM_training_data(test_df, scaler=scaler)

    # Train the model
    model, history = train_model_LSTM(X_train, y_train)

    # Save model
    # save_model(model)

    # Plot training and validation loss
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(3)  # show it briefly
    plt.savefig("training_loss.png")
    plt.close('all')

    # Predict multiple steps ahead
    print(f"\nPredicting {steps_ahead} steps ahead...\n")
    initial_sequence = X_test[-1]  # Use the last sequence from the test set
    predictions = predict_multi_step(model, scaler, initial_sequence, steps_ahead)

    # Align predictions with timestamps
    last_timestamp = test_df['timestamp'].iloc[-1]
    future_times = [last_timestamp + timedelta(hours=i + 1) for i in range(steps_ahead)]

    # Get actual prices for comparison
    actual_prices = test_df['close'].iloc[-steps_ahead:].values  # Last `steps_ahead` actual prices

    # Calculate MAE and RMSE
    mae = mean_absolute_error(actual_prices, predictions)
    rmse = np.sqrt(mean_squared_error(actual_prices, predictions))
    metrics_text = f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}"

    # Plot predictions vs actual prices
    plt.figure(figsize=(12, 5))
    plt.plot(future_times, actual_prices, marker='o', label="Actual BTC Price", color='red')
    plt.plot(future_times, predictions, marker='o', label="Predicted BTC Price", color='blue')
    plt.title(f"{steps_ahead}-Hour BTC Price Forecast (LSTM)")
    plt.xlabel("Time")
    plt.ylabel("BTC Price (USD)")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()

    # Add MAE and RMSE to the plot
    plt.gcf().text(
        0.5, 0.90, metrics_text, fontsize=10, ha='center', va='top',
        bbox=dict(facecolor='white', edgecolor='black', alpha=0.8)
    )

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(3)  # show it briefly
    plt.savefig("multi_step_forecast.png")
    plt.close('all')

    return df, model, scaler

# With MC Dropout Forecasting
def run_multi_step_test(start_date="2024-01-01", steps_ahead=24, model_path='btc_lstm_model.keras', load_if_exists=True):
    from pathlib import Path

    # Get historical BTC/USD bars
    df = crypto_bars('BTC/USD', start_date, datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), None, TimeFrame.Hour)

    if df.empty:
        print("No data fetched. Exiting.")
        return

    df = df.reset_index()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['close'] = df[[col for col in df.columns if 'close' in col.lower()]].squeeze()

    describe_data(df)
    check_missing_and_outliers(df)

    total_len = len(df)
    train_len = int(total_len * 0.8)
    train_df = df.iloc[:train_len - 60]
    test_df = df.iloc[train_len - 60:]

    X_train, y_train, scaler = prepare_LSTM_training_data(train_df)
    X_test, y_test, _ = prepare_LSTM_training_data(test_df, scaler=scaler)

    # Load model from file if available
    if load_if_exists and Path(model_path).exists():
        print(f"[Model] Loading model from {model_path}")
        model = load_model_from_file(model_path)
        history = None
    else:
        # Train and save a new model
        print("[Model] Training new model...")
        model, history = train_model_LSTM(X_train, y_train)
        save_model(model, model_path)
        print(f"[Model] Saved to {model_path}")

    # (Optional) Plot loss
    # if history:
    #     plt.figure(figsize=(8, 4))
    #     plt.plot(history.history['loss'], label='Training Loss')
    #     if 'val_loss' in history.history:
    #         plt.plot(history.history['val_loss'], label='Validation Loss')
    #     plt.title('Loss Over Epochs')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss (MSE)')
    #     plt.grid(True)
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show(block=False)
    #     plt.pause(3)
    #     plt.savefig("training_loss.png")
    #     plt.close('all')

    # Forecast with uncertainty
    initial_sequence = X_test[-1]
    mean_preds, std_preds = predict_multi_step_with_uncertainty(
        model, scaler, initial_sequence, steps_ahead=steps_ahead, n_simulations=30
    )

    last_timestamp = test_df['timestamp'].iloc[-1]
    future_times = [last_timestamp + timedelta(hours=i + 1) for i in range(steps_ahead)]
    actual_prices = test_df['close'].iloc[-steps_ahead:].values

    mae = mean_absolute_error(actual_prices, mean_preds)
    rmse = np.sqrt(mean_squared_error(actual_prices, mean_preds))
    metrics_text = f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}"

    # Plot predictions
    plt.figure(figsize=(12, 5))
    plt.plot(future_times, actual_prices, marker='o', label="Actual BTC Price", color='red')
    plt.plot(future_times, mean_preds, marker='o', label="Predicted Mean", color='blue')
    plt.fill_between(
        future_times,
        mean_preds - 2 * std_preds,
        mean_preds + 2 * std_preds,
        color='blue', alpha=0.2,
        label="95% Confidence Interval"
    )
    plt.title(f"{steps_ahead}-Hour BTC Price Forecast (LSTM w/ Uncertainty)")
    plt.xlabel("Time")
    plt.ylabel("BTC Price (USD)")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.gcf().text(
        0.5, 0.90, metrics_text, fontsize=10, ha='center', va='top',
        bbox=dict(facecolor='white', edgecolor='black', alpha=0.8)
    )
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(3)
    plt.savefig("multi_step_forecast_with_uncertainty.png")
    plt.close('all')

    return df, model, scaler

def predict_multi_step_with_uncertainty(model, scaler, initial_sequence, steps_ahead, n_simulations=30):
    predictions = []

    for sim in range(n_simulations):
        input_seq = initial_sequence.copy()
        sim_preds = []
        for _ in range(steps_ahead):
            input_seq_reshaped = input_seq.reshape(1, 60, 1)
            predicted_scaled = model(input_seq_reshaped, training=True)  # MC Dropout ON
            predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]
            sim_preds.append(predicted_price)
            input_seq = np.append(input_seq[1:], predicted_scaled).reshape(60, 1)
        predictions.append(sim_preds)

    predictions = np.array(predictions)
    mean_preds = predictions.mean(axis=0)
    std_preds = predictions.std(axis=0)

    return mean_preds, std_preds

def run_rolling_forecasts(start_date = '2024-01-01', last_n_days=3, steps_ahead=24, model_dir='models', log_path='prediction_logs.csv'):
    os.makedirs(model_dir, exist_ok=True)

    # last_n_days = last_n_days - 1

    # Load full dataset
    df = crypto_bars('BTC/USD', start_date, datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), None, TimeFrame.Hour)
    df = df.reset_index()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['close'] = df[[col for col in df.columns if 'close' in col.lower()]].squeeze()
    df['date'] = df['timestamp'].dt.date

    available_dates = sorted(df['date'].unique())
    forecast_dates = []

    for day in available_dates[-last_n_days - 1:]:
        day_index = df[df['date'] == day].index.min()

        if day_index is None:
            continue

        if day_index >= 60 and (day_index + steps_ahead) <= len(df):
            forecast_dates.append(day)
        else:
            print(f"Skipping {day} - Not enough data for training/prediction.")

    logs = []

    for day in forecast_dates:
        print(f"Forecasting day: {day}")
        day_index = df[df['date'] == day].index.min()
        train_df = df.iloc[:day_index]
        future_df = df.iloc[day_index:day_index + steps_ahead]

        # Calculate training range
        train_start_dt = train_df['timestamp'].iloc[0]
        train_end_dt = train_df['timestamp'].iloc[-1]
        train_days = (train_end_dt - train_start_dt).days + 1

        train_start_str = train_start_dt.strftime('%Y%m%d')
        train_end_str = train_end_dt.strftime('%Y%m%d')
        trained_on_str = day.strftime('%Y%m%d')

        # Including date range trained on and the date the model is trained allows us to update stale models when needed
        model_filename = f"lstm_{train_start_str}-{train_end_str}_TRAINED_{trained_on_str}.keras"
        model_path = os.path.join(model_dir, model_filename)

        if os.path.exists(model_path):
            print(f"Loading existing model from {model_path}")
            model = load_model_from_file(model_path)
            model_trained = True
            # Just need the scaler for prediction
            _, _, scaler = prepare_LSTM_training_data(train_df)
        else:
            print(f"Training new model: {model_filename}")
            X_train, y_train, scaler = prepare_LSTM_training_data(train_df)
            model, _ = train_model_LSTM(X_train, y_train)
            model.save(model_path)
            model_trained = False
            print(f"Model saved to {model_path}")

        # Prepare sequence for prediction
        last_seq = scaler.transform(train_df['close'].values[-60:].reshape(-1, 1))
        mean_preds, std_preds = predict_multi_step_with_uncertainty(
            model, scaler, last_seq, steps_ahead=steps_ahead, n_simulations=30
        )

        actual_prices = future_df['close'].values
        mae = mean_absolute_error(actual_prices, mean_preds)
        rmse = np.sqrt(mean_squared_error(actual_prices, mean_preds))

        # Update filename with metrics if training just occurred
        # if not os.path.exists(model_path.replace('.keras', f'_MAE{int(mae)}_RMSE{int(rmse)}.keras')):
        #     print(f"Renaming model file with metrics: MAE={mae:.2f}, RMSE={rmse:.2f}")
        #     train_start_str = train_start_dt.strftime('%Y%m%d')
        #     train_end_str = train_end_dt.strftime('%Y%m%d')
        #     trained_on_str = day.strftime('%Y%m%d')

        #     updated_model_path = os.path.join(
        #         model_dir,
        #         f"lstm_MAE{int(mae)}_RMSE{int(rmse)}_{train_start_str}-{train_end_str}_TRAINED_{trained_on_str}.keras"
        #     )
        #     os.rename(model_path, updated_model_path)
        #     model_path = updated_model_path

        # Plot forecast
        future_times = future_df['timestamp'].values
        train_range_str = f"Trained on: {train_start_dt.strftime('%Y-%m-%d')} → {train_end_dt.strftime('%Y-%m-%d')} (Duration: {train_days} days)"

        plt.figure(figsize=(12, 5))
        plt.plot(future_times, actual_prices, marker='o', label="Actual BTC Price", color='red')
        plt.plot(future_times, mean_preds, marker='o', label="Predicted Mean", color='blue')
        plt.fill_between(
            future_times,
            mean_preds - 2 * std_preds,
            mean_preds + 2 * std_preds,
            color='blue', alpha=0.2,
            label="95% Confidence Interval"
        )

        plt.title(f"24-Hour BTC Forecast for {day} (MAE={mae:.2f}, RMSE={rmse:.2f})")
        plt.xlabel("Time")
        plt.ylabel("BTC Price (USD)")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()

        plt.gcf().text(
            0.5, 0.92, train_range_str, fontsize=9, ha='center', va='top',
            bbox=dict(facecolor='white', edgecolor='black', alpha=0.7)
        )

        plt.tight_layout()
        plt.savefig(f"forecast_{day}.png")
        plt.close()

        logs.append({
            'date': str(day),
            'MAE': mae,
            'RMSE': rmse,
            'model_path': model_path,
            'train_start': train_start_dt.strftime('%Y-%m-%d'),
            'train_end': train_end_dt.strftime('%Y-%m-%d'),
            'train_days': train_days,
            'trained_on': day.strftime('%Y-%m-%d')
        })

    # Save logs
    log_df = pd.DataFrame(logs)
    log_df.to_csv(log_path, index=False)
    print(f"Logs saved to: {log_path}")
    print(log_df)

def prepare_eod_training_data_hybrid(df, x_scaler=None, y_scaler=None, lookback=120):
    df = add_technical_indicators(df)
    df['date'] = df['timestamp'].dt.date

    feature_cols = [
        'close', 'rsi', 'bb_width', 'volume_scaled',
        'macd_line', 'macd_signal', 'macd_hist',
        'obv', 'stoch_k', 'stoch_d', 'adx', 'cci'
    ]


    df = df.dropna(subset=feature_cols)

    X, y = [], []
    all_dates = sorted(df['date'].unique())

    for i in range(3, len(all_dates)):
        day = all_dates[i]
        day_data = df[df['date'] == day]
        if len(day_data) < 24:
            continue

        end_of_day_price = day_data['close'].iloc[-1]
        hist = df[df['timestamp'] < day_data['timestamp'].iloc[0]].tail(lookback)

        if len(hist) < lookback:
            continue

        seq = hist[feature_cols].values  # shape: (lookback, features)
        X.append(seq)
        y.append(end_of_day_price)

    X = np.array(X)  # (samples, lookback, features)
    y = np.array(y).reshape(-1, 1)

    if x_scaler is None:
        x_scaler = MinMaxScaler()
        X_shape = X.shape
        X = x_scaler.fit_transform(X.reshape(-1, X_shape[2])).reshape(X_shape)
    else:
        X_shape = X.shape
        X = x_scaler.transform(X.reshape(-1, X_shape[2])).reshape(X_shape)

    if y_scaler is None:
        y_scaler = MinMaxScaler()
        y = y_scaler.fit_transform(y)
    else:
        y = y_scaler.transform(y)

    return X, y, x_scaler, y_scaler

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

def train_hybrid_model(df, lookback=120):
    X, y, x_scaler, y_scaler = prepare_eod_training_data_hybrid(df, lookback=lookback)
    input_shape = (X.shape[1], X.shape[2])
    model = build_hybrid_LSTM_model(input_shape)

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X, y, epochs=100, batch_size=32, validation_split=0.1, callbacks=[early_stop], verbose=1)

    return model, x_scaler, y_scaler, history

# def predict_eod_with_uncertainty(model, scaler, input_sequence, n_simulations=30):
#     preds = []
#     for _ in range(n_simulations):
#         input_seq_reshaped = input_sequence.reshape(1, 60, 1)
#         predicted_scaled = model(input_seq_reshaped, training=True)
#         predicted = scaler.inverse_transform(predicted_scaled)[0][0]
#         preds.append(predicted)
#     return np.mean(preds), np.std(preds)

def predict_eod_with_uncertainty(model, scaler, input_sequence, n_simulations=30):
    preds = []
    input_seq_reshaped = input_sequence.reshape(1, *input_sequence.shape)  # 🛠️ Generalize
    for _ in range(n_simulations):
        predicted_scaled = model(input_seq_reshaped, training=True)
        predicted = scaler.inverse_transform(predicted_scaled)[0][0]
        preds.append(predicted)
    return np.mean(preds), np.std(preds)

def run_eod_forecasts(start_date='2024-01-01', last_n_days=5, model_dir='models', log_path='eod_prediction_logs.csv'):
    os.makedirs(model_dir, exist_ok=True)

    df = crypto_bars('BTC/USD', start_date, datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), None, TimeFrame.Hour)
    df = df.reset_index()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['close'] = df[[col for col in df.columns if 'close' in col.lower()]].squeeze()
    df['date'] = df['timestamp'].dt.date

    available_dates = sorted(df['date'].unique())
    forecast_dates = []

    for day in available_dates[-last_n_days:]:
        day_rows = df[df['date'] == day]
        if len(day_rows) < 24:
            continue
        prev_60 = df[df['timestamp'] < day_rows['timestamp'].iloc[0]].tail(60)
        if len(prev_60) < 60:
            continue
        forecast_dates.append(day)

    logs = []
    all_dates = []
    all_preds = []
    all_actuals = []
    all_stds = []

    for day in forecast_dates:
        print(f"Forecasting EOD for {day}")
        day_rows = df[df['date'] == day]
        train_df = df[df['timestamp'] < day_rows['timestamp'].iloc[0]]
        future_df = day_rows

        train_start_dt = train_df['timestamp'].iloc[0]
        train_end_dt = train_df['timestamp'].iloc[-1]
        train_days = (train_end_dt - train_start_dt).days + 1

        train_start_str = train_start_dt.strftime('%Y%m%d')
        train_end_str = train_end_dt.strftime('%Y%m%d')
        trained_on_str = day.strftime('%Y%m%d')

        model_filename = f"lstm_eod_{train_start_str}-{train_end_str}_TRAINED_{trained_on_str}.keras"
        model_path = os.path.join(model_dir, model_filename)

        if os.path.exists(model_path):
            print(f"Loading existing model from {model_path}")
            model = load_model_from_file(model_path)
            _, _, scaler = prepare_eod_training_data(train_df)
        else:
            print(f"Training new model for {day}")
            X_train, y_train, scaler = prepare_eod_training_data(train_df)
            model, _ = train_model_LSTM(X_train, y_train)
            model.save(model_path)
            print(f"Model saved to {model_path}")

        last_seq = scaler.transform(train_df['close'].values[-60:].reshape(-1, 1))
        mean_pred, std_pred = predict_eod_with_uncertainty(model, scaler, last_seq, n_simulations=30)
        actual_price = future_df['close'].iloc[-1]

        mae = mean_absolute_error([actual_price], [mean_pred])
        rmse = np.sqrt(mean_squared_error([actual_price], [mean_pred]))

        logs.append({
            'date': str(day),
            'MAE': mae,
            'RMSE': rmse,
            'model_path': model_path,
            'train_start': train_start_dt.strftime('%Y-%m-%d'),
            'train_end': train_end_dt.strftime('%Y-%m-%d'),
            'train_days': train_days,
            'trained_on': day.strftime('%Y-%m-%d')
        })

        # Store for combined plotting
        all_dates.append(day)
        all_preds.append(mean_pred)
        all_actuals.append(actual_price)
        all_stds.append(std_pred)

    # Save log
    log_df = pd.DataFrame(logs)
    log_df.to_csv(log_path, index=False)
    print(f"Logs saved to: {log_path}")
    print(log_df)

    # Plot combined predictions vs actual
    if all_dates:
        all_dates = pd.to_datetime(all_dates)

        plt.figure(figsize=(12, 6))
        plt.plot(all_dates, all_actuals, label="Actual EOD", color="red", marker='o')
        plt.errorbar(all_dates, all_preds, yerr=np.array(all_stds) * 2, fmt='o', color='blue', capsize=5, label="Predicted EOD ±95% CI")
        plt.title(f"BTC EOD Forecasts (Last {len(all_dates)} Days)")
        plt.xlabel("Date")
        plt.ylabel("BTC Price (USD)")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("eod_forecast_summary.png")
        plt.show(block=False)
        plt.pause(3)
        plt.close('all')

def test_lookback_windows_for_eod(
    df, 
    lookback_windows=[60, 72, 96, 120], 
    epochs=50, 
    batch_size=32,
    verbose=1
):
    results = []

    for lb in lookback_windows:
        print(f"\nTesting lookback window: {lb} hours")
        try:
            X, y, scaler = prepare_eod_training_data(df, lookback=lb)
        except Exception as e:
            print(f"[!] Failed at lookback={lb}: {e}")
            continue

        if len(X) < 10:
            print(f"[!] Skipping lookback={lb} — not enough samples ({len(X)})")
            continue

        # Train the model
        model, history = train_model_LSTM(X, y, input_shape=(lb, 1), validation_split=0.1)

        # Predict on training set just to test fit (optionally split real test set)
        y_pred_scaled = model.predict(X)
        y_pred = scaler.inverse_transform(y_pred_scaled)
        y_true = scaler.inverse_transform(y)

        # Evaluate
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        print(f"✅ Lookback={lb} → MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        results.append({
            "lookback": lb,
            "MAE": mae,
            "RMSE": rmse
        })

    # Show summary
    print("\nLookback Performance Summary:")
    results_df = pd.DataFrame(results).sort_values("RMSE")
    print(results_df.to_string(index=False))

    return results_df

def test_extended_lookbacks(df):
    # Define new lookback set
    extended_lookbacks = [144, 168]
    return test_lookback_windows_for_eod(df, lookback_windows=extended_lookbacks)

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

def evaluate_hybrid_model_on_holdout(df, lookback=120):
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    df = add_technical_indicators(df)
    df['date'] = df['timestamp'].dt.date

    feature_cols = [
        'close', 'rsi', 'bb_width', 'volume_scaled',
        'macd_line', 'macd_signal', 'macd_hist',
        'obv', 'stoch_k', 'stoch_d', 'adx', 'cci'
    ]
    
    df = df.dropna(subset=feature_cols)

    all_dates = sorted(df['date'].unique())
    split_idx = int(len(all_dates) * 0.8)
    train_dates = all_dates[:split_idx]
    test_dates = all_dates[split_idx:]

    train_df = df[df['date'].isin(train_dates)]
    test_df = df[df['date'].isin(test_dates)]

    X_train, y_train, x_scaler, y_scaler = prepare_eod_training_data_hybrid(train_df, lookback=lookback)
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_hybrid_LSTM_model(input_shape)

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.1,
        callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
        verbose=1
    )

    all_preds = []
    all_actuals = []
    all_dates_eval = []

    for day in test_dates:
        day_data = test_df[test_df['date'] == day]
        if len(day_data) < 24:
            continue

        hist = df[df['timestamp'] < day_data['timestamp'].iloc[0]].tail(lookback)
        if len(hist) < lookback:
            continue

        input_seq = hist[feature_cols].values
        input_scaled = x_scaler.transform(input_seq).reshape(1, lookback, len(feature_cols))

        predicted_scaled = model.predict(input_scaled, verbose=0)
        predicted = y_scaler.inverse_transform(predicted_scaled)[0][0]
        actual = day_data['close'].iloc[-1]

        all_preds.append(predicted)
        all_actuals.append(actual)
        all_dates_eval.append(day)

    mae = mean_absolute_error(all_actuals, all_preds)
    rmse = np.sqrt(mean_squared_error(all_actuals, all_preds))
    metrics_text = f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}'

    print(f"\nEvaluation on Holdout Set ({len(all_preds)} days):")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    all_dates_eval = pd.to_datetime(all_dates_eval)

    plt.figure(figsize=(12, 6))
    plt.plot(all_dates_eval, all_actuals, label="Actual EOD", color="red", marker='o')
    plt.plot(all_dates_eval, all_preds, label="Predicted EOD", color="blue", marker='o')
    plt.title(f"Hybrid LSTM EOD Prediction on 20% Holdout ({len(all_preds)} Days)")
    plt.xlabel("Date")
    plt.ylabel("BTC Price (USD)")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.gcf().text(0.5, 0.90, metrics_text, fontsize=10, ha='center', va='top',
                   bbox=dict(facecolor='white', edgecolor='black', alpha=0.8))
    plt.savefig("hybrid_eod_holdout_prediction.png")
    plt.show(block=False)
    plt.pause(3)
    plt.close('all')

    return mae, rmse

def simulate_eod_trading_on_holdout(df, lookback=120, initial_cash=10000, qty=0.01, 
                                    cooldown_days=2, lookahead_days=3, model_dir="models",
                                    model_config={"units": 64, "dropout": 0.2},
                                    retrain_daily=False, force_retrain=False):

    os.makedirs(model_dir, exist_ok=True)

    df = add_technical_indicators(df)
    df['date'] = df['timestamp'].dt.date

    feature_cols = [
        'close', 'rsi', 'bb_width', 'volume_scaled',
        'macd_line', 'macd_signal', 'macd_hist',
        'obv', 'stoch_k', 'stoch_d', 'adx', 'cci'
    ]

    df = df.dropna(subset=feature_cols)
    all_dates = sorted(df['date'].unique())

    split_idx = int(len(all_dates) * 0.8)
    test_dates = all_dates[split_idx:]

    train_eod = df[df['date'].isin(all_dates[:split_idx])].groupby('date')['close'].last()
    daily_deltas = train_eod.diff().dropna()
    delta_mean = daily_deltas.abs().mean()
    delta_std = daily_deltas.abs().std()
    dynamic_threshold = delta_mean + delta_std

    print(f"Estimated EOD Delta Mean: {delta_mean:.2f} | Std Dev: {delta_std:.2f}")
    print(f"Using dynamic trade threshold: {dynamic_threshold:.2f}")

    cash = initial_cash
    btc = 0
    trade_log = []
    portfolio_values = []
    std_history = []
    rolling_window = 5
    dynamic_std_limit_min = 2000

    buy_dates = []
    sell_dates = []
    buy_prices = []
    sell_prices = []

    for i in range(1, len(test_dates)):
        prev_day = test_dates[i - 1]
        curr_day = test_dates[i]
        curr_data = df[df['date'] == curr_day]
        if len(curr_data) < 24:
            continue

        actual_prev_eod = df[df['date'] == prev_day]['close'].iloc[-1]
        actual_today_eod = curr_data['close'].iloc[-1]

        updated_train_df = df[df['timestamp'] < curr_data['timestamp'].iloc[0]]
        if len(updated_train_df) < lookback * 2:
            print(f"Skipping {curr_day}: Not enough training data.")
            continue

        train_start_dt = updated_train_df['timestamp'].iloc[0]
        train_end_dt = updated_train_df['timestamp'].iloc[-1]

        hash_input = f"{lookback}_{feature_cols}_{model_config}_{train_start_dt}_{train_end_dt}"
        model_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
        model_filename = f"lstm_{model_hash}.keras"
        model_path = os.path.join(model_dir, model_filename)

        retrain_condition = retrain_daily or force_retrain or not os.path.exists(model_path)

        if retrain_condition:
            print(f"[{curr_day}] Training model (reason: {'forced' if force_retrain else 'daily' if retrain_daily else 'missing'})")
            try:
                X_train, y_train, x_scaler, y_scaler = prepare_eod_training_data_hybrid(updated_train_df, lookback=lookback)
            except Exception as e:
                print(f"[{curr_day}] Training prep error: {e}")
                continue

            input_shape = (X_train.shape[1], X_train.shape[2])
            model = build_hybrid_LSTM_model(input_shape, **model_config)
            model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=0)
            model.save(model_path)
            print(f"[{curr_day}] Saved model to {model_path}")
        else:
            print(f"[{curr_day}] Loading model from cache: {model_path}")
            model = load_model_from_file(model_path)
            _, _, x_scaler, y_scaler = prepare_eod_training_data_hybrid(updated_train_df, lookback=lookback)

        hist = updated_train_df.tail(lookback)
        input_seq = hist[feature_cols].values
        input_scaled = x_scaler.transform(input_seq).reshape(1, lookback, len(feature_cols))
        mean_pred, std_pred = predict_eod_with_uncertainty(model, y_scaler, input_seq, n_simulations=30)

        std_history.append(std_pred)
        if len(std_history) >= rolling_window:
            dynamic_std_limit = max(np.mean(std_history[-rolling_window:]) * 1.25, dynamic_std_limit_min)
        else:
            dynamic_std_limit = float('inf')

        predicted_cum = [mean_pred]
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
            predicted_cum.append(f_pred)

        predicted_cum_delta = predicted_cum[-1] - actual_today_eod
        decision = "HOLD"
        qty_scaled = qty

        if std_pred <= dynamic_std_limit:
            confidence = 1 - (std_pred / dynamic_std_limit)
            qty_scaled = qty * confidence
            if predicted_cum_delta >= dynamic_threshold and cash >= actual_today_eod * qty_scaled:
                btc += qty_scaled
                cash -= actual_today_eod * qty_scaled
                decision = "BUY"
                buy_dates.append(curr_day)
                buy_prices.append(actual_today_eod)
            elif predicted_cum_delta <= -dynamic_threshold and btc >= qty_scaled:
                btc -= qty_scaled
                cash += actual_today_eod * qty_scaled
                decision = "SELL"
                sell_dates.append(curr_day)
                sell_prices.append(actual_today_eod)

        total_value = cash + btc * actual_today_eod
        trade_log.append({
            'date': str(curr_day),
            'actual_eod': actual_today_eod,
            'predicted_eod': mean_pred,
            'pred_std': std_pred,
            'prev_eod': actual_prev_eod,
            'decision': decision,
            'confidence': confidence if std_pred <= dynamic_std_limit else 0,
            'qty_scaled': qty_scaled if std_pred <= dynamic_std_limit else 0,
            'cash': cash,
            'btc': btc,
            'portfolio_value': total_value
        })

        portfolio_values.append(total_value)

    print("\nSimulation Complete")
    print(f"Final Portfolio Value: ${portfolio_values[-1]:,.2f}")
    print(f"Total Trades: {sum(1 for t in trade_log if t['decision'] != 'HOLD')}")

    pd.DataFrame(trade_log).to_csv("eod_trade_log.csv", index=False)

    all_dates_eval = pd.to_datetime([row['date'] for row in trade_log])
    all_actuals = [row['actual_eod'] for row in trade_log]
    all_preds = [row['predicted_eod'] for row in trade_log]
    all_stds = [row['pred_std'] for row in trade_log]

    plt.figure(figsize=(12, 6))
    plt.plot(all_dates_eval, all_actuals, color='red', label="Actual EOD")
    plt.plot(all_dates_eval, all_preds, color='blue', label="Predicted EOD")
    plt.fill_between(
        all_dates_eval,
        np.array(all_preds) - 2 * np.array(all_stds),
        np.array(all_preds) + 2 * np.array(all_stds),
        color='blue', alpha=0.2, label="±95% CI"
    )
    plt.scatter(pd.to_datetime(buy_dates), buy_prices, marker='^', color='green', label='Buy Signal', zorder=5)
    plt.scatter(pd.to_datetime(sell_dates), sell_prices, marker='v', color='red', label='Sell Signal', zorder=5)
    plt.legend()
    plt.title("Hybrid LSTM EOD Forecasts with Daily Retraining")
    plt.xlabel("Date")
    plt.ylabel("BTC Price (USD)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("eod_forecast_prediction.png")
    plt.show(block=False)
    plt.pause(5)
    plt.close('all')

    plt.figure(figsize=(12, 6))
    plt.plot(all_dates_eval, portfolio_values, color='green', linestyle='-', label="Portfolio Value")
    plt.title("Simulated Portfolio Value Over Time")
    plt.xlabel("Date")
    plt.ylabel("USD")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig("eod_portfolio_value.png")
    plt.show(block=False)
    plt.pause(5)
    plt.close('all')

    return trade_log


def main():
    # Load best Optuna params (optional if not using optimized)
    # try:
    #     best_params = load_best_params()
    # except FileNotFoundError:
    #     best_params = {
    #         "units": 50,
    #         "dropout": 0.2,
    #         "batch_size": 32,
    #         "epochs": 100
    #     }

    # run_backtest1(best_params=best_params)

    # trade_log = load_trade_log('trade_log.csv')
    # analyze_trade_wins(trade_log)

    # run_multi_step_test(load_if_exists=False)
    set_random_seed(42)
    # run_rolling_forecasts("2024-01-01", last_n_days=1)
    # run_eod_forecasts("2024-01-01", last_n_days=10)

    # Load historical BTC data
    df = crypto_bars('BTC/USD', "2024-01-01", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), None, TimeFrame.Hour)
    df = df.reset_index()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['close'] = df[[col for col in df.columns if 'close' in col.lower()]].squeeze()

    # Run the test loop
    # test_results = test_lookback_windows_for_eod(df)

    # Step 2: Run test for 6–7 day windows
    # results_df = test_extended_lookbacks(df)

    # print("\nTraining hybrid LSTM model with RSI, BB, Volume...")
    # model, x_scaler, y_scaler, history = train_hybrid_model(df, lookback=120)

    # # Plot training loss (optional)
    # plt.figure(figsize=(8, 4))
    # plt.plot(history.history['loss'], label='Train Loss')
    # if 'val_loss' in history.history:
    #     plt.plot(history.history['val_loss'], label='Val Loss')
    # plt.title("Training Loss (Hybrid LSTM)")
    # plt.xlabel("Epochs")
    # plt.ylabel("MSE Loss")
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("hybrid_training_loss.png")
    # plt.close()

    # # Save the trained model
    # save_model(model, path="hybrid_eod_model.keras")
    # print("Model saved as hybrid_eod_model.keras")

    # # (Optional) Predict next EOD using last sequence
    # last_seq_df = add_technical_indicators(df).dropna().tail(120)
    # input_seq = last_seq_df[['close', 'rsi', 'bb_width', 'volume_scaled']].values
    # input_seq_scaled = x_scaler.transform(input_seq).reshape(1, 120, 4)
    # predicted_eod_scaled = model.predict(input_seq_scaled)
    # predicted_eod = y_scaler.inverse_transform(predicted_eod_scaled)[0][0]


    # print(f"\nPredicted BTC EOD Price: ${predicted_eod:,.2f}")

    # evaluate_hybrid_model_on_holdout(df, lookback=120)

    # simulate_eod_trading_on_holdout(
    #     df,
    #     lookback=120,
    #     threshold=500,
    #     std_limit=6000,
    #     initial_cash=10000,
    #     qty=0.001
    # )

    simulate_eod_trading_on_holdout(df, retrain_daily=False)




if __name__ == "__main__":
    main()

