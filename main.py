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
def train_model_LSTM(X, y, input_shape=(60, 1), validation_split=0.1):
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

def run_rolling_forecasts(start_date="2024-01-01", last_n_days=3, steps_ahead=24, model_dir='models', log_path='prediction_logs.csv'):
    os.makedirs(model_dir, exist_ok=True)

    # Roll back one day because current day won't have 24 hours of data
    last_n_days = last_n_days + 1

    # Load full dataset
    df = crypto_bars('BTC/USD', start_date, datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), None, TimeFrame.Hour)
    df = df.reset_index()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['close'] = df[[col for col in df.columns if 'close' in col.lower()]].squeeze()

    # Determine last full day in data
    df['date'] = df['timestamp'].dt.date
    available_dates = sorted(df['date'].unique())
    last_day = available_dates[-1]
    forecast_dates = available_dates[-last_n_days:]

    logs = []

    for day in forecast_dates:
        print(f"\nForecasting day: {day}")
        day_index = df[df['date'] == day].index.min()

        # Ensure we have 60 back + 24 future hours of data
        if day_index < 60 or day_index + steps_ahead > len(df):
            print("Not enough data around this date. Skipping.")
            continue

        # Training set = everything before this day
        train_df = df.iloc[:day_index]

        # Forecast target range
        future_df = df.iloc[day_index:day_index + steps_ahead]

        # Prepare LSTM training
        X_train, y_train, scaler = prepare_LSTM_training_data(train_df)
        model, _ = train_model_LSTM(X_train, y_train)

        # Predict future 24 hours w/ MC dropout
        last_seq = scaler.transform(train_df['close'].values[-60:].reshape(-1, 1))
        mean_preds, std_preds = predict_multi_step_with_uncertainty(
            model, scaler, last_seq, steps_ahead=steps_ahead, n_simulations=30
        )

        actual_prices = future_df['close'].values
        mae = mean_absolute_error(actual_prices, mean_preds)
        rmse = np.sqrt(mean_squared_error(actual_prices, mean_preds))

        # Save model with version info
        model_filename = f"lstm_MAE{int(mae)}_RMSE{int(rmse)}_{day.strftime('%Y%m%d')}.keras"
        model_path = os.path.join(model_dir, model_filename)
        model.save(model_path)

        # Plot forecast
        future_times = future_df['timestamp'].values
        train_start = train_df['timestamp'].iloc[0].strftime('%Y-%m-%d')
        train_end = train_df['timestamp'].iloc[-1].strftime('%Y-%m-%d')
        train_range_str = f"Trained on: {train_start} → {train_end}"

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

        # Add training range annotation to plot
        plt.gcf().text(
            0.5, 0.92, train_range_str, fontsize=9, ha='center', va='top',
            bbox=dict(facecolor='white', edgecolor='black', alpha=0.7)
        )

        plt.tight_layout()
        plt.savefig(f"forecast_{day}.png")
        plt.close('all')
        print(f"Forecast plot saved for {day}.")


        train_start_dt = train_df['timestamp'].iloc[0]
        train_end_dt = train_df['timestamp'].iloc[-1]
        train_days = (train_end_dt - train_start_dt).days + 1

        logs.append({
            'date': str(day),
            'MAE': mae,
            'RMSE': rmse,
            'model_path': model_path,
            'train_start': train_start_dt.strftime('%Y-%m-%d'),
            'train_end': train_end_dt.strftime('%Y-%m-%d'),
            'train_days': train_days
        })

    # Save prediction log
    log_df = pd.DataFrame(logs)
    log_df.to_csv(log_path, index=False)
    print(f"\nLogs saved to: {log_path}")
    print(log_df)

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
    run_rolling_forecasts("2024-01-01", last_n_days=1, training=True)


if __name__ == "__main__":
    main()

