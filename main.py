from datetime import datetime, timedelta, timezone
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

def stock_bars(symbol, start_date, end_date, limit, timeframe, latest=False):
    stock_data_client = StockHistoricalDataClient(
        api_key,
        secret_key
    )

    # Define request
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        start=start_date,
        end=end_date,
        limit=limit,
        timeframe=timeframe,
    )

    # Get the data.
    data = stock_data_client.get_stock_bars(request_params=request)

    if latest:
        # Get the latest bar for the stock.
        request = StockLatestBarRequest(symbol_or_symbols=[symbol])
        latest_bar = stock_data_client.get_stock_latest_bar(request_params=request)
        return latest_bar

    # Return the data as a dataframe.
    return data.df

def stock_quotes(symbol, start_date, end_date, limit, timeframe):
    # Initialize the StockHistoricalDataClient.
    stock_data_client = StockHistoricalDataClient(
        api_key,
        secret_key
    )

    # Define a request using the StockQuotesRequest class.
    request = StockQuotesRequest(
        symbol_or_symbols=symbol,
        start=start_date,
        end=end_date,
        limit=limit,
        timeframe=timeframe
    )

    # Get the data.
    data = stock_data_client.get_stock_quotes(request_params=request)

    # Return the data as a dataframe.
    return data.df

# Function to get all or individual crypto or stock assets
# Left this in, but will need to update to switch cases
def get_assets(symbol):
    trading_client = TradingClient(
        api_key,
        secret_key,
        paper=True
    )

    # search for crypto assets
    request_crypto = GetAssetsRequest(
        asset_class=AssetClass.CRYPTO,
    )

    # search for stocks assets
    request_stocks = GetAssetsRequest(
        asset_class=AssetClass.US_EQUITY,
    )

    # Get all crypto assets
    assets = trading_client.get_all_assets(request_crypto)
    #print(assets)

    # Get all stocks assets
    assets = trading_client.get_all_assets(request_stocks)
    #print(assets)   

    # Grab a specific asset
    asset = trading_client.get_asset(symbol_or_asset_id=symbol)
    print(asset)

def make_stock_order(symbol, qty, time_in_force):
    trading_client = TradingClient(
        api_key,
        secret_key,
        paper=True
    )

    # Let's define a new order request.
    order_request = OrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        type=OrderType.MARKET,
        order_class=OrderClass.SIMPLE,
        time_in_force=time_in_force,
        extended_hours=False
    )

    # Submit the order.
    order_submission_response = trading_client.submit_order(order_data=order_request)
    return order_submission_response

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

    # if account.cash >= estimated_risk:
    #     print(f"Estimated risk is within acceptable limits. Proceeding with order for {qty} of {symbol}.")
    # else:
    #     print(f"Estimated risk exceeds acceptable limits. Cannot proceed with order for {qty} of {symbol}.")
    # Submit the order.
    order_submission_response = trading_client.submit_order(order_data=order_request)
    return order_submission_response

# Can only get a max of 50 articles.
def get_news(symbol, limit=50):
    # Initialize the NewsClient.
    news_data_client = NewsClient(
        api_key,
        secret_key
    )

    # Initialize the NewsRequest.
    request = NewsRequest(
        symbols=symbol,
        limit=limit
    )

    # List to store all news data
    all_news_data = []

    # Now let's get the data.
    news_data = news_data_client.get_news(request)
    all_news_data.append(news_data)

    # If there are more articles, we can get them by using the next_page_token.
    while next_page_token := news_data.next_page_token:
        request = NewsRequest(
            symbols=symbol,
            limit=limit,
            page_token=next_page_token
        )
        news_data = news_data_client.get_news(request)
        all_news_data.append(news_data)

    return all_news_data

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
    train_df = df.iloc[:train_len]
    test_df = df.iloc[train_len - 60:]  # include 60-step back for test sequences

    # Prepare training/test sets
    X_train, y_train, scaler = prepare_LSTM_training_data(train_df)
    X_test, y_test, _ = prepare_LSTM_training_data(test_df, scaler=scaler)

    # Train the model
    model, history = train_model_LSTM(X_train, y_train)

    # Save model
    save_model(model)

    # Plot training vs validation loss
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

        if t % 240 == 0:
            log.info("Still working... index: {}".format(t))

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

    if trade_log:
        daily_trades = Counter(pd.to_datetime([t['timestamp'] for t in trade_log]).dt.date)
        print("\nTrades per day:")
        for day, count in sorted(daily_trades.items()):
            print(f"{day}: {count}")

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
    window_size=1000,     
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


def main():
    # Load best Optuna params (optional if not using optimized)
    try:
        best_params = load_best_params()
    except FileNotFoundError:
        best_params = {
            "units": 50,
            "dropout": 0.2,
            "batch_size": 32,
            "epochs": 100
        }

    run_backtest1(best_params=best_params)

    trade_log = load_trade_log('trade_log.csv')
    analyze_trade_wins(trade_log)

if __name__ == "__main__":
    main()

