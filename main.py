from datetime import datetime, timedelta, timezone
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import run_test
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Input
from keras.callbacks import EarlyStopping
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

# Load environment variables from .env file
load_dotenv()
BASE_URL = "https://paper-api.alpaca.markets"
api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")
api = tradeapi.REST(key_id=api_key, secret_key=secret_key, base_url=BASE_URL, api_version='v2')

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
    # Initialize the CryptoHistoricalDataClient.
    crypto_data_client = CryptoHistoricalDataClient(
        api_key,
        secret_key
    )

    # Define a request
    request = CryptoBarsRequest(
        symbol_or_symbols=symbol,
        start=start_date,
        end=end_date,
        limit=limit,
        timeframe=timeframe
    )

    # Get the data.
    bar_data = crypto_data_client.get_crypto_bars(request_params=request)

    # Return the data as a dataframe.
    return bar_data.df

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

def build_model_LSTM(input_shape):
    regressor = Sequential()
    regressor.add(Input(shape=input_shape))  # Use Input layer to define the input shape
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units=1))
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    return regressor

# Modified code from <https://www.simplilearn.com/tutorials/machine-learning-tutorial/stock-price-prediction-using-machine-learning>
def train_model_LSTM(data, model):
    dataset_train = pd.DataFrame(data)

    if len(dataset_train) < 60:
        raise ValueError("Not enough data to train the model. At least 60 rows are required.")

    training_set = dataset_train.iloc[:, 0:1].values
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    X_train = []
    y_train = []
    for i in range(60, len(training_set)):
        X_train.append(training_set_scaled[i-60:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Define early stopping
    early_stop = EarlyStopping(
        monitor='loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    # Train the model and capture history
    history = model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=32,
        verbose=1,
        callbacks=[early_stop]
    )

    # Plot and save the training loss
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return model, sc

def predict_and_plot_LSTM(data, model, scaler):
    full_data = pd.DataFrame(data)
    real_prices = full_data.iloc[:, 0:1].values
    scaled_data = scaler.transform(real_prices)

    # Prepare inputs for prediction
    X_test = []
    for i in range(60, len(scaled_data)):
        X_test.append(scaled_data[i-60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Predict
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # True prices to compare
    true_prices = real_prices[60:]

    # Evaluation metrics
    mae = mean_absolute_error(true_prices, predicted_prices)
    mse = mean_squared_error(true_prices, predicted_prices)
    rmse = np.sqrt(mse)

    # Plot
    #plt.figure(figsize=(14, 5))
    plt.plot(true_prices, color='red', label='Actual BTC Price')
    plt.plot(predicted_prices, color='blue', label='Predicted BTC Price')
    plt.title('BTC Price Prediction')
    plt.xlabel('Time (hours)')
    plt.ylabel('BTC Price')
    plt.legend()

    # Add metrics as text on the plot
    metrics_text = f'MAE: {mae:.2f}\nMSE: {mse:.2f}\nRMSE: {rmse:.2f}'
    plt.text(0.01, 0.95, metrics_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))

    plt.show()

def run_LSTM():
    btc_data = crypto_bars('BTC/USD', '2024-01-01', '2025-01-31', 1000, TimeFrame.Hour)

    # Raw data inspection
    describe_data(btc_data)
    check_missing_and_outliers(btc_data)

    # Train and evaluate LSTM model
    model = build_model_LSTM((60, 1))
    trained_model, scaler = train_model_LSTM(btc_data, model)
    predict_and_plot_LSTM(btc_data, trained_model, scaler)

def main():

    # run_test.run_test()

    run_LSTM()
    
if __name__ == "__main__":
    main()

