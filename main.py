import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import run_test
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
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

# Copied tutorial code from <https://www.simplilearn.com/tutorials/machine-learning-tutorial/stock-price-prediction-using-machine-learning>
def train_model(data):
    dataset_train = pd.DataFrame(data)
    # print("\ntraining model\n", dataset_train)
    
    # Use the open stock price column to train the model
    training_set = dataset_train.iloc[:, 0:1].values
    print("\ntraining_set\n", training_set)
    print("\ntraining_set.shape\n", training_set.shape)

    # Normalize the data
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    print("\ntraining_set_scaled\n", training_set_scaled)
    print("\ntraining_set_scaled.shape\n", training_set_scaled.shape)

    # Creating X_train and y_train
    X_train = []
    y_train = []
    for i in range(60, len(training_set)):
        X_train.append(training_set_scaled[i-60:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    print("\nX_train\n", X_train)
    print("\ny_train\n", y_train)
    print("\nX_train.shape\n", X_train.shape)
    print("\ny_train.shape\n", y_train.shape)

    # Reshape the data
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    print("\nX_train.shape\n", X_train.shape)

    # Build the LSTM model
    regressor = Sequential()
    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units=1))

    # Compile and fit the model
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    regressor.fit(X_train, y_train, epochs=100, batch_size=32)
    print("\nModel trained.")

    actual_stock_price = dataset_train.iloc[:, 0:1].values

    # Preparing the input for the model
    dataset_total = pd.concat((dataset_train['open'], dataset_train['open']), axis=0)
    inputs = dataset_total[len(dataset_total) - len(dataset_train) - 60:].values
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)
    X_test = []
    for i in range(60, len(inputs)):
        X_test.append(inputs[i-60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Predicting the stock price
    predicted_stock_price = regressor.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    # Plotting the actual vs predicted stock price
    plt.plot(actual_stock_price, color='red', label='Actual Stock Price')
    plt.plot(predicted_stock_price, color='blue', label='Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
    print("\nModel trained and plotted.")

def main():

    run_test.run_test()
    
if __name__ == "__main__":
    main()

