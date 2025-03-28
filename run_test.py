import main as m
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.enums import TimeInForce

def run_test():
    # # This will be good for checking account balance and other stuff
    # account = m.account_details()
    # print("\nAccount details:\n", account)

    # # Get the quantity of shares for each position in the portfolio.
    # portfolio = m.portfolio_details()
    # print("\nPortfolio details:")
    # for position in portfolio:
    #     print("{} shares of {}".format(position.qty, position.symbol))

    # # Get bar data for specific crypto, date range, limit, and timeframe
    # crypto_bars_data = m.crypto_bars('BTC/USD', '2023-11-01', '2023-11-10', 5, TimeFrame.Hour)
    # print("\nCrypto bar data:\n")
    # print(crypto_bars_data)
    
    # # Get quote data for specific crypto, date range, limit, and timeframe
    # crypto_quotes_data = m.crypto_quotes('BTC/USD', '2023-11-01', '2023-11-10', 5, TimeFrame.Hour)
    # print("\nCrypto quote data:\n")
    # print(crypto_quotes_data)
    
    # # Get bar data for specific stock, date range, limit, and timeframe
    # # Added option to only return the latest bar, but don't know if that'll be helpful
    # stock_bars_data = m.stock_bars('AAPL', '2023-11-01', '2023-11-10', 5, TimeFrame.Hour, latest=False)
    # print("\nStock bars data:\n")
    # print(stock_bars_data)
    
    # # Get quote data for specific stock, date range, limit, and timeframe
    # stock_quotes_data = m.stock_quotes('AAPL', '2023-11-01', '2023-11-10', 5, TimeFrame.Hour)
    # print("\nStock quote data:\n")
    # print(stock_quotes_data)

    # # Get all assets or a specific asset (need up uncomment which one in the function)
    # # Current setting just prints specific asset info
    # # Don't know if we'll need this, but it's here. Maybe for checking if a stock/crypto exists?
    # m.get_assets('BTC/USD')

    # # Make a stock order. Will probably need to add more options to this.
    # # Think we should stick with crypto, because stock market closes and stuff
    # stock_order_response = m.make_stock_order('AAPL', 1, TimeInForce.GTC)
    # print("\nStock order response:\n")
    # print(stock_order_response)

    # # Make a crypto order. Will probably need to add more options to this.
    # crypto_order_response = m.make_crypto_order('BTC/USD', 0.001, TimeInForce.GTC)
    # print("\nCrypto order response:\n")
    # print(crypto_order_response)

    # # Get news for a specific stock or crypto
    # get_news_data = m.get_news('MSFT', 50) 
    # print("\nNews data:\n")
    # for news in get_news_data:
    #     print(news)

    # Train a model using the data
    # Followed tutorial code from <https://www.simplilearn.com/tutorials/machine-learning-tutorial/stock-price-prediction-using-machine-learning>
    # Will replace with my own code
    btc_data = m.crypto_bars('BTC/USD', '2025-01-01', '2025-01-31', 1000, TimeFrame.Hour)
    m.train_model(btc_data)