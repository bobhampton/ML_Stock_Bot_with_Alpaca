#fetch data from api
import alpaca_trade_api as tradeapi
from config.config import API_KEY, SECRET_KEY,BASE_URL

def get_btc_data():
    api=tradeapi.REST(API_KEY,SECRET_KEY,base_url=BASE_URL)
    bars=api.get_crypto_bars("BTC/USD",timeframe="1Hour").df
    return bars[bars['symbol']=='BTC/USD']