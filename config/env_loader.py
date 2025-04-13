import os
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

def init_env():
    load_dotenv()
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    base_url = "https://paper-api.alpaca.markets"

    if not api_key or not secret_key:
        raise EnvironmentError("Missing ALPACA_API_KEY or ALPACA_SECRET_KEY in environment variables.")

    api = tradeapi.REST(key_id=api_key, secret_key=secret_key, base_url=base_url, api_version='v2')
    return api_key, secret_key, base_url, api
