from core.utils import set_random_seed
from config.env_loader import init_env
from core.data_handler import crypto_bars, describe_data, check_missing_and_outliers
from core.strategy import simulate_eod_trading_on_holdout
from alpaca.data.timeframe import TimeFrame
import pandas as pd

def main():
    set_random_seed(42)
    init_env()

    df = crypto_bars(
        symbol='BTC/USD',
        start_date="2021-01-01",
        end_date="2025-04-04",
        limit=None,
        timeframe=TimeFrame.Hour
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
