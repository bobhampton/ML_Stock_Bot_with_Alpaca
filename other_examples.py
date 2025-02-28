""" 
    # Other examples of requests that may be useful?
    
    # Request to grab the latest orderbook for the stock 'BTC/USD'.
    request = CryptoLatestOrderbookRequest(
        symbol_or_symbols=['BTC/USD']
    )
    # Send the request
    book_data = crypto_data_client.get_crypto_latest_orderbook(request_params=request)
    
    # Request to grab the trades for the stock 'BTC/USD'.
    request = CryptoTradesRequest(
        symbol_or_symbols=['BTC/USD'],
        start=start_date,
        end=end_date,
        limit=1000
    )
    trade_data = crypto_data_client.get_crypto_trades(request_params=request)

    # Request to grab a snapshot for the stock 'BTC/USD'.
    # Snapshots contain latest trade, latest quote, latest minute bar, latest
    # daily bar and previous daily bar data for the queried symbols.
    request = CryptoSnapshotRequest(
        symbol_or_symbols=['BTC/USD']
    )
    snapshot_data = crypto_data_client.get_crypto_snapshot(request_params=request) 
    """