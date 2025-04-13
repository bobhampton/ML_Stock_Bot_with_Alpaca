import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error

from core.data_handler import add_technical_indicators, extract_high_volatility_window
from core.training import prepare_eod_training_data_hybrid, build_hybrid_LSTM_model, load_model_from_file
from core.prediction import predict_eod_with_uncertainty
from core.analysis import analyze_trade_wins

def simulate_eod_trading_on_holdout(df, lookback=120, initial_cash=10000, qty=0.01, 
                                    cooldown_days=2, lookahead_days=3, model_dir="models"):

    os.makedirs(model_dir, exist_ok=True)
    df = add_technical_indicators(df)
    df['date'] = df['timestamp'].dt.date

    # Feature engineering
    df['log_return'] = np.log(df['close'].shift(-1) / df['close'])
    df['target'] = df['log_return'] * 100

    df['ma_20'] = df['close'].rolling(20).mean()
    df['ma_ratio'] = df['close'] / df['ma_20']
    df['volatility_20'] = df['close'].rolling(20).std()
    df['daily_range'] = (df['high'] - df['low']) / df['close']
    df['body_size'] = abs(df['close'] - df['open']) / df['close']

    feature_cols = [
        'close', 'rsi', 'bb_width', 'volume_scaled',
        'macd_line', 'macd_signal', 'macd_hist',
        'obv', 'stoch_k', 'stoch_d', 'adx', 'cci',
        'ma_ratio', 'volatility_20', 'daily_range', 'body_size'
    ]

    df = df.dropna(subset=feature_cols + ['target'])
    all_dates = sorted(df['date'].unique())

    split_idx = int(len(all_dates) * 0.8)
    train_dates = all_dates[:split_idx]
    test_dates = all_dates[split_idx:]

    train_df = df[df['date'].isin(train_dates)]
    train_df = extract_high_volatility_window(train_df)

    test_df = df[df['date'].isin(test_dates)]

    train_eod = train_df.groupby('date')['close'].last()
    daily_deltas = train_eod.diff().dropna()
    delta_mean = daily_deltas.abs().mean()
    delta_std = daily_deltas.abs().std()
    print(f"Estimated EOD Delta Mean: {delta_mean:.2f} | Std Dev: {delta_std:.2f}")

    model_name = f"hybrid_{train_dates[0]}_{train_dates[-1]}.keras"
    model_path = os.path.join(model_dir, model_name)

    X_train, y_train, x_scaler, y_scaler = prepare_eod_training_data_hybrid(train_df, lookback)
    input_shape = (X_train.shape[1], X_train.shape[2])

    if os.path.exists(model_path):
        print(f"[MODEL] Loading cached model from {model_path}")
        model = load_model_from_file(model_path)
    else:
        print(f"[MODEL] Training new model -> {model_path}")
        model = build_hybrid_LSTM_model(input_shape)
        model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=0)

        y_pred_train = model.predict(X_train, verbose=0)
        y_true_inv = y_scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
        y_pred_inv = y_scaler.inverse_transform(y_pred_train).flatten()

        mae = mean_absolute_error(y_true_inv, y_pred_inv)
        rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
        print(f"[TRAIN METRICS] MAE: {mae:.4f} | RMSE: {rmse:.4f}")

        model.save(model_path)
        print(f"[MODEL] Saved to {model_path}")

    y_pred_train = model.predict(X_train, verbose=0)
    y_true_inv = y_scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    y_pred_inv = y_scaler.inverse_transform(y_pred_train).flatten()

    cash, btc_available, btc_locked_hodl = initial_cash, 0.0, 0.0
    active_trades = []
    trade_log, portfolio_values = [], []
    last_buy_date = None
    std_history = []
    rolling_window = 5
    dynamic_std_limit_min = 2000
    unlock_log = []
    initial_portfolio_value = initial_cash
    profit_lock_threshold = initial_cash * 1.10
    low_cash_threshold = initial_cash * 0.10
    min_cash_restore_target = initial_cash * 0.25

    for i in range(1, len(test_dates)):
        prev_day = test_dates[i - 1]
        curr_day = test_dates[i]

        prev_data = df[df['date'] == prev_day]
        curr_data = df[df['date'] == curr_day]
        hist = df[df['timestamp'] < curr_data['timestamp'].iloc[0]].tail(lookback)

        if len(hist) < lookback or len(prev_data) < 24 or len(curr_data) < 24:
            continue

        actual_prev_eod = prev_data['close'].iloc[-1]
        actual_today_eod = curr_data['close'].iloc[-1]

        input_seq = hist[feature_cols].values
        input_scaled = x_scaler.transform(input_seq).reshape(1, lookback, len(feature_cols))
        mean_pred, std_pred = predict_eod_with_uncertainty(model, y_scaler, input_seq, n_simulations=30)

        predicted_log_return = mean_pred / 100
        predicted_price = actual_prev_eod * np.exp(predicted_log_return)

        total_btc = btc_available + btc_locked_hodl
        total_value = cash + total_btc * actual_today_eod

        last_unlock_value = initial_cash
        position_multiplier = 1.0

        if total_value >= profit_lock_threshold and btc_locked_hodl > 0:
            unlocked_qty = btc_locked_hodl * 0.5
            if unlocked_qty < 1e-6:
                continue
            btc_locked_hodl -= unlocked_qty
            btc_available += unlocked_qty
            growth_factor = total_value / last_unlock_value
            position_multiplier = min(1.0 + (growth_factor - 1) * 2, 2.0)
            print(f"*** RELEASED LOCKED BTC: +{unlocked_qty:.4f} BTC due to 10% portfolio growth.")
            print(f"*** NEW POSITION MULTIPLIER: x{position_multiplier:.2f}")
            last_unlock_value = total_value
            profit_lock_threshold = total_value * 1.10

        std_history.append(std_pred)
        if len(std_history) < rolling_window:
            dynamic_std_limit = dynamic_std_limit_min
        else:
            recent_std = np.mean(std_history[-rolling_window:])
            dynamic_std_limit = max(recent_std * 1.25, dynamic_std_limit_min)

        predicted_cum = [predicted_price]
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
            f_price = actual_today_eod * np.exp(f_pred / 100)
            predicted_cum.append(f_price)

        predicted_cum_delta = predicted_cum[-1] - actual_prev_eod
        buy_threshold = delta_mean * 0.005
        sell_threshold = -delta_mean * 0.5
        decision = "HOLD"
        can_trade = not last_buy_date or (curr_day - last_buy_date).days >= cooldown_days

        print(f"BUY CHECK: Δ={predicted_cum_delta:.2f} vs Threshold={buy_threshold:.2f}")
        print(f"Cash: ${cash:,.2f}, Required: ${actual_today_eod * qty:,.2f}")
        print(f"Can Trade: {can_trade}")

        unlock_days = 2
        min_profit_pct = 0.02
        release_fraction = 0.5

        # === Enhanced Exit Logic v2.0 ===

        for trade in active_trades:
            if trade['sold']:
                continue

            days_held = (curr_day - trade['buy_date']).days
            current_profit_pct = (actual_today_eod - trade['buy_price']) / trade['buy_price']

            # Track max price since entry for trailing stop
            if 'max_price' not in trade or actual_today_eod > trade['max_price']:
                trade['max_price'] = actual_today_eod

            trailing_stop_pct = 0.015
            profit_target_pct = 0.03
            max_hold_days = 7

            exit_reason = None

            # Profit Target Hit
            if current_profit_pct >= profit_target_pct:
                exit_reason = "Target Profit Reached"

            # Trailing Stop Triggered
            elif actual_today_eod < trade['max_price'] * (1 - trailing_stop_pct):
                exit_reason = "Trailing Stop Triggered"

            # Confidence Collapse (model uncertainty too high)
            elif std_pred > dynamic_std_limit * 1.25:
                exit_reason = "Uncertainty Spike"

            # Time-based Exit
            elif days_held > max_hold_days and current_profit_pct > 0:
                exit_reason = "Max Hold Reached"

            if exit_reason:
                sell_qty = trade['buy_qty']
                if btc_available >= sell_qty:
                    btc_available -= sell_qty
                    cash += actual_today_eod * sell_qty
                    trade['sold'] = True
                    decision = f"SELL ({exit_reason})"

                    print(f"Exit: {exit_reason} | Sold {sell_qty:.4f} BTC @ ${actual_today_eod:,.2f}")


        if cash < low_cash_threshold and btc_available > 0:
            btc_to_sell = min(btc_available, (min_cash_restore_target - cash) / actual_today_eod)
            if btc_to_sell > 1e-6:
                btc_available -= btc_to_sell
                cash += btc_to_sell * actual_today_eod
                decision = "LIQUIDATE BTC"
                print(f"\n!!! LOW CASH: Sold {btc_to_sell:.4f} BTC to restore liquidity. Cash: ${cash:,.2f}")

        if cash < low_cash_threshold and btc_available < 0.01 and btc_locked_hodl > 0.01:
            emergency_unlock = btc_locked_hodl * 0.25
            btc_locked_hodl -= emergency_unlock
            btc_available += emergency_unlock
            print(f"\n!!! Emergency unlock: {emergency_unlock:.4f} BTC released from HODL to liquidity pool.")

        if std_pred <= dynamic_std_limit and can_trade:
            base_qty = 0.01
            min_qty = 0.001
            confidence_score = 1 / (1 + std_pred)
            adjusted_qty = np.clip(base_qty * confidence_score * position_multiplier, min_qty, base_qty * position_multiplier)
            required_cash = actual_today_eod * adjusted_qty

            if predicted_cum_delta >= buy_threshold and cash >= required_cash:
                btc_available += adjusted_qty * 0.5
                btc_locked_hodl += adjusted_qty * 0.5
                cash -= required_cash
                decision = "BUY"
                last_buy_date = curr_day
                active_trades.append({
                    'buy_date': curr_day,
                    'buy_price': actual_today_eod,
                    'buy_qty': adjusted_qty,
                    'sold': False
                })

            elif predicted_cum_delta <= sell_threshold and btc_available >= qty:
                btc_available -= adjusted_qty
                cash += actual_today_eod * adjusted_qty
                decision = "SELL"

        total_btc = btc_available + btc_locked_hodl
        total_value = cash + total_btc * actual_today_eod

        trade_log.append({
            'date': str(curr_day),
            'actual_eod': actual_today_eod,
            'predicted_eod': predicted_price,
            'pred_std': std_pred,
            'prev_eod': actual_prev_eod,
            'decision': decision,
            'cash': cash,
            'btc_available': btc_available,
            'btc_locked': btc_locked_hodl,
            'btc_total': total_btc,
            'portfolio_value': total_value
        })

        portfolio_values.append(total_value)

        print(f"\nDate: {curr_day}")
        print(f"  Prediction: {predicted_price:.2f} ± {std_pred:.2f}")
        print(f"  Previous EOD: {actual_prev_eod:.2f}")
        print(f"  Cumulative Predicted Δ (next {lookahead_days} days): {predicted_cum_delta:.2f}")
        print(f"  Decision: {decision}")
        print(f"  Cash: ${cash:,.2f} | BTC Available: {btc_available:.4f} | HODL BTC: {btc_locked_hodl:.4f} | Portfolio Value: ${total_value:,.2f}")
        if decision == "HOLD" and std_pred > dynamic_std_limit:
            print(f"  Skipping trade: std {std_pred:.2f} > limit {dynamic_std_limit:.2f}")

    print("\nSimulation Complete")
    print(f"Final Portfolio Value: ${portfolio_values[-1]:,.2f}")
    print(f"Total Trades: {sum(1 for t in trade_log if t['decision'] != 'HOLD')}")

    os.makedirs("trade_logs", exist_ok=True)
    pd.DataFrame(trade_log).to_csv('trade_logs/trade_log.csv', index=False)
    print("Trade log saved to trade_log.csv")

    mae = mean_absolute_error(y_true_inv, y_pred_inv)

    # Plot actual vs predicted
    plt.figure(figsize=(12, 6))
    plt.plot(pd.to_datetime([t['date'] for t in trade_log]), [t['actual_eod'] for t in trade_log], label="Actual EOD", color='red', marker='o')
    plt.plot(pd.to_datetime([t['date'] for t in trade_log]), [t['predicted_eod'] for t in trade_log], label="Predicted EOD", color='blue', marker='o')
    plt.fill_between(
        pd.to_datetime([t['date'] for t in trade_log]),
        [t['predicted_eod'] - t['pred_std'] for t in trade_log],
        [t['predicted_eod'] + t['pred_std'] for t in trade_log],
        color='blue', alpha=0.2, label="95% Confidence Interval"
    )
    plt.title("BTC EOD Prediction with Hybrid LSTM")
    plt.gcf().text(0.5, 0.90, f"Final Portfolio Value: ${portfolio_values[-1]:,.2f}\nMAE: {mae:.2f}",
                   fontsize=10, ha='center', va='top', bbox=dict(facecolor='white', edgecolor='black', alpha=0.8))
    plt.xlabel("Date")
    plt.ylabel("BTC Price (USD)")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/eod_prediction.png")
    plt.show(block=False)
    plt.pause(3)
    plt.close('all')

    # Portfolio value over time
    plot_dates = pd.to_datetime([t['date'] for t in trade_log])
    portfolio_vals = [t['portfolio_value'] for t in trade_log]

    plt.figure(figsize=(12, 6))
    plt.plot(plot_dates, portfolio_vals, label="Portfolio Value", color='blue')
    plt.title("Portfolio Value Over Time")
    plt.gcf().text(0.5, 0.90, f"Final Portfolio Value: ${portfolio_values[-1]:,.2f}",
                   fontsize=10, ha='center', va='top', bbox=dict(facecolor='white', edgecolor='black', alpha=0.8))
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (USD)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/portfolio_value_over_time.png")
    plt.show(block=False)
    plt.pause(3)
    plt.close('all')

    # Analyze effectiveness
    analyze_trade_wins(trade_log)

    return trade_log
