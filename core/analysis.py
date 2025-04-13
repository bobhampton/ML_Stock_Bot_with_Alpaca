import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean
import os

def analyze_trade_wins(trade_log, csv_path="trade_logs/trade_performance_summary.csv", plot_dir="analytics"):
    os.makedirs(plot_dir, exist_ok=True)

    if not trade_log:
        print("No trades to analyze.")
        return {}

    trades = pd.DataFrame(trade_log)
    trades['date'] = pd.to_datetime(trades['date'])
    trades = trades.sort_values('date')

    # Track open and closed trades
    holding = 0
    entry_price = 0
    pnl_list = []

    for _, row in trades.iterrows():
        if row['decision'] == 'BUY':
            holding += 1
            entry_price += row['actual_eod']
        elif row['decision'] == 'SELL' and holding > 0:
            avg_entry = entry_price / holding
            pnl = row['actual_eod'] - avg_entry
            pnl_list.append({
                'date': row['date'],
                'pnl': pnl
            })
            holding -= 1
            entry_price -= avg_entry

    if not pnl_list:
        print("No closed positions (BUY followed by SELL).")
        return {}

    pnl_df = pd.DataFrame(pnl_list)
    pnl_df['cumulative_pnl'] = pnl_df['pnl'].cumsum()

    wins = pnl_df[pnl_df['pnl'] > 0]['pnl'].tolist()
    losses = pnl_df[pnl_df['pnl'] <= 0]['pnl'].tolist()

    stats = {
        'Total Trades': len(pnl_df),
        'Win Rate (%)': round(len(wins) / len(pnl_df) * 100, 2),
        'Avg Win': round(mean(wins), 2) if wins else 0.0,
        'Avg Loss': round(mean(losses), 2) if losses else 0.0,
        'Best Trade': round(max(wins), 2) if wins else 0.0,
        'Worst Trade': round(min(losses), 2) if losses else 0.0,
        'Net Profit': round(sum(pnl_df['pnl']), 2)
    }

    print("\n=== TRADE PERFORMANCE SUMMARY ===")
    for k, v in stats.items():
        print(f"{k}: {v}")

    pd.DataFrame([stats]).to_csv(csv_path, index=False)
    print(f"[✓] Performance summary saved to: {csv_path}")

    # P&L Histogram
    plt.figure(figsize=(10, 5))
    plt.hist(wins, bins=15, alpha=0.6, label='Wins', color='green')
    plt.hist(losses, bins=15, alpha=0.6, label='Losses', color='red')
    plt.axvline(0, color='black', linestyle='--')
    plt.title("Distribution of Trade P&L")
    plt.xlabel("Profit/Loss ($)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    hist_path = os.path.join(plot_dir, "pnl_histogram.png")
    plt.savefig(hist_path)
    print(f"[✓] Histogram saved to: {hist_path}")
    plt.close()

    # Cumulative P&L Chart
    plt.figure(figsize=(10, 5))
    plt.plot(pnl_df['date'], pnl_df['cumulative_pnl'], marker='o', color='blue')
    plt.title("Cumulative Profit Over Time")
    plt.xlabel("date")
    plt.ylabel("Cumulative P&L ($)")
    plt.grid(True)
    plt.tight_layout()
    cum_path = os.path.join(plot_dir, "cumulative_profit.png")
    plt.savefig(cum_path)
    print(f"[✓] Cumulative profit chart saved to: {cum_path}")
    plt.close()

    # Cumulative Loss Only Chart
    pnl_df['loss_only'] = pnl_df['pnl'].apply(lambda x: x if x < 0 else 0)
    pnl_df['cumulative_loss'] = pnl_df['loss_only'].cumsum()

    plt.figure(figsize=(10, 5))
    plt.plot(pnl_df['date'], pnl_df['cumulative_loss'], marker='o', color='red')
    plt.title("Cumulative Loss Over Time")
    plt.xlabel("date")
    plt.ylabel("Cumulative Loss ($)")
    plt.grid(True)
    plt.tight_layout()
    loss_path = os.path.join(plot_dir, "cumulative_loss.png")
    plt.savefig(loss_path)
    print(f"[✓] Cumulative loss chart saved to: {loss_path}")
    plt.close()

    return stats
