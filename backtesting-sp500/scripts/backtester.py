import pandas as pd
import matplotlib.pyplot as plt
import os

def backtest(df, sp500):
    df = df.copy()
    os.makedirs("results/plots", exist_ok=True)

    if df.empty or sp500.empty:
        raise ValueError("DataFrames are empty")

    df["pnl"] = df["signal"].astype(int) * df["monthly_future_return"]
    df = df.dropna(subset=["pnl"])

    strategy = df.groupby(level='Date').agg({"pnl": "sum", "signal": "sum"})
    strategy["strategy_return"] = strategy["pnl"] / strategy["signal"].replace(0, 1)

    sp500 = sp500.copy()
    # SP500 benchmark: invest $20 each month
    sp500_signal = pd.Series(20, index=sp500.index)
    sp500["pnl"] = sp500_signal * sp500["monthly_return"]
    sp500["strategy_return"] = sp500["monthly_return"]

    common = strategy.index.intersection(sp500.index)
    if len(common) == 0:
        raise ValueError("No common dates")

    strategy, sp500 = strategy.loc[common], sp500.loc[common]
    strategy["cum_pnl"] = strategy["pnl"].cumsum()
    sp500["cum_pnl"] = sp500["pnl"].cumsum()

    total_pnl = strategy["cum_pnl"].iloc[-1]

    print(f"Strategy PnL: ${total_pnl:.2f} | SP500: ${sp500['cum_pnl'].iloc[-1]:.2f}")
    if total_pnl > 75:
        raise ValueError(
            f"PnL ${total_pnl:.2f} exceeds $75 — outliers were not corrected correctly."
        )

    strategy_total_return = strategy["strategy_return"].mean()
    sp500_total_return = sp500["strategy_return"].mean()

    with open("results/results.txt", "w") as f:
        f.write(f"Strategy PnL: ${strategy['cum_pnl'].iloc[-1]:.2f}\n")
        f.write(f"SP500 PnL: ${sp500['cum_pnl'].iloc[-1]:.2f}\n")
        f.write(f"Trading months: {len(strategy)}\n")
        f.write(f"Avg monthly PnL: ${strategy['pnl'].mean():.4f}\n")
        f.write(f"Avg positions: {strategy['signal'].mean():.1f}\n")
        f.write(f"Strategy avg monthly return: {strategy_total_return:.4%}\n")
        f.write(f"SP500 avg monthly return: {sp500_total_return:.4%}\n")

    # Dynamic y-axis limits with padding
    all_pnl = pd.concat([strategy["cum_pnl"], sp500["cum_pnl"]])
    y_min = all_pnl.min()
    y_max = all_pnl.max()
    y_pad = (y_max - y_min) * 0.1
    y_lo = y_min - y_pad
    y_hi = y_max + y_pad

    plt.figure(figsize=(12, 6))
    plt.plot(sp500.index, sp500["cum_pnl"], label="SP500", linewidth=2)
    plt.plot(strategy.index, strategy["cum_pnl"], label="Strategy", linewidth=2)

    plt.xlim(strategy.index.min(), strategy.index.max())
    plt.ylim(y_lo, y_hi)

    plt.title("Cumulative Performance")
    plt.xlabel("Date")
    plt.ylabel("PnL ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/plots/performance.png", dpi=150)
    plt.close()

    print("Backtest complete. Results in results/")
    return strategy, sp500