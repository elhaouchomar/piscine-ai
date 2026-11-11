# Backtesting S&P 500 – Momentum Strategy

## Overview

This project implements a quantitative **momentum-based stock-picking strategy** using historical S&P 500 constituent data. As a quantitative analyst at a hedge fund, the goal is to clean and preprocess messy financial time series, develop a stock selection signal based on 1-year average returns, and backtest the strategy against the S&P 500 benchmark.

---

## Project Structure

```
backtesting-sp500/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── sp500.csv
│   └── stock_prices.csv
│
├── notebook/
│   └── analysis.ipynb
│
├── scripts/
│   ├── memory_reducer.py
│   ├── preprocessing.py
│   ├── create_signal.py
│   ├── backtester.py
│   └── main.py
│
└── results/
    ├── plots/
    │   ├── missing_per_year.png
    │   ├── outliers_per_ticker.png
    │   ├── average_price_over_time.png
    │   └── performance.png
    ├── results.txt
    └── outliers.txt
```

---

## Getting Started

### 1. Clone
```
git clone  https://learn.zone01oujda.ma/git/oelhaouc/backtesting-sp500.git
```
### 2. Create virtual environment
```
python3 -m venv sp500
```

### 3. Activate environment
```
source ./sp500/bin/activate
```

### 4. Install dependencies
```
pip install -r requirements.txt
```

### 5. Run the complete pipeline
```
python ./scripts/main.py
```

### 6. Explore EDA notebook
```
jupyter lab
```

---

## Data Description

| File | Description |
|------|-------------|
| `sp500.csv` | S&P 500 index historical data (date, OHLC, adjusted close, volume) |
| `stock_prices.csv` | Close prices for all current and past S&P 500 constituents |

> **Note:** The dataset contains intentional data quality issues: missing values, price spikes, stock splits, dividend adjustments, and changing S&P 500 composition over time. Preprocessing handles the major issues without overfitting to the noise.

---

## Module Descriptions

### `memory_reducer.py`

Loads a CSV file and returns a memory-optimized DataFrame.

- Converts float columns to `np.float32` (minimum precision, as required)
- Downcasts integer columns to the smallest fitting type (`uint8` → `uint32` → `float32`)
- Converts string/object columns to `category`
- Prints the resulting memory usage for both DataFrames

### `preprocessing.py`

Cleans and transforms the raw price data into a model-ready format.

- Strips column whitespace and parses dates
- Melts wide-format price data into long format (Date, ticker, price)
- Resamples to month-end frequency, keeping the last observed price
- Filters prices outside the range $0.10 – $10,000
- Computes monthly past returns and monthly future returns
- Sets return outliers to NaN (returns > 1 or < –0.5), **except for 2008–2009**
- Forward-fills missing past returns within each company independently
- Drops rows where values cannot be filled
- Asserts zero remaining null values

### `create_signal.py`

Generates the momentum trading signal.

- Computes `average_return_1y`: 12-month rolling average of past returns, grouped by ticker
- Ranks companies by `average_return_1y` within each month (rank 1 = highest)
- Sets `signal = True` for the top 20 companies each month

### `backtester.py`

Runs the backtest and computes performance metrics.

- Computes PnL as `signal × monthly_future_return` (vectorized, no loops)
- Computes strategy return as `PnL / number of active positions`
- Benchmarks against S&P 500 with a fixed $20 monthly investment
- Computes cumulative PnL using `cumsum()` for both strategies
- Raises an error if total PnL exceeds $75 (signals uncorrected outliers)
- Saves metrics to `results/results.txt`
- Saves cumulative performance chart to `results/plots/performance.png`

### `main.py`

Orchestrates the full pipeline end-to-end:

```python
prices, sp500 = memory_reducer(paths)
prices, sp500 = preprocessing(prices, sp500)
prices        = create_signal(prices)
backtest(prices, sp500)
```

---

## Strategy Logic

### Signal Construction

1. Compute monthly past returns for each company
2. Calculate the 12-month rolling average return (1-year momentum)
3. Rank all companies within each month — rank 1 = highest average return
4. Select the top 20 companies: `signal = True`

### Benchmark

The S&P 500 benchmark uses a fixed monthly investment of $20 (matching the 20 selected stocks), applied to the S&P 500 monthly return.

---

## Performance & Conclusion

The momentum strategy selects the 20 best-performing stocks each month based on their 1-year average return, rebalancing monthly. Over the full backtesting period the strategy produces a cumulative PnL below $75, confirming that outlier preprocessing is effective.

The chart in `results/plots/performance.png` shows the cumulative PnL of the strategy versus the S&P 500 benchmark. The strategy generally tracks the benchmark with periodic outperformance driven by momentum persistence in top-returning stocks.

Key results are saved in `results/results.txt`:
- Strategy cumulative PnL
- S&P 500 benchmark cumulative PnL
- Average monthly PnL and average monthly returns for both strategies
- Number of trading months and average active positions

---

## Requirements

See `requirements.txt` for the full list of dependencies. Key libraries:

| Library | Purpose |
|---------|---------|
| `pandas` | Data manipulation and time series resampling |
| `numpy` | Numerical operations and dtype optimization |
| `matplotlib` | Performance charts and EDA plots |