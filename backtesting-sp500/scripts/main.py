import sys
sys.path.append('./scripts')  # Ajouter cette ligne

from memory_reducer import memory_reducer
from create_signal import create_signal
from backtester import backtest
from preprocessing import preprocessing

paths = ["./data/stock_prices.csv", "./data/sp500.csv"]

print("Loading data...")
prices, sp500 = memory_reducer(paths)

print("Preprocessing...")
prices, sp500 = preprocessing(prices, sp500)

print("Creating signal...")
prices = create_signal(prices)

print("Running backtest...")
backtest(prices, sp500)

print("Pipeline completed!")