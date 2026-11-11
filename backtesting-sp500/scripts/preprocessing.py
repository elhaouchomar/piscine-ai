import pandas as pd
import numpy as np

def preprocessing(prices, sp500):
    prices.columns = [c.strip() for c in prices.columns]
    sp500.columns = [c.strip() for c in sp500.columns]

    prices['Date'] = pd.to_datetime(prices['Date'])
    sp500['Date'] = pd.to_datetime(sp500['Date'])

    if 'Ticker' not in prices.columns:
        prices = prices.melt(id_vars='Date', var_name='ticker', value_name='price')
    else:
        prices = prices.rename(columns={'Ticker': 'ticker'})

    prices = prices.sort_values(['ticker', 'Date'])

    prices = (
        prices.set_index('Date')
        .groupby('ticker')
        .resample('ME', include_groups=False)
        .last()
        .reset_index()
    )

    prices = prices[(prices['price'] >= 0.1) & (prices['price'] <= 10000)]

    prices['monthly_past_return'] = prices.groupby('ticker')['price'].pct_change()
    prices['monthly_future_return'] = prices.groupby('ticker')['price'].shift(-1) / prices['price'] - 1

    crisis = prices['Date'].dt.year.isin([2008, 2009])

    past_outlier = (prices['monthly_past_return'] > 1) | (prices['monthly_past_return'] < -0.5)
    future_outlier = (prices['monthly_future_return'] > 1) | (prices['monthly_future_return'] < -0.5)

    prices.loc[~crisis & past_outlier, 'monthly_past_return'] = np.nan
    prices.loc[~crisis & future_outlier, 'monthly_future_return'] = np.nan

    # Fill missing values within each company independently.
    # Only fill monthly_past_return — do NOT ffill monthly_future_return,
    # because NaN at the last row means data ends there; filling it would
    # introduce look-ahead bias.
    prices['monthly_past_return'] = (
        prices.groupby('ticker')['monthly_past_return'].ffill()
    )

    prices = prices.dropna(subset=['monthly_past_return', 'monthly_future_return'])

    assert prices.isnull().sum().sum() == 0

    prices = prices.set_index(['Date', 'ticker'])

    sp500 = (
        sp500.sort_values('Date')
        .set_index('Date')
        .resample('ME')
        .last()
    )

    sp500['monthly_return'] = sp500['Adjusted Close'].pct_change()
    sp500 = sp500.dropna()

    return prices, sp500