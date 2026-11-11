import pandas as pd

def create_signal(prices):
    prices = prices.copy()
    
    # average return per company (12 months)
    prices['average_return_1y'] = (
        prices.groupby('ticker')['monthly_past_return']
        .rolling(12, min_periods=12)
        .mean()
        .reset_index(level=0, drop=True)
    )
    
    # remove rows without full 1y history
    prices = prices.dropna(subset=['average_return_1y'])
    
    # Rank companies
    prices['rank'] = (
        prices.groupby(level='Date')['average_return_1y']
        .rank(method='first', ascending=False)
    )
    
    # Signal = Top 20
    prices['signal'] = prices['rank'] <= 20
    
    return prices