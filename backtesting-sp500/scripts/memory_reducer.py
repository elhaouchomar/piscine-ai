import pandas as pd
import numpy as np

def reducer(data):
    for col in data.columns:
        rows = data[col]

        if is_numeric_dtype(rows):
            min_value = rows.min(skipna=True)
            max_value = rows.max(skipna=True)
            is_int_candidate = False

            if rows.dtype.kind == 'f':
                if not rows.isnull().any():
                    if np.array_equal(rows, rows.astype(int)):
                        is_int_candidate = True

            # Integer types
            if rows.dtype.kind in 'iu' or is_int_candidate:
                if min_value >= np.iinfo(np.int8).min and max_value <= np.iinfo(np.int8).max:
                    data[col] = rows.astype(np.int8)
                elif min_value >= np.iinfo(np.int16).min and max_value <= np.iinfo(np.int16).max:
                    data[col] = rows.astype(np.int16)
                elif min_value >= np.iinfo(np.int32).min and max_value <= np.iinfo(np.int32).max:
                    data[col] = rows.astype(np.int32)
                else:
                    data[col] = rows.astype(np.int64)

            # Float types
            elif rows.dtype.kind == 'f':
                data[col] = rows.astype(np.float32)
            
        else:
            # Handle Strings and Dates
            if rows.dtype == 'object':
                if 'date' in col.lower():
                    data[col] = pd.to_datetime(rows)
                else:
                    data[col] = rows.astype('category')

    return data

def memory_reducer(paths):
    prices = reducer(paths[0])
    sp500  = reducer(paths[1])

    prices_mb = prices.memory_usage(deep=True).sum() / 1024 ** 2
    sp500_mb  = sp500.memory_usage(deep=True).sum() / 1024 ** 2

    print(f"Memory usage — prices: {prices_mb:.2f} MB, sp500: {sp500_mb:.3f} MB")
    return prices, sp500