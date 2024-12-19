import pandas as pd
from configs import configs

def read_data(filepath, Date=True):
    if Date:
        data = pd.read_csv(filepath, index_col='Date')
    else:
        data = pd.read_csv(filepath, index_col=Date)
    return data

def exp_smooth(dataset, alpha=0.2):
    return dataset.iloc[::-1].ewm(alpha=alpha).mean().iloc[::-1]

def make_label(stock, seq_len=10):
    stock['sma10'] = stock['close'].rolling(window=10).mean()
    diff = stock['sma10'].diff(seq_len).dropna()
    label = [1 if i > 0 else -1 for i in diff]
    stock = stock.iloc[seq_len:].drop(['sma10'], axis=1)
    return stock, label
