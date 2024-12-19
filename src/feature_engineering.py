from talib.abstract import SMA, MACD, RSI, MOM, ROC, WILLR, MFI


def simple_feature_engineer(stock):
    stock['sma_10'] = SMA(stock, timeperiod=10)
    stock['sma_30'] = SMA(stock, timeperiod=30)
    stock['macd'] = MACD(stock)['macd']
    stock['rsi'] = RSI(stock, timeperiod=10)
    stock['mom'] = MOM(stock, timeperiod=12)
    stock['roc'] = ROC(stock, timeperiod=10)
    stock['willr'] = WILLR(stock, timeperiod=14)

    if 'volume' in stock.columns:
        stock['delta_volume'] = stock['volume'].diff(10)
        stock['mfi'] = MFI(stock, timeperiod=12)

    return stock.dropna().reset_index(drop=True)