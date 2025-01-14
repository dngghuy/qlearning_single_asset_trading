
import pandas as pd
import numpy as np

from src.configs.configs import CLOSE_COL, TICKER_COL, VOLUME_COL


def telling_trend(difference):
    """
    Define whether it is bullish or bearish market. 1 for uptrend and 0 for
    downtrend
    """
    if difference >= 0:
        return 1
    else:
        return 0


def making_trend(data, side_window):
    """
    Catching the trend of the given stock.
    Args:
        data: Input dataframe
        side_window: Determine how many timestep to be included in calculating
        the moving average
    """
    if type(data) == pd.DataFrame:
        difference = [data[CLOSE_COL][i+side_window] - data[CLOSE_COL][i] for i in range(len(data)-side_window)]
        trend = [telling_trend(i) for i in difference]
    else:
        difference = [data[i+side_window] - data[i] for i in range(len(data)-side_window)]
        trend = np.array([telling_trend(i) for i in difference])

    return trend


def get_stock(path):
    data = pd.read_csv(path)
    data = data.iloc[::-1].reset_index(drop=True)

    return data


def get_current_state(data, wealth, tickers, portfolio, clf, t):
    wealth_t = wealth + [portfolio[ticker] * data.loc[data[TICKER_COL] == ticker][CLOSE_COL][t] for ticker in tickers]
    data_t = data.iloc[t, :]

    state_t = data_t[CLOSE_COL]
    volume_t = data_t[VOLUME_COL]
    print(data_t)
    pred_t = clf.predict(data_t)

    return np.array([volume_t, state_t, wealth_t, pred_t]).reshape(1, -1)


def initialize_portfolio(data, wealth):
    all_stocks = np.unique(np.array(data[TICKER_COL]))
    port = {}
    for stock in all_stocks:
        port[stock] = [0]
    port['cash'] = [wealth]

    return port, all_stocks


def get_current_state_single_stock(data, ticker, portfolio, clf, t):
    """
    data should be cleaned
    """
    # getting other cols

    pred_t = clf.predict(data)[0]

    return np.hstack([data.values.reshape(-1), np.array([pred_t])]).reshape(1, -1)
