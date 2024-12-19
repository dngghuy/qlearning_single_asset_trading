
import pandas as pd
import numpy as np


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
        difference = [data.CLOSE[i+side_window] - data.CLOSE[i] for i in range(len(data)-side_window)]
        trend = [telling_trend(i) for i in difference]
    else:
        difference = [data[i+side_window] - data[i] for i in range(len(data)-side_window)]
        trend = np.array([telling_trend(i) for i in difference])

    return trend


def get_stock(path):
    data = pd.read_csv(path)
    data = data.iloc[::-1].reset_index(drop=True)

    return data


def get_current_state(data, wealth, portfolio, clf, t):
    W_t = wealth + portfolio[data.TICKER[t]] * data.CLOSE[t]
    data_t = data.iloc[t, :]

    S_t = data_t.CLOSE
    V_t = data_t.VOLUME
    print(data_t)
    pred_t = clf.predict(data_t)

    return np.array([V_t, S_t, W_t, pred_t]).reshape(1, -1)


def initialize_portfolio(data):
    all_stocks = np.unique(np.array(data.TICKER))
    port = {}
    for stock in all_stocks:
        port[stock] = 0

    return port, all_stocks[0]