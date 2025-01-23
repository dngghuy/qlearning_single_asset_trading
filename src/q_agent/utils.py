import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
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


def get_current_state_single_stock(data, clf):
    """
    data should be cleaned
    """
    # getting other cols

    pred_t = clf.predict_proba(data)[0][0]

    return np.hstack([data.values.reshape(-1), np.array([pred_t])]).reshape(1, -1)


def remove_prefix(state_dict, prefix):
    """
    Removes a prefix from the keys of a state dictionary.

    Args:
        state_dict (OrderedDict): The state dictionary to modify.
        prefix (str): The prefix to remove.

    Returns:
        OrderedDict: The modified state dictionary.
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[len(prefix) + 1:] if k.startswith(prefix + '.') else k  # remove the prefix
        new_state_dict[name] = v
    return new_state_dict


def plot_metrics(trainer, filename, window_size=50):
    """
    Plots various metrics tracked during training on a single subplot.

    Args:
        trainer: The Trainer object containing the training data.
        window_size: The window size for the moving average calculation.
    """

    # Moving average function (keep this inside plot_metrics for clarity)
    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    # --- Create the figure and subplot ---
    plt.figure(figsize=(12, 6))
    ax = plt.subplot(111)

    # --- Plot rewards per episode ---
    ax.plot(trainer.rewards_per_episode, label='Rewards per Episode', color='blue')
    ax.plot(moving_average(trainer.rewards_per_episode, window_size), label=f'Moving Average Reward (Window={window_size})', color='skyblue')

    # --- Plot loss history ---
    ax.plot(trainer.loss_history, label='Average Loss per Episode', color='orange')

    # --- Plot action counts ---
    for action_type, counts in trainer.action_counts.items():
        ax.plot(counts, label=f'{action_type.capitalize()} Actions', linestyle='--')

    # --- Plot profit history ---
    ax.plot(trainer.profit_history, label='Profit per Episode', color='green')
    ax.plot(moving_average(trainer.profit_history, window_size), label=f'Moving Average Profit (Window={window_size})', color='limegreen')

    # --- Set labels and title ---
    ax.set_xlabel('Episode')
    ax.set_ylabel('Value')
    ax.set_title('Training Metrics')

    # --- Add legend ---
    ax.legend()

    plt.savefig(filename)

