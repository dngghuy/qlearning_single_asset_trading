import pandas as pd
import numpy as np
from typing import Union
from src.configs import configs
from src.configs.configs import WINDOW_LENGTH, DATE_COLUMN, CLOSE_COL, SMA_COL, RATIO, MACD_COL, RSI_COL, MOM_COL, \
    ROC_COL, WILLR_COL, VOLUME_COL, DELTA_VOLUME_COL, MFI_COL, MACD_SIGNAL_COL, MACD_HIST_COL, HIGH_COL, LOW_COL
from talib import SMA, MACD, RSI, MOM, ROC, WILLR, MFI


def read_data(filepath, has_date_col: Union[bool, str] = True):
    if has_date_col == True:
        data = pd.read_csv(filepath, index_col=DATE_COLUMN)
    else:
        data = pd.read_csv(filepath, index_col=has_date_col)
    return data


def exp_smooth(dataset, alpha=0.2):
    """
    Exponential Smoothing
    :param dataset:
    :param alpha:
    :return:
    """
    return dataset.iloc[::-1].ewm(alpha=alpha).mean().iloc[::-1]


def make_label(stock, window_length=WINDOW_LENGTH):
    sma_col = f'{SMA_COL}{window_length}'
    stock[sma_col] = stock[CLOSE_COL].rolling(window=window_length).mean()
    diff = stock[sma_col].diff(window_length).dropna()
    label = [1 if i > 0 else 0 for i in diff]
    stock = stock.iloc[window_length:].drop([sma_col], axis=1)
    return stock, label


def make_onehot(label):
    unique_label = np.unique(label)
    num_type = len(unique_label)
    one_hot = np.zeros(shape=(len(label), num_type))
    for row, ilabel in enumerate(label):
        temp = np.where(unique_label == ilabel)[0][0]
        one_hot[row, temp] = 1

    return one_hot


def simple_feature_engineer(stock):
    stock[f'{SMA_COL}_10'] = SMA(stock[CLOSE_COL], timeperiod=10)
    stock[f'{SMA_COL}_30'] = SMA(stock[CLOSE_COL], timeperiod=30)
    # MACD
    macd, macd_signal, macd_hist = MACD(
        stock[CLOSE_COL],
        fastperiod=12,
        slowperiod=26,
        signalperiod=9
    )
    stock[MACD_COL] = macd
    stock[MACD_SIGNAL_COL] = macd_signal
    stock[MACD_HIST_COL] = macd_hist
    stock[RSI_COL] = RSI(stock[CLOSE_COL], timeperiod=10)
    stock[MOM_COL] = MOM(stock[CLOSE_COL], timeperiod=12)
    stock[ROC_COL] = ROC(stock[CLOSE_COL], timeperiod=10)
    stock[WILLR_COL] = WILLR(stock[HIGH_COL], stock[LOW_COL], stock[CLOSE_COL], timeperiod=14)

    if VOLUME_COL in stock.columns:
        stock[DELTA_VOLUME_COL] = stock[VOLUME_COL].diff(10)
        stock[MFI_COL] = MFI(
            stock[HIGH_COL],
            stock[LOW_COL],
            stock[CLOSE_COL],
            stock[VOLUME_COL],
            timeperiod=12
        )

    return stock.dropna().reset_index(drop=True)


def preprocess(dataset):
    dataset.columns = [dat.lower() for dat in dataset.columns]

    #Split to train and test sets
    index = int(len(dataset) * RATIO)
    train_dataset = dataset.iloc[0:index, :]
    test_dataset = dataset.iloc[index:, :]

    #Make label
    train_dataset, train_label = make_label(train_dataset, window_length=WINDOW_LENGTH)
    test_dataset, test_label = make_label(test_dataset, window_length=WINDOW_LENGTH)

    #Exponential smoothing
    train_dataset = exp_smooth(train_dataset)
    test_dataset = exp_smooth(test_dataset)

    #Create new feature
    train_dataset = simple_feature_engineer(train_dataset)
    test_dataset = simple_feature_engineer(test_dataset)

    # Make features matrix and label matrix to be equal
    train_len_diff = len(train_label) - len(train_dataset)
    train_label = train_label[train_len_diff:]

    test_len_diff = len(test_label) - len(test_dataset)
    test_label = test_label[test_len_diff:]

    return train_dataset, train_label, test_dataset, test_label


def load_and_preprocess(filename):
    dataset = read_data(filename, has_date_col=DATE_COLUMN)
    return preprocess(dataset)

