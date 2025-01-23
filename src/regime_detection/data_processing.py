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


class RateOfChange:
    def __call__(self, dataset, columns, period=1, default_suffix='roc'):
        for col in columns:
            # Calculate relative change
            dataset[f'{col}_{default_suffix}'] = dataset[col].pct_change(periods=period).fillna(0)
        return dataset


class WrapSMA:
    def __init__(self, timeperiod):
        self.timeperiod = timeperiod

    def __call__(self, dataset, use_roc: bool = False):
        if use_roc:
            dataset[f'{SMA_COL}_{self.timeperiod}'] = SMA(dataset[f'{CLOSE_COL}_roc'], timeperiod=self.timeperiod)
        else:
            dataset[f'{SMA_COL}_{self.timeperiod}'] = SMA(dataset[CLOSE_COL], timeperiod=self.timeperiod)

        return dataset


class WrapMACD:
    def __init__(self, fastperiod, slowperiod, signalperiod):
        self.fastperiod = fastperiod
        self.slowperiod = slowperiod
        self.signalperiod = signalperiod

    def __call__(self, dataset, use_roc: bool = False):
        if use_roc:
            macd, macd_signal, macd_hist = MACD(
                dataset[f'{CLOSE_COL}_roc'],
                fastperiod=self.fastperiod,
                slowperiod=self.slowperiod,
                signalperiod=self.signalperiod
            )
        else:
            macd, macd_signal, macd_hist = MACD(
                dataset[CLOSE_COL],
                fastperiod=self.fastperiod,
                slowperiod=self.slowperiod,
                signalperiod=self.signalperiod
            )

        dataset[MACD_COL] = macd
        dataset[MACD_SIGNAL_COL] = macd_signal
        dataset[MACD_HIST_COL] = macd_hist

        return dataset


class WrapRSI:
    def __init__(self, timeperiod):
        self.timeperiod = timeperiod

    def __call__(self, dataset, use_roc: bool = False):
        if use_roc:
            dataset[RSI_COL] = RSI(dataset[f'{CLOSE_COL}_roc'], timeperiod=self.timeperiod)
        else:
            dataset[RSI_COL] = RSI(dataset[CLOSE_COL], timeperiod=self.timeperiod)

        return dataset


class WrapMOM:
    def __init__(self, timeperiod):
        self.timeperiod = timeperiod

    def __call__(self, dataset, use_roc: bool = False):
        if use_roc:
            dataset[MOM_COL] = MOM(dataset[f'{CLOSE_COL}_roc'], timeperiod=self.timeperiod)
        else:
            dataset[MOM_COL] = MOM(dataset[CLOSE_COL], timeperiod=self.timeperiod)

        return dataset


class WrapROC:
    def __init__(self, timeperiod):
        self.timeperiod = timeperiod

    def __call__(self, dataset):
        dataset[f"{ROC_COL}_{self.timeperiod}"] = ROC(dataset[CLOSE_COL], timeperiod=self.timeperiod)

        return dataset


class WrapWILLR:
    def __init__(self, timeperiod):
        self.timeperiod = timeperiod

    def __call__(self, dataset):
        dataset[WILLR_COL] = WILLR(dataset[HIGH_COL], dataset[LOW_COL], dataset[CLOSE_COL], timeperiod=self.timeperiod)

        return dataset


class WrapMFI:
    def __init__(self, timeperiod):
        self.timeperiod = timeperiod

    def __call__(self, dataset):
        dataset[MFI_COL] = MFI(
            dataset[HIGH_COL],
            dataset[LOW_COL],
            dataset[CLOSE_COL],
            dataset[VOLUME_COL],
            timeperiod=self.timeperiod
        )

        return dataset


class WrapDELTA:
    def __init__(self, timeperiod):
        self.timeperiod = timeperiod

    def __call__(self, dataset):
        dataset[DELTA_VOLUME_COL] = dataset[VOLUME_COL].diff(self.timeperiod)

        return dataset


class ExpSmooth:
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, dataset):
        return dataset.iloc[::-1].ewm(alpha=self.alpha).mean().iloc[::-1]


class DataFeatureEngineer:
    def __init__(self, transformation_list):
        self.transformation_list = transformation_list
        self.feature_list = []

    def process(self, dataset):
        old_columns = dataset.columns
        for transformation in self.transformation_list:
            dataset = transformation(dataset)
        self.feature_list = [i for i in dataset.columns if i not in old_columns]

        return dataset


def make_label(stock, window_length=WINDOW_LENGTH):
    sma_col = f'{SMA_COL}{window_length}'
    stock[sma_col] = stock[CLOSE_COL].rolling(window=window_length).mean()
    diff = stock[sma_col].diff(window_length).dropna()
    roc = (diff / stock[sma_col].diff(window_length)) * 100
    label = [1 if i >= 5 else 0 for i in diff]
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


def preprocess(dataset, transformation_list):
    """

    Args:
        dataset:

    Returns:

    """

    dataset.columns = [dat.lower() for dat in dataset.columns]
    # Split to train and test sets
    index = int(len(dataset) * RATIO)
    train_dataset = dataset.iloc[0:index, :]
    test_dataset = dataset.iloc[index:, :]

    # Make label
    train_dataset, train_label = make_label(train_dataset, window_length=WINDOW_LENGTH)
    test_dataset, test_label = make_label(test_dataset, window_length=WINDOW_LENGTH)

    # exp_smooth = ExpSmooth(alpha=0.2)

    feature_engineer = DataFeatureEngineer(transformation_list)

    # Exponential smoothing
    # train_dataset = exp_smooth(train_dataset)
    # test_dataset = exp_smooth(test_dataset)

    # Create new feature
    train_dataset = feature_engineer.process(train_dataset)
    train_dataset = train_dataset.dropna().reset_index(drop=True)
    train_dataset = train_dataset[feature_engineer.feature_list]
    test_dataset = feature_engineer.process(test_dataset)
    test_dataset = test_dataset.dropna().reset_index(drop=True)
    test_dataset = test_dataset[feature_engineer.feature_list]

    # Make features matrix and label matrix to be equal
    train_len_diff = len(train_label) - len(train_dataset)
    train_label = train_label[train_len_diff:]

    test_len_diff = len(test_label) - len(test_dataset)
    test_label = test_label[test_len_diff:]

    return train_dataset, train_label, test_dataset, test_label, feature_engineer


