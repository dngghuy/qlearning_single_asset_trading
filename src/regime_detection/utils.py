import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def read_data(path):
    return pd.read_csv(path)


def reverse_data(data):
    """
    Reverses the rows of a pandas DataFrame and resets the index.

    Parameters:
        data (pd.DataFrame): The DataFrame to be reversed.

    Returns:
        pd.DataFrame: A new DataFrame with reversed rows and a reset index.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    return data.iloc[::-1].reset_index(drop=True)


def make_window(data, window_size):
    X = []
    Y = []
    i = 0
    while (i + window_size) <= (len(data) - 1):
        X.append(data[i:(i + window_size)])
        Y.append(data[i + window_size])
        i += 1

    assert len(X) == len(Y)
    return X, Y


def scaling(X_train, X_test):
    minmax = MinMaxScaler()
    X_train = minmax.fit_transform(X_train)
    X_test = minmax.transform(X_test)

    return X_train, X_test


def train_test_split(data, train_ratio):
    train_size = len(data) * train_ratio

    data_train = data[0:train_size]
    data_test = data[train_size:]

    return data_train, data_test


def main(path, window_size, train_ratio):
    data = read_data(path)
    data = reverse_data(data)
    data_train, data_test = train_test_split(data, train_ratio)
    data_train, data_test = scaling(data_train, data_test)
    X_train, Y_train = make_window(data_train, window_size)
    X_test, Y_test = make_window(data_test, window_size)

    return X_train, Y_train, X_test, Y_test

