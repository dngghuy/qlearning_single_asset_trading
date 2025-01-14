import numpy as np
import pickle as pkl

from src.configs.configs import DATE_COLUMN, TRAIN_RF_PATH, RF_GRID_PARAMS, N_SPLIT, RF_MODEL_PATH
from src.regime_detection.data_processing import read_data, preprocess
from src.regime_detection.grid_search import GridSearchTS
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def train_trend_detector(data_path):
    data = read_data(data_path, has_date_col=DATE_COLUMN)
    train, train_label, test, test_label = preprocess(data)

    train_label = np.array(train_label).ravel()
    test_label = np.array(test_label).ravel()

    clf = RandomForestClassifier()
    gsts = GridSearchTS(clf, RF_GRID_PARAMS, N_SPLIT)
    gsts.fit(train, train_label)

    clf.set_params(**gsts.best_params)
    clf.fit(train, train_label)

    preds = clf.predict(test)
    print("Accuracy:", accuracy_score(test_label, preds))
    print("Report:\n", classification_report(test_label, preds))

    with open(RF_MODEL_PATH, 'wb') as f:
        pkl.dump(clf, f)


def load_trend_detector():
    with open(RF_MODEL_PATH, 'rb') as f:
        clf = pkl.load(f)
    return clf


if __name__ == '__main__':
    train_trend_detector(data_path=TRAIN_RF_PATH)
