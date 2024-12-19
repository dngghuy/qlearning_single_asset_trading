import os
import pickle as pkl
import config
from src.data_processing import read_data, make_label, exp_smooth
from src.feature_engineering import simple_feature_engineer
from src.grid_search import GridSearchTS
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def preprocess(data):
    train_size = int(len(data) * config.train_test_ratio)
    train, test = data[:train_size], data[train_size:]

    train, train_label = make_label(exp_smooth(train), seq_len=config.seq_len)
    test, test_label = make_label(exp_smooth(test), seq_len=config.seq_len)

    return simple_feature_engineer(train), train_label, simple_feature_engineer(test), test_label

def main():
    if os.path.exists(config.path_to_clf):
        with open(config.path_to_clf, 'rb') as f:
            clf = pkl.load(f)
    else:
        data = read_data(config.data_raw, Date=config.Date)
        train, train_label, test, test_label = preprocess(data)

        clf = RandomForestClassifier()
        gsts = GridSearchTS(clf, config.params, config.n_split)
        gsts.fit(train, train_label)

        clf.set_params(**gsts.best_params)
        clf.fit(train, train_label)

        preds = clf.predict(test)
        print("Accuracy:", accuracy_score(test_label, preds))
        print("Report:\n", classification_report(test_label, preds))

        with open(config.path_to_clf, 'wb') as f:
            pkl.dump(clf, f)

if __name__ == '__main__':
    main()
