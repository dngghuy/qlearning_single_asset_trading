from itertools import product
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score


class GridSearchTS:
    def __init__(self, clf, params, n_split):
        self.clf = clf
        self.params = params
        self.n_split = n_split
        self.best_params = None

    def fit(self, X, y):
        tscv = TimeSeriesSplit(n_splits=self.n_split)
        best_score = 0

        for param_comb in product(*self.params.values()):
            param_dict = dict(zip(self.params.keys(), param_comb))
            self.clf.set_params(**param_dict)
            scores = []

            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                self.clf.fit(X_train, y_train)
                preds = self.clf.predict(X_val)
                scores.append(accuracy_score(y_val, preds))

            avg_score = sum(scores) / len(scores)
            if avg_score > best_score:
                best_score = avg_score
                self.best_params = param_dict

        return best_score
