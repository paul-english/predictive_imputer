# -*- coding: utf-8 -*-
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class PredictiveImputer(BaseEstimator, TransformerMixin):
    def __init__(self, N=10, initial_strategy='mean'):
        self.N = N
        self.initial_strategy = initial_strategy
        self.initial_imputer = Imputer(strategy=initial_strategy)

    def fit(self, X, y=None):
        X = check_array(X, dtype=np.float64, force_all_finite=False)

        most_by_nan = np.isnan(X).sum(axis=0).argsort()[::-1]

        imputed = self.initial_imputer.fit_transform(X)

        self.statistics_ = np.ma.getdata(X)
        self.trees_ = [None for n in range(len(most_by_nan))]
        self.gamma_ = []

        new_imputed = imputed.copy()
        for iter in range(self.N):
            last_imputed = new_imputed.copy()
            
            for i in most_by_nan:
                X_s = np.delete(last_imputed, i, 1)
                y_s = X[:, i]

                X_train = X_s[~np.isnan(y_s)]
                y_train = y_s[~np.isnan(y_s)]

                X_unk = X_s[np.isnan(y_s)]

                clf = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=i)
                clf.fit(X_train, y_train)

                if len(X_unk) > 0:
                    new_imputed[np.isnan(y_s), i] = clf.predict(X_unk)
                self.trees_[i] = clf
                
            diff = np.linalg.norm(new_imputed-last_imputed)/new_imputed.std()
            if diff < 0.001:
                break
            
            gamma = np.linalg.norm(new_imputed-imputed)/new_imputed.std()
            self.gamma_.append(gamma)

        return self

    def transform(self, X):
        check_is_fitted(self, ['statistics_', 'trees_', 'gamma_'])
        X = check_array(X, copy=True, dtype=np.float64, force_all_finite=False)
        if X.shape[1] != self.statistics_.shape[1]:
            raise ValueError("X has %d features per sample, expected %d"
                             % (X.shape[1], self.statistics_.shape[1]))
        imputed = self.initial_imputer.fit_transform(X)
        for i, clf in enumerate(self.trees_):
            X_s = np.delete(imputed, i, 1)
            y_s = X[:, i]

            X_unk = X_s[np.isnan(y_s)]
            if len(X_unk) > 0:
                X[np.isnan(y_s), i] = clf.predict(X_unk)

        return X
