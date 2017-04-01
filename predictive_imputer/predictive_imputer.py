# -*- coding: utf-8 -*-
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import Imputer
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class PredictiveImputer(BaseEstimator, TransformerMixin):
    def __init__(self, max_iter=10, initial_strategy='mean', tol=1e-3, f_model="RandomForest"):
        self.max_iter = max_iter
        self.initial_strategy = initial_strategy
        self.initial_imputer = Imputer(strategy=initial_strategy)
        self.tol = tol
        self.f_model = f_model

    def fit(self, X, y=None, **kwargs):
        X = check_array(X, dtype=np.float64, force_all_finite=False)

        X_nan = np.isnan(X)
        most_by_nan = X_nan.sum(axis=0).argsort()[::-1]

        imputed = self.initial_imputer.fit_transform(X)
        new_imputed = imputed.copy()

        self.statistics_ = np.ma.getdata(X)
        self.gamma_ = []

        if self.f_model == "RandomForest":
            self.estimators_ = [RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=i, **kwargs) for i in range(X.shape[1])]
        elif self.f_model == "KNN":
            self.estimators_ = [KNeighborsRegressor(n_neighbors=min(5, sum(~X_nan[:, i])), **kwargs) for i in range(X.shape[1])]
        elif self.f_model == "PCA":
            self.estimators_ = [PCA(n_components=int(np.sqrt(min(X.shape))), whiten=True, **kwargs)]

        for iter in range(self.max_iter):
            if len(self.estimators_) > 1:
                for i in most_by_nan:

                    X_s = np.delete(new_imputed, i, 1)
                    y_nan = X_nan[:, i]

                    X_train = X_s[~y_nan]
                    y_train = new_imputed[~y_nan, i]
                    X_unk = X_s[y_nan]

                    estimator_ = self.estimators_[i]
                    estimator_.fit(X_train, y_train)
                    if len(X_unk) > 0:
                        new_imputed[y_nan, i] = estimator_.predict(X_unk)

            else:
                estimator_ = self.estimators_[0]
                estimator_.fit(new_imputed)
                new_imputed[X_nan] = estimator_.inverse_transform(estimator_.transform(new_imputed))[X_nan]

            gamma = ((new_imputed-imputed)**2/(1e-6+new_imputed.var(axis=0))).sum()/(1e-6+X_nan.sum())
            self.gamma_.append(gamma)
            if np.abs(np.diff(self.gamma_[-2:])) < self.tol:
                break

        return self

    def transform(self, X):
        check_is_fitted(self, ['statistics_', 'estimators_', 'gamma_'])
        X = check_array(X, copy=True, dtype=np.float64, force_all_finite=False)
        if X.shape[1] != self.statistics_.shape[1]:
            raise ValueError("X has %d features per sample, expected %d"
                             % (X.shape[1], self.statistics_.shape[1]))

        X_nan = np.isnan(X)
        imputed = self.initial_imputer.fit_transform(X)

        if len(self.estimators_) > 1:
            for i, estimator_ in enumerate(self.estimators_):
                X_s = np.delete(imputed, i, 1)
                y_nan = X_nan[:, i]

                X_unk = X_s[y_nan]
                if len(X_unk) > 0:
                    X[y_nan, i] = estimator_.predict(X_unk)

        else:
            estimator_ = self.estimators_[0]
            X[X_nan] = estimator_.inverse_transform(estimator_.transform(imputed))[X_nan]

        return X
