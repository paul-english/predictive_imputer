# -*- coding: utf-8 -*-
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class PredictiveImputer(BaseEstimator, TransformerMixin):
    def __init__(self, max_iter=10, initial_strategy='mean', tol=1e-6, f_model = "RandomForest"):
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

        self.statistics_ = np.ma.getdata(X)
        self.gamma_ = []
        
        if self.f_model == "RandomForest":
            self.estimators_ = [None]*X.shape[1]
        
        new_imputed = imputed.copy()
        for iter in range(self.max_iter):
            
            
            if self.f_model == "RandomForest":
                for i in most_by_nan:
                    X_s = np.delete(new_imputed, i, 1)
                    y_nan = X_nan[:, i]

                    X_train = X_s[~y_nan]
                    y_train = new_imputed[~y_nan, i]

                    X_unk = X_s[y_nan]

                    clf = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=i, **kwargs)
                    clf.fit(X_train, y_train)

                    if len(X_unk) > 0:
                        new_imputed[y_nan, i] = clf.predict(X_unk)
                    self.estimators_[i] = clf
                    
            elif self.f_model == "PCA":
                self.estimator_ = PCA(n_components=int(np.sqrt(min(X.shape))), whiten=True, **kwargs)
                self.estimator_.fit(new_imputed)
                new_imputed[np.isnan(X)] = self.estimator_.inverse_transform(self.estimator_.transform(new_imputed))[np.isnan(X)]
            
            gamma = (np.linalg.norm(new_imputed-imputed, axis = 0)/(self.tol+np.linalg.norm(new_imputed-new_imputed.mean(axis=0), axis=0))).mean()
            self.gamma_.append(gamma)
            
            if np.abs(np.diff(self.gamma_[-2:])) < self.tol:
                break

        return self

    def transform(self, X):
        check_is_fitted(self, ['statistics_', 'gamma_'])
        X = check_array(X, copy=True, dtype=np.float64, force_all_finite=False)
        if X.shape[1] != self.statistics_.shape[1]:
            raise ValueError("X has %d features per sample, expected %d"
                             % (X.shape[1], self.statistics_.shape[1]))
        
        X_nan = np.isnan(X)
        imputed = self.initial_imputer.fit_transform(X)
        
        if self.f_model == "RandomForest":
            for i, clf in enumerate(self.estimators_):
                X_s = np.delete(imputed, i, 1)
                y_nan = X_nan[:, i]

                X_unk = X_s[y_nan]
                if len(X_unk) > 0:
                    X[y_nan, i] = clf.predict(X_unk)
                    
        elif self.f_model == "PCA":
            X[X_nan] = self.estimator_.inverse_transform(self.estimator_.transform(imputed))[X_nan]

        return X