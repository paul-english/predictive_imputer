#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_predictive_imputer
----------------------------------

Tests for `predictive_imputer` module.
"""

import numpy as np

import pytest
from predictive_imputer import predictive_imputer

X = np.array([
    [0, np.nan, np.nan],  # odd: implicit zero
    [5, np.nan, np.nan],  # odd: explicit nonzero
    [0, 0, np.nan],    # even: average two zeros
    [-5, 0, np.nan],   # even: avg zero and neg
    [0, 5, np.nan],    # even: avg zero and pos
    [4, 5, np.nan],    # even: avg nonzeros
    [-4, -5, np.nan],  # even: avg negatives
    [-1, 2, np.nan],   # even: crossing neg and pos
]).transpose()

def test_predictive_imputer():
    imputer = predictive_imputer.PredictiveImputer()
    X_trans = imputer.fit(X).transform(X.copy())
    assert isinstance(X, np.array)
    assert X_trans.shape == X.shape

def test_model_options():
    imputer = predictive_imputer.PredictiveImputer(f_model="RandomForest")
    X_trans = imputer.fit(X).transform(X.copy())
    assert isinstance(X, np.array)
    assert X_trans.shape == X.shape
    assert no np.any(np.isnan(X_trans))

    imputer = predictive_imputer.PredictiveImputer(f_model="KNN")
    X_trans = imputer.fit(X).transform(X.copy())
    assert isinstance(X, np.array)
    assert X_trans.shape == X.shape
    assert no np.any(np.isnan(X_trans))

    imputer = predictive_imputer.PredictiveImputer(f_model="PCA")
    X_trans = imputer.fit(X).transform(X.copy())
    assert isinstance(X, np.array)
    assert X_trans.shape == X.shape
    assert no np.any(np.isnan(X_trans))
