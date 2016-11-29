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


def test_predictive_imputer():
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

    imputer = predictive_imputer.PredictiveImputer()
    X_trans = imputer.fit(X).transform(X.copy())
