# pylint: disable=missing-function-docstring,too-many-lines,fixme
"""Test tree functions."""
import itertools
import math
import pickle
import numpy as np
import pandas as pd
import pytest
import sklearn
import fasttreeshap


def test_fasttreeshap_xgboost():
    xgboost = pytest.importorskip('xgboost')

    # train XGBoost model
    X, y = fasttreeshap.datasets.boston()
    model = xgboost.train({"learning_rate": 0.01, "silent": 1}, xgboost.DMatrix(X, label=y), 100)

    # compute SHAP values using FastTreeSHAP v0 (i.e., original TreeSHAP)
    explainer_v0 = fasttreeshap.TreeExplainer(model, algorithm = "v0", shortcut = False)
    shap_values_v0 = explainer_v0(X).values

    # compute SHAP values using FastTreeSHAP v1
    explainer_v1 = fasttreeshap.TreeExplainer(model, algorithm = "v1", shortcut = False)
    shap_values_v1 = explainer_v1(X).values

    # compute SHAP values using FastTreeSHAP v2
    explainer_v2 = fasttreeshap.TreeExplainer(model, algorithm = "v2", shortcut = False)
    shap_values_v2 = explainer_v2(X).values

    # justify the correctness of FastTreeSHAP v1 in SHAP value computation
    assert np.allclose(shap_values_v0, shap_values_v1)

    # justify the correctness of FastTreeSHAP v2 in SHAP value computation
    assert np.allclose(shap_values_v0, shap_values_v2)


def test_fasttreeshap_sklearn():

    # train model
    X, y = fasttreeshap.datasets.boston()
    models = [
        sklearn.ensemble.RandomForestRegressor(n_estimators=10),
        sklearn.ensemble.ExtraTreesRegressor(n_estimators=10),
    ]
    for model in models:
        model.fit(X, y)

        # compute SHAP values using FastTreeSHAP v0 (i.e., original TreeSHAP)
        explainer_v0 = fasttreeshap.TreeExplainer(model, algorithm = "v0")
        shap_values_v0 = explainer_v0(X).values

        # compute SHAP values using FastTreeSHAP v1
        explainer_v1 = fasttreeshap.TreeExplainer(model, algorithm = "v1")
        shap_values_v1 = explainer_v1(X).values

        # compute SHAP values using FastTreeSHAP v2
        explainer_v2 = fasttreeshap.TreeExplainer(model, algorithm = "v2")
        shap_values_v2 = explainer_v2(X).values

        # justify the correctness of FastTreeSHAP v1 in SHAP value computation
        assert np.allclose(shap_values_v0, shap_values_v1)

        # justify the correctness of FastTreeSHAP v2 in SHAP value computation
        assert np.allclose(shap_values_v0, shap_values_v2)


def test_fasttreeshap_xgboost_interaction():
    xgboost = pytest.importorskip('xgboost')

    # train XGBoost model
    X, y = fasttreeshap.datasets.boston()
    model = xgboost.train({"learning_rate": 0.01, "silent": 1}, xgboost.DMatrix(X, label=y), 100)

    # compute SHAP interaction values using FastTreeSHAP v0 (i.e., original TreeSHAP)
    explainer_v0 = fasttreeshap.TreeExplainer(model, algorithm = "v0", shortcut = False)
    shap_interaction_values_v0 = explainer_v0(X, interactions = True).values

    # compute SHAP interaction values using FastTreeSHAP v1
    explainer_v1 = fasttreeshap.TreeExplainer(model, algorithm = "v1", shortcut = False)
    shap_interaction_values_v1 = explainer_v1(X, interactions = True).values

    # justify the correctness of FastTreeSHAP v1 in SHAP interaction value computation
    assert np.allclose(shap_interaction_values_v0, shap_interaction_values_v1)


def test_fasttreeshap_xgboost_sklearn():

    # train model
    X, y = fasttreeshap.datasets.boston()
    models = [
        sklearn.ensemble.RandomForestRegressor(n_estimators=10, max_depth=6),
        sklearn.ensemble.ExtraTreesRegressor(n_estimators=10, max_depth=6),
    ]
    for model in models:
        model.fit(X, y)

        # compute SHAP interaction values using FastTreeSHAP v0 (i.e., original TreeSHAP)
        explainer_v0 = fasttreeshap.TreeExplainer(model, algorithm = "v0")
        shap_interaction_values_v0 = explainer_v0(X, interactions = True).values

        # compute SHAP interaction values using FastTreeSHAP v1
        explainer_v1 = fasttreeshap.TreeExplainer(model, algorithm = "v1")
        shap_interaction_values_v1 = explainer_v1(X, interactions = True).values

        # justify the correctness of FastTreeSHAP v1 in SHAP interaction value computation
        assert np.allclose(shap_interaction_values_v0, shap_interaction_values_v1)