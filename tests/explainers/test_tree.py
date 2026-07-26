# pylint: disable=missing-function-docstring
"""Core FastTreeSHAP tests: the v0/v1/v2 algorithms must agree with each other
for SHAP values and SHAP interaction values, across model backends and tasks."""
import numpy as np
import pytest
import sklearn.ensemble

import fasttreeshap

BACKENDS = ["sklearn", "xgboost", "lightgbm"]
TASKS = ["regression", "multiclass"]


def _make_fitted_models(backend, task):
    """Return (X, [fitted models], explainer_kwargs) for a backend/task combo."""
    X, y = fasttreeshap.datasets.boston() if task == "regression" else fasttreeshap.datasets.iris()

    if backend == "sklearn":
        models = [
            sklearn.ensemble.RandomForestRegressor(n_estimators=100, max_depth=6),
            sklearn.ensemble.ExtraTreesRegressor(n_estimators=100, max_depth=6),
        ]
        kwargs = {}
    elif backend == "xgboost":
        xgboost = pytest.importorskip("xgboost")
        models = [xgboost.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1)]
        kwargs = {"shortcut": False}
    else:  # lightgbm
        lightgbm = pytest.importorskip("lightgbm")
        models = [lightgbm.LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1)]
        kwargs = {"shortcut": False}

    for model in models:
        model.fit(X, y)
    return X, models, kwargs


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("task", TASKS)
def test_fasttreeshap_value_agreement(backend, task):
    """FastTreeSHAP v1 and v2 must produce the same SHAP values as v0 (TreeSHAP)."""
    X, models, kwargs = _make_fitted_models(backend, task)
    for model in models:
        v0 = fasttreeshap.TreeExplainer(model, algorithm="v0", **kwargs)(X).values
        v1 = fasttreeshap.TreeExplainer(model, algorithm="v1", **kwargs)(X).values
        v2 = fasttreeshap.TreeExplainer(model, algorithm="v2", **kwargs)(X).values
        assert np.allclose(v0, v1)
        assert np.allclose(v0, v2)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("task", TASKS)
def test_fasttreeshap_interaction_agreement(backend, task):
    """FastTreeSHAP v1 must produce the same SHAP interaction values as v0."""
    X, models, kwargs = _make_fitted_models(backend, task)
    for model in models:
        v0 = fasttreeshap.TreeExplainer(model, algorithm="v0", **kwargs)(X, interactions=True).values
        v1 = fasttreeshap.TreeExplainer(model, algorithm="v1", **kwargs)(X, interactions=True).values
        assert np.allclose(v0, v1)
