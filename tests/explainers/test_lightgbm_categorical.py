""" Regression tests for LightGBM categorical ("==" / set-membership) splits.

These are intentionally self-contained (they depend only on ``fasttreeshap``,
``numpy`` and ``lightgbm``) so they can run even when the heavier comparison
stack used by ``test_tree.py`` is unavailable.

Ground truth is LightGBM's own TreeSHAP implementation, exposed through
``Booster.predict(..., pred_contrib=True)``, which handles categorical splits
correctly. FastTreeSHAP's ``tree_path_dependent`` output should match it to
floating-point precision for regression and binary models.
"""
import numpy as np
import pytest

import fasttreeshap

lgb = pytest.importorskip("lightgbm")


def _make_dataset(seed, n, cardinality, n_cat=1, extra_numeric=1):
    rng = np.random.RandomState(seed)
    cols = [rng.randn(n)]
    cat_idx = []
    for _ in range(n_cat):
        cols.append(rng.randint(0, cardinality, size=n).astype(float))
        cat_idx.append(len(cols) - 1)
    for _ in range(extra_numeric):
        cols.append(rng.randn(n))
    X = np.column_stack(cols).astype(np.float64)
    y = 1.5 * X[:, 0] + 0.7 * X[:, -1]
    for ci in cat_idx:
        effect = rng.randn(int(X[:, ci].max()) + 1) * 2.0
        y = y + effect[X[:, ci].astype(int)]
    return X, y, cat_idx, rng


def _train(X, y, cat_idx, objective="regression", num_class=None, rounds=40):
    params = dict(num_leaves=31, min_data_in_leaf=20, learning_rate=0.1, verbose=-1)
    if objective == "regression":
        params["objective"] = "regression"
        label = y
    elif objective == "binary":
        params["objective"] = "binary"
        label = (y > np.median(y)).astype(int)
    else:
        raise ValueError(objective)
    ds = lgb.Dataset(X, label=label, categorical_feature=cat_idx, free_raw_data=False)
    return lgb.train(params, ds, num_boost_round=rounds)


def _assert_has_categorical_split(booster):
    seen = set()

    def walk(node):
        if "decision_type" in node:
            seen.add(node["decision_type"])
            walk(node["left_child"])
            walk(node["right_child"])

    for tree in booster.dump_model()["tree_info"]:
        walk(tree["tree_structure"])
    assert "==" in seen, "test model unexpectedly has no categorical splits"


@pytest.mark.parametrize("algorithm", ["v0", "v1", "v2"])
@pytest.mark.parametrize("cardinality", [5, 70])  # 70 exercises the multi-word bitset
def test_categorical_matches_lightgbm_regression(algorithm, cardinality):
    X, y, cat_idx, _ = _make_dataset(seed=0, n=4000, cardinality=cardinality)
    booster = _train(X, y, cat_idx, "regression", rounds=60)
    _assert_has_categorical_split(booster)

    Xtest = X[:100]
    ground_truth = booster.predict(Xtest, pred_contrib=True)[:, :-1]

    explainer = fasttreeshap.TreeExplainer(
        booster, feature_perturbation="tree_path_dependent", algorithm=algorithm
    )
    sv = np.asarray(explainer.shap_values(Xtest, check_additivity=True))
    assert np.abs(sv - ground_truth).max() < 1e-6


@pytest.mark.parametrize("algorithm", ["v0", "v1", "v2"])
def test_categorical_multiple_features(algorithm):
    X, y, cat_idx, _ = _make_dataset(seed=3, n=4000, cardinality=12, n_cat=2)
    booster = _train(X, y, cat_idx, "regression", rounds=50)
    _assert_has_categorical_split(booster)

    Xtest = X[:100]
    ground_truth = booster.predict(Xtest, pred_contrib=True)[:, :-1]
    explainer = fasttreeshap.TreeExplainer(
        booster, feature_perturbation="tree_path_dependent", algorithm=algorithm
    )
    sv = np.asarray(explainer.shap_values(Xtest, check_additivity=True))
    assert np.abs(sv - ground_truth).max() < 1e-6


def test_categorical_binary_classification():
    X, y, cat_idx, _ = _make_dataset(seed=5, n=4000, cardinality=8)
    booster = _train(X, y, cat_idx, "binary", rounds=50)
    _assert_has_categorical_split(booster)

    Xtest = X[:100]
    ground_truth = booster.predict(Xtest, pred_contrib=True)[:, :-1]
    explainer = fasttreeshap.TreeExplainer(
        booster, feature_perturbation="tree_path_dependent", algorithm="v1"
    )
    sv = explainer.shap_values(Xtest, check_additivity=True)
    # binary LightGBM returns a single contribution matrix; FastTreeSHAP returns
    # [negative_class, positive_class]
    sv_pos = np.asarray(sv[1] if isinstance(sv, list) else sv)
    assert np.abs(sv_pos - ground_truth).max() < 1e-6


def test_categorical_interventional_is_additive():
    # The interventional (feature-independent) path uses a categorical-aware
    # traversal too; verify it is self-consistent (SHAP values sum to the model
    # output). NOTE: feature_perturbation="global_path_dependent" is *not*
    # covered here because it has a pre-existing additivity issue in
    # FastTreeSHAP that also affects purely numeric models (the merged-tree
    # expectation can be NaN); it is unrelated to categorical support.
    X, y, cat_idx, _ = _make_dataset(seed=9, n=3000, cardinality=6)
    booster = _train(X, y, cat_idx, "regression", rounds=40)
    _assert_has_categorical_split(booster)

    background = X[:60]
    Xtest = X[:40]
    pred = booster.predict(Xtest)
    explainer = fasttreeshap.TreeExplainer(
        booster, data=background, feature_perturbation="interventional"
    )
    sv = np.asarray(explainer.shap_values(Xtest, check_additivity=False))
    err = np.abs(sv.sum(1) + explainer.expected_value - pred).max()
    assert err < 1e-4, err
