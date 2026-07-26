""" Regression tests for two production LightGBM issues:

* Bug 1 - FastTreeSHAP v2 ignored ``children_default`` for missing values, so
  ``algorithm="v2"`` disagreed with v0/v1 (and with LightGBM ``pred_contrib``)
  on any row containing NaNs.
* Bug 2 - ``TreeExplainer.shap_values`` crashed on pandas categorical / string
  DataFrames (``could not convert string to float``) because the LightGBM
  category codes were never applied on the Python side.

Ground truth throughout is LightGBM's own TreeSHAP:
``Booster.predict(X, pred_contrib=True)[:, :-1]``. Models are trained WITH
missing values so ``missing_type`` is "NaN" and per-node default directions are
learned (the production scenario).
"""
import os
import tempfile

import numpy as np
import pytest

import fasttreeshap

lgb = pytest.importorskip("lightgbm")
pd = pytest.importorskip("pandas")

ALGORITHMS = ["v0", "v1", "v2"]


def _pos_class(sv):
    return np.asarray(sv[1] if isinstance(sv, list) else sv)


# --------------------------------------------------------------------------- #
# Bug 1 - missing-value routing (numeric NaNs)                                #
# --------------------------------------------------------------------------- #
def _train_numeric_with_nans(seed=0, n=4000):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 6)
    y = (X[:, 0] + X[:, 1] - X[:, 2] ** 2 > 0).astype(int)
    Xtr = X.copy()
    Xtr[rng.rand(*Xtr.shape) < 0.2] = np.nan  # train with NaNs -> missing_type=NaN
    booster = lgb.train(
        dict(objective="binary", num_leaves=31, min_data_in_leaf=20, verbose=-1),
        lgb.Dataset(Xtr, label=y, free_raw_data=False), num_boost_round=50,
    )
    Xtest = X[:200].copy()
    Xtest[rng.rand(*Xtest.shape) < 0.3] = np.nan
    return booster, Xtest


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_numeric_nans_match_pred_contrib(algorithm):
    booster, Xtest = _train_numeric_with_nans()
    gt = np.asarray(booster.predict(Xtest, pred_contrib=True))[:, :-1]
    expl = fasttreeshap.TreeExplainer(booster, algorithm=algorithm, shortcut=False)
    sv = _pos_class(expl.shap_values(Xtest, check_additivity=True))
    assert np.abs(sv - gt).max() < 1e-6


def _train_categorical_with_nans(seed=1, n=4000):
    rng = np.random.RandomState(seed)
    X = np.column_stack([rng.randn(n), rng.randint(0, 6, n).astype(float),
                         rng.randn(n), rng.randn(n)])
    eff = rng.randn(6) * 2
    y = (X[:, 0] + eff[X[:, 1].astype(int)] - X[:, 3] > 0).astype(int)
    Xtr = X.copy()
    for col in (0, 2, 3):  # NaNs only in numeric columns
        Xtr[rng.rand(n) < 0.2, col] = np.nan
    booster = lgb.train(
        dict(objective="binary", num_leaves=31, min_data_in_leaf=20, verbose=-1),
        lgb.Dataset(Xtr, label=y, categorical_feature=[1], free_raw_data=False),
        num_boost_round=60,
    )
    Xtest = X[:200].copy()
    for col in (0, 2, 3):
        Xtest[rng.rand(Xtest.shape[0]) < 0.3, col] = np.nan
    return booster, Xtest


def _has_categorical_split(booster):
    seen = set()

    def walk(node):
        if "decision_type" in node:
            seen.add(node["decision_type"])
            walk(node["left_child"])
            walk(node["right_child"])

    for t in booster.dump_model()["tree_info"]:
        walk(t["tree_structure"])
    return "==" in seen


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_categorical_plus_numeric_nans_match_pred_contrib(algorithm):
    booster, Xtest = _train_categorical_with_nans()
    assert _has_categorical_split(booster)
    gt = np.asarray(booster.predict(Xtest, pred_contrib=True))[:, :-1]
    expl = fasttreeshap.TreeExplainer(booster, algorithm=algorithm, shortcut=False)
    sv = _pos_class(expl.shap_values(Xtest, check_additivity=True))
    assert np.abs(sv - gt).max() < 1e-6


def test_v2_additivity_with_nans():
    # additivity (shap sum + expected == raw model output) must hold for v2 on
    # rows with NaNs -- this failed before the children_default fix.
    booster, Xtest = _train_categorical_with_nans()
    expl = fasttreeshap.TreeExplainer(booster, algorithm="v2", shortcut=False)
    # check_additivity=True raises if the sum does not match the model output
    expl.shap_values(Xtest, check_additivity=True)


# --------------------------------------------------------------------------- #
# Bug 2 - pandas categorical / string DataFrame input                         #
# --------------------------------------------------------------------------- #
def _make_pandas_df(n, seed):
    r = np.random.RandomState(seed)
    df = pd.DataFrame({
        "num1": r.randn(n),
        "company": r.choice(["Private Company", "Public", "NGO", "Gov"], n),
        "num2": r.randn(n),
        "region": r.choice(["NA", "EU", "APAC"], n),
    })
    df["company"] = df["company"].astype("category")
    df["region"] = df["region"].astype("category")
    return df


def _train_pandas_categorical(n=4000):
    df = _make_pandas_df(n, 0)
    y = (df["num1"] + (df["company"].cat.codes == 2).astype(float) * 2 - df["num2"] > 0).astype(int)
    dftr = df.copy()
    dftr.loc[np.random.RandomState(1).rand(n) < 0.2, "num1"] = np.nan
    booster = lgb.train(
        dict(objective="binary", num_leaves=31, min_data_in_leaf=30, verbose=-1),
        lgb.Dataset(dftr, label=y, free_raw_data=False), num_boost_round=60,
    )
    return booster


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_pandas_categorical_dataframe(algorithm):
    booster = _train_pandas_categorical()
    assert _has_categorical_split(booster)
    Xt = _make_pandas_df(200, 99)
    Xt.loc[np.random.RandomState(3).rand(200) < 0.3, "num1"] = np.nan
    gt = np.asarray(booster.predict(Xt, pred_contrib=True))[:, :-1]
    expl = fasttreeshap.TreeExplainer(booster, algorithm=algorithm, shortcut=False)
    # no manual category->code conversion by the caller
    sv = _pos_class(expl.shap_values(Xt, check_additivity=True))
    assert np.abs(sv - gt).max() < 1e-6


def test_pandas_categorical_call_api():
    booster = _train_pandas_categorical()
    Xt = _make_pandas_df(120, 5)
    gt = np.asarray(booster.predict(Xt, pred_contrib=True))[:, :-1]
    expl = fasttreeshap.TreeExplainer(booster, algorithm="v2", shortcut=False)
    values = np.asarray(expl(Xt).values)
    assert np.abs(values - gt).max() < 1e-6


def test_pandas_categorical_booster_from_file():
    booster = _train_pandas_categorical()
    Xt = _make_pandas_df(120, 42)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.txt")
        booster.save_model(path)
        loaded = lgb.Booster(model_file=path)
    gt = np.asarray(loaded.predict(Xt, pred_contrib=True))[:, :-1]
    expl = fasttreeshap.TreeExplainer(loaded, algorithm="v2", shortcut=False)
    sv = _pos_class(expl.shap_values(Xt, check_additivity=True))
    assert np.abs(sv - gt).max() < 1e-6


def test_pandas_categorical_unseen_level():
    booster = _train_pandas_categorical()
    Xu = _make_pandas_df(60, 7)
    Xu["company"] = Xu["company"].astype(str)
    Xu.loc[:5, "company"] = "BrandNewCategory"  # unseen at training time
    Xu["company"] = Xu["company"].astype("category")
    gt = np.asarray(booster.predict(Xu, pred_contrib=True))[:, :-1]
    expl = fasttreeshap.TreeExplainer(booster, algorithm="v2", shortcut=False)
    sv = _pos_class(expl.shap_values(Xu, check_additivity=False))
    assert np.abs(sv - gt).max() < 1e-6
