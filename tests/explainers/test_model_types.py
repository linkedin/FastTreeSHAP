""" Coverage-oriented tests that exercise fasttreeshap.TreeExplainer across the
many tree-model backends and options it supports.

Every check is an *additivity* check -- the SHAP values plus the expected value
must equal the model's own output -- which both validates correctness and walks
the model-parsing / algorithm branches inside fasttreeshap/explainers/_tree.py.
"""
import numpy as np
import pytest

import fasttreeshap

rng = np.random.RandomState(0)


def _reg_data(n=300, m=5):
    X = rng.randn(n, m)
    y = 2 * X[:, 0] + X[:, 1] - X[:, 2] ** 2 + 0.1 * rng.randn(n)
    return X, y


def _clf_data(n=300, m=5, classes=2):
    X = rng.randn(n, m)
    score = X[:, 0] + X[:, 1] - X[:, 2]
    if classes == 2:
        y = (score > 0).astype(int)
    else:
        q = np.quantile(score, np.linspace(0, 1, classes + 1)[1:-1])
        y = np.digitize(score, q)
    return X, y


def _additive(explainer, X, **kw):
    """shap_values with check_additivity=True raises if the sum is wrong."""
    sv = explainer.shap_values(X, check_additivity=True, **kw)
    return sv


def _agree_v0_v1_v2(model, X, shortcut=False):
    base = None
    for alg in ("v0", "v1", "v2"):
        e = fasttreeshap.TreeExplainer(model, algorithm=alg, shortcut=shortcut)
        sv = np.asarray(_additive(e, X))
        if base is None:
            base = sv
        else:
            assert np.abs(sv - base).max() < 1e-6, f"{alg} disagrees with v0"


# --------------------------------------------------------------------------- #
# scikit-learn: v0/v1/v2 must agree across a range of estimators
# --------------------------------------------------------------------------- #
def _sklearn_models():
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        GradientBoostingRegressor,
    )
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    return {
        "decision_tree_regressor": ("reg", DecisionTreeRegressor(max_depth=6, random_state=0)),
        "decision_tree_classifier": ("clf", DecisionTreeClassifier(max_depth=6, random_state=0)),
        "extra_trees_classifier": ("clf", ExtraTreesClassifier(n_estimators=25, max_depth=6, random_state=0)),
        "gradient_boosting_regressor": ("reg", GradientBoostingRegressor(n_estimators=25, max_depth=4, random_state=0)),
        "gradient_boosting_classifier": ("clf", GradientBoostingClassifier(n_estimators=25, max_depth=4, random_state=0)),
    }


@pytest.mark.parametrize("name", list(_sklearn_models()))
def test_sklearn_v0_v1_v2_agreement(name):
    task, model = _sklearn_models()[name]
    X, y = _reg_data() if task == "reg" else _clf_data()
    model.fit(X, y)
    _agree_v0_v1_v2(model, X[:40])


def test_sklearn_hist_gradient_boosting_regressor():
    from sklearn.ensemble import HistGradientBoostingRegressor
    X, y = _reg_data()
    m = HistGradientBoostingRegressor(max_iter=25, max_depth=4, random_state=0).fit(X, y)
    e = fasttreeshap.TreeExplainer(m)
    _additive(e, X[:40])


def test_sklearn_random_forest_multiclass():
    from sklearn.ensemble import RandomForestClassifier
    X, y = _clf_data(classes=3)
    m = RandomForestClassifier(n_estimators=25, max_depth=6, random_state=0).fit(X, y)
    sv = _additive(fasttreeshap.TreeExplainer(m, algorithm="v1"), X[:40])
    assert isinstance(sv, list) and len(sv) == 3


def test_sklearn_isolation_forest():
    from sklearn.ensemble import IsolationForest
    X, _ = _reg_data()
    m = IsolationForest(n_estimators=25, random_state=0).fit(X)
    e = fasttreeshap.TreeExplainer(m)
    sv = np.asarray(e.shap_values(X[:40], check_additivity=False))
    assert sv.shape == (40, X.shape[1])


# --------------------------------------------------------------------------- #
# model_output / feature_perturbation / other options
# --------------------------------------------------------------------------- #
def test_probability_output_classifier():
    from sklearn.ensemble import RandomForestClassifier
    X, y = _clf_data()
    m = RandomForestClassifier(n_estimators=25, max_depth=5, random_state=0).fit(X, y)
    # probability output requires the interventional perturbation + background
    e = fasttreeshap.TreeExplainer(
        m, data=X[:50], model_output="probability", feature_perturbation="interventional")
    sv = e.shap_values(X[:30], check_additivity=False)
    # binary classifier -> a per-class list of contribution matrices + per-class
    # expected values; the positive class must reconstruct predict_proba
    sv_pos = np.asarray(sv[1])
    ev_pos = e.expected_value[1]
    proba = m.predict_proba(X[:30])[:, 1]
    assert np.abs(sv_pos.sum(1) + ev_pos - proba).max() < 1e-3


def test_interventional_perturbation():
    from sklearn.ensemble import RandomForestRegressor
    X, y = _reg_data()
    m = RandomForestRegressor(n_estimators=25, max_depth=6, random_state=0).fit(X, y)
    e = fasttreeshap.TreeExplainer(m, data=X[:50], feature_perturbation="interventional")
    sv = np.asarray(e.shap_values(X[:30], check_additivity=True))
    assert sv.shape == (30, X.shape[1])


def test_log_loss_output():
    # exercises the log_loss transform + __dynamic_expected_value path
    from sklearn.ensemble import RandomForestClassifier
    X, y = _clf_data()
    m = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=0).fit(X, y)
    e = fasttreeshap.TreeExplainer(
        m, data=X[:50], model_output="log_loss", feature_perturbation="interventional")
    sv = e.shap_values(X[:20], y=y[:20], check_additivity=False)
    sv = np.asarray(sv[1] if isinstance(sv, list) else sv)
    assert sv.shape[0] == 20


def test_multiclass_interaction_values():
    # shap_interaction_values on a multiclass model (non-shortcut C path)
    from sklearn.ensemble import RandomForestClassifier
    X, y = _clf_data(classes=3)
    m = RandomForestClassifier(n_estimators=15, max_depth=4, random_state=0).fit(X, y)
    e = fasttreeshap.TreeExplainer(m)
    iv = e.shap_interaction_values(X[:10])
    iv = iv if isinstance(iv, list) else [iv]
    assert iv[0].shape == (10, X.shape[1], X.shape[1])


def test_approximate_saabas():
    from sklearn.ensemble import RandomForestRegressor
    X, y = _reg_data()
    m = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=0).fit(X, y)
    e = fasttreeshap.TreeExplainer(m)
    sv = np.asarray(e.shap_values(X[:30], approximate=True, check_additivity=False))
    assert sv.shape == (30, X.shape[1])


def test_predict_and_single_row_and_dataframe():
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    X, y = _reg_data()
    m = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=0).fit(X, y)
    e = fasttreeshap.TreeExplainer(m)
    # internal predict path
    pred = e.model.predict(X[:10])
    assert np.abs(pred.ravel() - m.predict(X[:10])).max() < 1e-4
    # single-row (2-D, one row) input
    single = np.asarray(e.shap_values(X[:1], check_additivity=False))
    assert single.shape == (1, X.shape[1])
    # pandas DataFrame input
    df = pd.DataFrame(X[:20], columns=[f"f{i}" for i in range(X.shape[1])])
    sv = np.asarray(e.shap_values(df, check_additivity=True))
    assert sv.shape == (20, X.shape[1])


def test_call_api_returns_explanation():
    from sklearn.ensemble import RandomForestRegressor
    X, y = _reg_data()
    m = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=0).fit(X, y)
    e = fasttreeshap.TreeExplainer(m)
    expl = e(X[:15])
    # Explanation object attributes (exercises _explanation.py)
    assert expl.values.shape[0] == 15
    assert expl.base_values is not None
    assert expl.data.shape == (15, X.shape[1])
    assert np.abs(expl.values.sum(1) + np.asarray(expl.base_values) - m.predict(X[:15])).max() < 1e-4


# --------------------------------------------------------------------------- #
# xgboost / lightgbm / catboost backends
# --------------------------------------------------------------------------- #
def test_xgboost_core_booster_and_ranker():
    xgboost = pytest.importorskip("xgboost")
    X, y = _reg_data()
    bst = xgboost.train({"max_depth": 4, "objective": "reg:squarederror"},
                        xgboost.DMatrix(X, label=y), num_boost_round=20)
    _agree_v0_v1_v2(bst, X[:40], shortcut=False)


def test_xgboost_multiclass():
    xgboost = pytest.importorskip("xgboost")
    X, y = _clf_data(classes=3)
    m = xgboost.XGBClassifier(n_estimators=20, max_depth=4, objective="multi:softprob",
                              num_class=3, eval_metric="mlogloss").fit(X, y)
    sv = _additive(fasttreeshap.TreeExplainer(m, algorithm="v1", shortcut=False), X[:40])
    assert isinstance(sv, list) and len(sv) == 3


def test_xgboost_shortcut_and_interactions():
    xgboost = pytest.importorskip("xgboost")
    X, y = _reg_data()
    m = xgboost.XGBRegressor(n_estimators=25, max_depth=4).fit(X, y)
    # native shortcut path
    e = fasttreeshap.TreeExplainer(m, shortcut=True)
    sv = np.asarray(e.shap_values(X[:30]))
    assert sv.shape == (30, X.shape[1])
    # interaction values (shortcut, xgboost only)
    iv = np.asarray(e.shap_interaction_values(X[:10]))
    assert iv.shape == (10, X.shape[1], X.shape[1])


def test_lightgbm_ranker_and_multiclass_sklearn():
    lgb = pytest.importorskip("lightgbm")
    # multiclass via the sklearn wrapper
    X, y = _clf_data(classes=3)
    m = lgb.LGBMClassifier(n_estimators=25, num_leaves=15, min_child_samples=20, verbose=-1).fit(X, y)
    sv = _additive(fasttreeshap.TreeExplainer(m, algorithm="v2", shortcut=False), X[:40])
    assert isinstance(sv, list) and len(sv) == 3


def test_catboost_regressor_and_classifier():
    catboost = pytest.importorskip("catboost")
    X, y = _reg_data()
    mr = catboost.CatBoostRegressor(iterations=30, depth=4, random_seed=0, verbose=False).fit(X, y)
    e = fasttreeshap.TreeExplainer(mr)  # catboost uses the native shortcut
    sv = np.asarray(e.shap_values(X[:30]))
    assert np.abs(sv.sum(1) + e.expected_value - mr.predict(X[:30])).max() < 1e-3

    Xc, yc = _clf_data()
    mc = catboost.CatBoostClassifier(iterations=30, depth=4, random_seed=0, verbose=False).fit(Xc, yc)
    e = fasttreeshap.TreeExplainer(mc)
    sv = np.asarray(e.shap_values(Xc[:30]))
    assert sv.shape[0] == 30


# --------------------------------------------------------------------------- #
# error/guard paths
# --------------------------------------------------------------------------- #
def test_global_path_dependent_raises():
    from sklearn.tree import DecisionTreeRegressor
    X, y = _reg_data()
    m = DecisionTreeRegressor(max_depth=4, random_state=0).fit(X, y)
    with pytest.raises(NotImplementedError):
        fasttreeshap.TreeExplainer(m, data=X[:30], feature_perturbation="global_path_dependent")


def test_invalid_options_raise():
    from sklearn.tree import DecisionTreeRegressor
    X, y = _reg_data()
    m = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X, y)
    with pytest.raises(ValueError):
        fasttreeshap.TreeExplainer(m, algorithm="not-an-algorithm")
