""" Regression tests for model-parsing fixes (self-contained: depend only on
fasttreeshap + the relevant model library).

Covers:
* XGBoost >= 2.x support via the JSON loader (the old binary ``save_raw`` parser
  broke with a UnicodeDecodeError).
* Multiclass raw LightGBM ``Booster`` (``num_class`` was previously ignored, so
  the per-class trees were summed into a single wrong output).
* ``feature_perturbation="global_path_dependent"`` failing loudly instead of
  returning NaN / incorrect values.

Ground truth is each library's own TreeSHAP (``pred_contribs`` / ``pred_contrib``).
"""
import numpy as np
import pytest

import fasttreeshap


def _agree(a, b):
    return np.abs(np.asarray(a) - np.asarray(b)).max()


class TestXGBoost:
    xgboost = pytest.importorskip("xgboost")

    def _data(self, seed=0, n=800, m=5):
        rng = np.random.RandomState(seed)
        X = rng.randn(n, m)
        y = 2 * X[:, 0] + X[:, 1] - X[:, 2] ** 2
        return X, y

    @pytest.mark.parametrize("algorithm", ["v0", "v1", "v2"])
    def test_regressor(self, algorithm):
        import xgboost
        X, y = self._data()
        model = xgboost.XGBRegressor(n_estimators=30, max_depth=4).fit(X, y)
        gt = model.get_booster().predict(xgboost.DMatrix(X[:80]), pred_contribs=True)
        expl = fasttreeshap.TreeExplainer(model, algorithm=algorithm, shortcut=False)
        sv = np.asarray(expl.shap_values(X[:80], check_additivity=True))
        # JSON dump carries float32 precision, so ~1e-5 is the expected tolerance
        assert _agree(sv, gt[:, :-1]) < 1e-4

    def test_binary_classifier(self):
        import xgboost
        X, y = self._data()
        yb = (y > np.median(y)).astype(int)
        model = xgboost.XGBClassifier(n_estimators=30, max_depth=4, eval_metric="logloss").fit(X, yb)
        gt = model.get_booster().predict(xgboost.DMatrix(X[:80]), pred_contribs=True)
        expl = fasttreeshap.TreeExplainer(model, algorithm="v1", shortcut=False)
        sv = expl.shap_values(X[:80], check_additivity=True)
        sv_pos = np.asarray(sv[1] if isinstance(sv, list) else sv)
        assert _agree(sv_pos, gt[:, :-1]) < 1e-4

    def test_multiclass_classifier(self):
        import xgboost
        X, y = self._data()
        ym = np.clip((y - y.min()) / (y.max() - y.min()) * 3, 0, 2.999).astype(int)
        model = xgboost.XGBClassifier(
            n_estimators=25, max_depth=4, objective="multi:softprob",
            num_class=3, eval_metric="mlogloss",
        ).fit(X, ym)
        gt = model.get_booster().predict(xgboost.DMatrix(X[:80]), pred_contribs=True)
        expl = fasttreeshap.TreeExplainer(model, algorithm="v1", shortcut=False)
        sv = expl.shap_values(X[:80], check_additivity=True)
        assert expl.model.num_stacked_models == 3
        assert max(_agree(sv[c], gt[:, c, :-1]) for c in range(3)) < 1e-4

    def test_core_booster(self):
        import xgboost
        X, y = self._data(seed=1)
        bst = xgboost.train(
            {"max_depth": 4, "objective": "reg:squarederror"},
            xgboost.DMatrix(X, label=y), num_boost_round=20,
        )
        gt = bst.predict(xgboost.DMatrix(X[:60]), pred_contribs=True)
        expl = fasttreeshap.TreeExplainer(bst, algorithm="v1", shortcut=False)
        sv = np.asarray(expl.shap_values(X[:60], check_additivity=True))
        assert _agree(sv, gt[:, :-1]) < 1e-4

    def test_boolean_columns(self):
        # Modern pandas get_dummies() returns bool columns; XGBoost then emits
        # boolean-indicator split nodes ("[fN]") in its JSON dump with no
        # split_condition / missing keys. The JSON loader must handle those.
        import pandas as pd
        import xgboost
        rng = np.random.RandomState(0)
        n = 1500
        df = pd.DataFrame({
            "num": rng.randn(n),
            "cat": rng.randint(0, 4, n),
        })
        df = pd.get_dummies(df, columns=["cat"])  # bool indicator columns
        assert any(df[c].dtype == bool for c in df.columns)
        y = df["num"].values * 2 + df.get("cat_1", pd.Series(np.zeros(n))).astype(float).values
        model = xgboost.XGBRegressor(n_estimators=40, max_depth=6).fit(df, y)
        gt = model.get_booster().predict(
            xgboost.DMatrix(df.iloc[:100]), pred_contribs=True
        )
        expl = fasttreeshap.TreeExplainer(model, algorithm="v1", shortcut=False)
        sv = np.asarray(expl.shap_values(df.iloc[:100].astype(float).values, check_additivity=True))
        assert _agree(sv, gt[:, :-1]) < 1e-4


class TestLightGBMMulticlassBooster:
    lgb = pytest.importorskip("lightgbm")

    def test_raw_booster_multiclass(self):
        import lightgbm as lgb
        rng = np.random.RandomState(3)
        n = 5000
        X = rng.randn(n, 3)
        y = X[:, 0] + X[:, 1] - X[:, 2]
        label = np.clip((y - y.min()) / (y.max() - y.min()) * 3, 0, 2.999).astype(int)
        booster = lgb.train(
            dict(objective="multiclass", num_class=3, num_leaves=31,
                 min_data_in_leaf=20, verbose=-1),
            lgb.Dataset(X, label=label, free_raw_data=False), num_boost_round=40,
        )
        expl = fasttreeshap.TreeExplainer(booster, algorithm="v1")
        assert expl.model.num_stacked_models == 3
        sv = expl.shap_values(X[:60], check_additivity=True)
        phi = booster.predict(X[:60], pred_contrib=True).reshape(60, 3, 4)
        assert max(_agree(sv[c], phi[:, c, :-1]) for c in range(3)) < 1e-6


def test_global_path_dependent_raises():
    from sklearn.tree import DecisionTreeRegressor
    X = np.random.RandomState(0).randn(300, 4)
    y = X[:, 0] + X[:, 1]
    model = DecisionTreeRegressor(max_depth=4, random_state=0).fit(X, y)
    with pytest.raises(NotImplementedError):
        fasttreeshap.TreeExplainer(
            model, data=X[:50], feature_perturbation="global_path_dependent"
        )
