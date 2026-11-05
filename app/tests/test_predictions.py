"""Run with:  pytest -q   (from the project root)"""
import warnings
import sys
from pathlib import Path

import pytest
from sklearn.metrics import accuracy_score, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import core  # noqa: E402

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def env():
    clf, reg = core.load_models()
    df = core.load_data()
    return clf, reg, df, core.full_data_scaler(df)


def test_models_expect_five_features(env):
    clf, reg, _, _ = env
    assert reg.n_features_in_ == 5 == len(core.FEATURES)


def test_known_fire_row(env):
    clf, reg, df, sc = env
    row = df[df["Classes"] == 1].sort_values("FFMC").iloc[-1]   # clear-cut fire day (highest FFMC)
    label, p, fwi = core.predict({f: float(row[f]) for f in core.FEATURES}, clf, reg, sc)
    assert label == 1 and p > 0.5 and fwi > 0


def test_known_non_fire_row(env):
    clf, reg, df, sc = env
    row = df[df["Classes"] == 0].sort_values("FFMC").iloc[0]    # clear-cut non-fire day (lowest FFMC)
    label, p, fwi = core.predict({f: float(row[f]) for f in core.FEATURES}, clf, reg, sc)
    assert label == 0 and p < 0.5


def test_overall_accuracy_on_full_data(env):
    clf, _, df, sc = env
    acc = (clf.predict(sc.transform(df[core.FEATURES])) == df["Classes"]).mean()
    assert acc > 0.95   # 234/243 correct; misses are borderline FFMC ~80-83 rows


def test_notebook_holdout_metrics_reproduced(env):
    clf, reg, df, _ = env
    y, p = core.classification_holdout(df, clf)
    assert accuracy_score(y, p) == pytest.approx(0.9589, abs=1e-3)   # notebook: 0.96
    y, p = core.regression_holdout(df, reg)
    assert r2_score(y, p) == pytest.approx(0.9787, abs=1e-3)          # notebook: 0.9787


def test_validation_rejects_bad_input():
    good = {"Temperature": 30, "Ws": 15, "FFMC": 80, "DMC": 10, "ISI": 5}
    assert core.validate_inputs(good) == []
    assert core.validate_inputs({**good, "FFMC": 150})          # above physical limit
    assert core.validate_inputs({**good, "Ws": -3})             # negative wind
    assert core.validate_inputs({**good, "ISI": None})          # missing


def test_out_of_range_detection(env):
    _, _, df, _ = env
    r = core.data_ranges(df)
    vals = {"Temperature": 50, "Ws": 15, "FFMC": 80, "DMC": 10, "ISI": 5}
    assert core.out_of_training_range(vals, r) == ["Temperature"]
