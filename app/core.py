"""Shared model / data helpers for the Forest Fire app (used by app.py and tests)."""
import bz2
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).parent
FEATURES = ["Temperature", "Ws", "FFMC", "DMC", "ISI"]  # order the models were trained on
FEATURE_LABELS = {
    "Temperature": "Temperature (°C)",
    "Ws": "Wind speed (km/h)",
    "FFMC": "Fine Fuel Moisture Code (FFMC)",
    "DMC": "Duff Moisture Code (DMC)",
    "ISI": "Initial Spread Index (ISI)",
}
# Hard physical limits for the input form (the training range is narrower; see data_ranges()).
HARD_LIMITS = {
    "Temperature": (-10.0, 55.0),
    "Ws": (0.0, 100.0),
    "FFMC": (0.0, 101.0),
    "DMC": (0.0, 400.0),
    "ISI": (0.0, 60.0),
}
HIGH_HAZARD_FWI = 15  # threshold used in the original app


def load_models():
    clf = pickle.load(bz2.BZ2File(BASE / "models" / "classification.pkl", "rb"))
    reg = pickle.load(bz2.BZ2File(BASE / "models" / "regression.pkl", "rb"))
    return clf, reg


def load_data() -> pd.DataFrame:
    """Cleaned dataset with the same preprocessing as the training notebook."""
    df = pd.read_csv(BASE / "data" / "Algerian_forest_fires_dataset_CLEANED.csv")
    df["Classes"] = np.where(df["Classes"] == "not fire", 0, 1)
    return df


def full_data_scaler(df: pd.DataFrame) -> StandardScaler:
    """Scaler fitted on all rows of the 5 features, exactly as the original Flask app.py did."""
    return StandardScaler().fit(df[FEATURES])


def data_ranges(df: pd.DataFrame) -> dict:
    return {f: (float(df[f].min()), float(df[f].max())) for f in FEATURES}


def validate_inputs(values: dict) -> list:
    """Return a list of error strings (empty list = valid)."""
    errors = []
    for f in FEATURES:
        v = values.get(f)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            errors.append(f"{FEATURE_LABELS[f]} is missing.")
            continue
        lo, hi = HARD_LIMITS[f]
        if not (lo <= v <= hi):
            errors.append(f"{FEATURE_LABELS[f]} must be between {lo:g} and {hi:g}.")
    return errors


def out_of_training_range(values: dict, ranges: dict) -> list:
    return [f for f in FEATURES if not (ranges[f][0] <= values[f] <= ranges[f][1])]


def predict(values: dict, clf, reg, scaler):
    x = scaler.transform(pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES))
    proba_fire = float(clf.predict_proba(x)[0][1])
    label = int(clf.predict(x)[0])
    fwi = float(reg.predict(x)[0])
    return label, proba_fire, fwi


def classification_holdout(df, clf):
    """Re-create the notebook's classification test split (test_size=0.3, random_state=36)."""
    X = df.drop("Classes", axis=1)
    y = df["Classes"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=36)
    sc = StandardScaler().fit(Xtr[FEATURES])
    pred = clf.predict(sc.transform(Xte[FEATURES]))
    return yte.to_numpy(), pred


def regression_holdout(df, reg):
    """Re-create the notebook's regression test split (test_size=0.25, random_state=0)."""
    X = df.drop("FWI", axis=1)
    y = df["FWI"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
    sc = StandardScaler().fit(Xtr[FEATURES])
    pred = reg.predict(sc.transform(Xte[FEATURES]))
    return yte.to_numpy(), pred
