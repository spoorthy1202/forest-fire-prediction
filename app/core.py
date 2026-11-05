"""Model/data helpers for the Algerian Forest Fire Streamlit app.

The deployed classifier is the fitted scikit-learn Pipeline produced by
src/train_model.py. Its StandardScaler is already included in the pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parent

# The order must match the feature order used by src/train_model.py.
DEFAULT_FEATURES = [
    "day", "month", "year", "Temperature", "RH", "Ws", "Rain",
    "FFMC", "DMC", "DC", "ISI", "BUI", "Region",
]

FEATURE_LABELS = {
    "day": "Day of month",
    "month": "Month",
    "year": "Year",
    "Temperature": "Temperature (°C)",
    "RH": "Relative humidity (%)",
    "Ws": "Wind speed (km/h)",
    "Rain": "Rainfall (mm)",
    "FFMC": "Fine Fuel Moisture Code (FFMC)",
    "DMC": "Duff Moisture Code (DMC)",
    "DC": "Drought Code (DC)",
    "ISI": "Initial Spread Index (ISI)",
    "BUI": "Build Up Index (BUI)",
    "Region": "Region",
}

REGION_LABELS = {1: "Bejaia", 2: "Sidi Bel-Abbes"}
TARGET_LABELS = {0: "Not Fire", 1: "Fire"}

MODEL_CANDIDATES = [
    BASE / "models" / "fire_classifier.joblib",
    BASE.parent / "models" / "fire_classifier.joblib",
]
METADATA_CANDIDATES = [
    BASE / "models" / "metadata.json",
    BASE / "metadata.json",
    BASE.parent / "models" / "metadata.json",
]
DATA_CANDIDATES = [
    BASE / "data" / "Algerian_forest_fires_dataset_CLEANED.csv",
    BASE / "dataset" / "Algerian_forest_fires_dataset_CLEANED.csv",
    BASE.parent / "dataset" / "Algerian_forest_fires_dataset_CLEANED.csv",
    BASE.parent / "data" / "Algerian_forest_fires_dataset_CLEANED.csv",
]


def _first_existing(paths: list[Path], description: str) -> Path:
    for path in paths:
        if path.is_file():
            return path
    searched = "\n".join(str(p) for p in paths)
    raise FileNotFoundError(
        f"Could not find {description}. Checked these locations:\n{searched}"
    )


def load_metadata() -> dict:
    path = _first_existing(METADATA_CANDIDATES, "models/metadata.json")
    with path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    features = metadata.get("features")
    if not isinstance(features, list) or not features:
        raise ValueError("metadata.json must contain a non-empty 'features' list.")
    if features != DEFAULT_FEATURES:
        raise ValueError(
            "The feature order in metadata.json differs from the expected order "
            "in train_model.py. Check the training script and metadata before "
            "making predictions.\n"
            f"Metadata: {features}\nExpected: {DEFAULT_FEATURES}"
        )
    return metadata


def load_model():
    """Load the fitted Random Forest Pipeline (including its scaler)."""
    model_path = _first_existing(MODEL_CANDIDATES, "models/fire_classifier.joblib")
    return joblib.load(model_path)


def load_data() -> pd.DataFrame:
    """Load the cleaned Algerian forest-fire dataset."""
    path = _first_existing(DATA_CANDIDATES, "Algerian_forest_fires_dataset_CLEANED.csv")
    df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip()

    if "Classes" not in df.columns:
        raise ValueError("The cleaned dataset must contain a 'Classes' column.")

    if not pd.api.types.is_numeric_dtype(df["Classes"]):
        class_values = df["Classes"].astype(str).str.strip().str.lower()
        df["Classes"] = class_values.map({"not fire": 0, "fire": 1})
    else:
        df["Classes"] = pd.to_numeric(df["Classes"], errors="coerce")

    if df["Classes"].isna().any():
        raise ValueError("Some Classes values could not be mapped to 0/1.")

    df["Classes"] = df["Classes"].astype(int)
    for column in DEFAULT_FEATURES:
        if column not in df.columns:
            raise ValueError(f"Dataset is missing required feature: {column}")
    return df


def data_ranges(df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    return {
        feature: (float(df[feature].min()), float(df[feature].max()))
        for feature in DEFAULT_FEATURES
    }


# Broad physical/data-entry limits. The training-range warning is separate.
HARD_LIMITS = {
    "day": (1, 31),
    "month": (1, 12),
    "year": (1900, 2100),
    "Temperature": (-10.0, 60.0),
    "RH": (0.0, 100.0),
    "Ws": (0.0, 150.0),
    "Rain": (0.0, 300.0),
    "FFMC": (0.0, 101.0),
    "DMC": (0.0, 500.0),
    "DC": (0.0, 1000.0),
    "ISI": (0.0, 100.0),
    "BUI": (0.0, 500.0),
    "Region": (1, 2),
}


def validate_inputs(values: dict) -> List[str]:
    errors = []
    for feature in DEFAULT_FEATURES:
        value = values.get(feature)
        if value is None:
            errors.append(f"{FEATURE_LABELS[feature]} is required.")
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            errors.append(f"{FEATURE_LABELS[feature]} must be numeric.")
            continue
        if not np.isfinite(numeric_value):
            errors.append(f"{FEATURE_LABELS[feature]} must be a finite number.")
            continue
        low, high = HARD_LIMITS[feature]
        if not low <= numeric_value <= high:
            errors.append(
                f"{FEATURE_LABELS[feature]} must be between {low:g} and {high:g}."
            )
        if feature in {"day", "month", "year", "Region"} and not numeric_value.is_integer():
            errors.append(f"{FEATURE_LABELS[feature]} must be a whole number.")
    return errors


def out_of_training_range(values: dict, ranges: dict) -> List[str]:
    outside = []
    for feature in DEFAULT_FEATURES:
        low, high = ranges[feature]
        value = float(values[feature])
        if value < low or value > high:
            outside.append(feature)
    return outside


def predict(values: dict, model) -> Tuple[int, float]:
    """Return (class_label, fire_probability) using the saved pipeline."""
    row = pd.DataFrame(
        [[values[feature] for feature in DEFAULT_FEATURES]],
        columns=DEFAULT_FEATURES,
    )
    label = int(model.predict(row)[0])
    probabilities = model.predict_proba(row)[0]
    classes = list(model.classes_)
    fire_index = classes.index(1)
    probability_fire = float(probabilities[fire_index])
    return label, probability_fire


def holdout_predictions(df: pd.DataFrame, model):
    """Recreate train_model.py's stratified 80/20 split (random_state=42)."""
    X = df[DEFAULT_FEATURES]
    y = df["Classes"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
    return y_test.to_numpy(), predictions, probabilities
