
import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Paths
APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent

DATA_PATH = PROJECT_DIR / "dataset" / "Algerian_forest_fires_dataset_CLEANED.csv"
MODEL_DIR = APP_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

# 1. Load dataset
print("Loading dataset...")

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

# 2. Prepare features and target
X = df.drop(columns=["Classes", "FWI"])
y = (
    df["Classes"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({"not fire": 0, "fire": 1})
)

if y.isna().any():
    raise ValueError("Unknown or missing class labels in dataset.")

feature_names = X.columns.tolist()

print("Number of features:", len(feature_names))
print("Features:", feature_names)

# 3. Train/test split (same as backend)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# 4. Build the Random Forest pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(random_state=42)),
])

# 5. Hyperparameter tuning (same grid as train_model.py)
param_grid = {
    "clf__n_estimators": [50, 100],
    "clf__max_depth": [4, 6, None],
    "clf__min_samples_split": [2, 4],
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

print("Training Random Forest...")

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring="f1",
    n_jobs=-1,
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# 6. Evaluate
accuracy = best_model.score(X_test, y_test)

print("Best parameters:", grid_search.best_params_)
print("Best CV F1:", grid_search.best_score_)
print("Test accuracy:", accuracy)

# 7. Save model
model_path = MODEL_DIR / "fire_classifier.joblib"

joblib.dump(best_model, model_path)

# 8. Save metadata
metadata = {
    "features": feature_names,
    "target_map": {
        "0": "Not Fire",
        "1": "Fire",
    },
}

with open(MODEL_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("\nMODEL SAVED SUCCESSFULLY!")
print("Model:", model_path)
print("Metadata:", MODEL_DIR / "metadata.json")
