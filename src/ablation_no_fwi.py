"""
Ablation Study: Model Performance Without FWI Component Features.

Evaluates XGBoost using only basic meteorological features (Temperature, RH, Ws, Rain),
plus Region and month, excluding all Canadian FWI indices (FFMC, DMC, DC, ISI, BUI, FWI).
Uses 5-Fold TimeSeriesSplit after sorting chronologically by DATE to prevent temporal leakage.
Saves metrics to outputs/ablation_no_fwi_metrics.csv.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def run_ablation():
    data_path = Path("dataset/processed/forest_fires_cleaned.csv")
    output_path = Path("outputs/ablation_no_fwi_metrics.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Ablation Study: XGBoost without FWI Indices (Meteorological Only)")
    print("=" * 70)
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # 1. Chronological sorting by DATE to prevent temporal leakage
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.sort_values(by=["DATE", "Region"]).reset_index(drop=True)

    # 2. Extract month feature from DATE
    df["month"] = df["DATE"].dt.month

    # 3. Deterministically encode Region to integer for StandardScaler
    region_map = {reg: idx for idx, reg in enumerate(sorted(df["Region"].unique()))}
    df["Region"] = df["Region"].map(region_map)

    # 4. Target variable
    y = df["Classes"].astype(int)

    # 5. Drop specified columns: FFMC, DMC, DC, ISI, BUI, FWI, Classes, DATE
    # Remaining features: Temperature, RH, Ws, Rain, Region, month
    drop_cols = ["FFMC", "DMC", "DC", "ISI", "BUI", "FWI", "Classes", "DATE"]
    X = df.drop(columns=drop_cols)

    # Ensure feature order: Temperature, RH, Ws, Rain, Region, month
    feature_order = ["Temperature", "RH", "Ws", "Rain", "Region", "month"]
    X = X[feature_order]

    print(f"Dataset shape: {df.shape}")
    print(f"Features used ({len(feature_order)}): {feature_order}")
    print(f"Dropped features: {drop_cols}")
    print(f"Target distribution: {y.value_counts().to_dict()}")

    # 6. Pipeline with StandardScaler and XGBClassifier
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss")),
    ])

    # 7. Evaluate using 5-Fold TimeSeriesSplit
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_metrics = []
    print(f"\nRunning {n_splits}-Fold TimeSeriesSplit Cross-Validation...")

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))

        fold_metrics.append({
            "Fold": f"Fold {fold + 1}",
            "Accuracy": acc,
            "F1-Score": f1,
        })
        print(f"  Fold {fold + 1} (Train n={len(train_idx)}, Test n={len(test_idx)}): Accuracy = {acc:.4f}, F1-Score = {f1:.4f}")

    metrics_df = pd.DataFrame(fold_metrics)
    mean_acc = float(metrics_df["Accuracy"].mean())
    std_acc = float(metrics_df["Accuracy"].std())
    mean_f1 = float(metrics_df["F1-Score"].mean())
    std_f1 = float(metrics_df["F1-Score"].std())

    # Append summary rows
    summary_rows = pd.DataFrame([
        {"Fold": "Mean", "Accuracy": mean_acc, "F1-Score": mean_f1},
        {"Fold": "Std", "Accuracy": std_acc, "F1-Score": std_f1},
    ])
    final_df = pd.concat([metrics_df, summary_rows], ignore_index=True)

    # Save to outputs/ablation_no_fwi_metrics.csv
    final_df.to_csv(output_path, index=False)
    print(f"\nSaved metrics to: {output_path.resolve()}")

    print("\n" + "=" * 70)
    print("Ablation Study Results Table")
    print("=" * 70)
    print(final_df.to_string(index=False))
    print("=" * 70)
    print(f"Mean Accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")
    print(f"Mean F1-Score: {mean_f1:.4f} +/- {std_f1:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    run_ablation()
