"""
Leave-One-Region-Out (LORO) Spatial Generalization Validation.

Evaluates how well the forest fire prediction model generalizes to unseen geographical
regions. Uses LeaveOneGroupOut across all 10 regions, training on 9 and testing on 1.
Region is excluded from the feature set to evaluate pure spatial transferability.
Results are saved to outputs/loro_results.csv and printed sorted by Accuracy.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def run_loro():
    data_path = Path("dataset/processed/forest_fires_cleaned.csv")
    output_path = Path("outputs/loro_results.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Leave-One-Region-Out (LORO) Cross-Validation")
    print("=" * 70)
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # 1. Derive month feature from DATE
    df["DATE"] = pd.to_datetime(df["DATE"])
    df["month"] = df["DATE"].dt.month

    # 2. Define groups based on the 10 geographical region-year groups
    groups = df["Region"]
    unique_regions = sorted(groups.unique())
    print(f"Total observations: {len(df)}")
    print(f"Number of geographical regions: {len(unique_regions)}")
    print(f"Regions: {unique_regions}")

    # 3. Define Features (X) and Target (y)
    # Exclude Region from training features to test spatial transferability
    # Also exclude target (Classes), leaky index (FWI), and raw string (DATE)
    drop_cols = ["Classes", "FWI", "DATE", "Region"]
    X = df.drop(columns=drop_cols)
    y = df["Classes"].astype(int)

    feature_names = X.columns.tolist()
    print(f"\nTraining features ({len(feature_names)} features, Region excluded): {feature_names}")

    # 4. Pipeline with StandardScaler and XGBClassifier
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss")),
    ])

    # 5. LeaveOneGroupOut Cross-Validation
    logo = LeaveOneGroupOut()
    loro_results = []

    print("\nRunning Leave-One-Region-Out validation across all 10 regions...")

    for train_idx, test_idx in logo.split(X, y, groups=groups):
        test_region = groups.iloc[test_idx].iloc[0]
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        # Fit model on 9 regions
        pipeline.fit(X_train, y_train)

        # Predict on the held-out test region
        y_pred = pipeline.predict(X_test)

        # Robust metric calculation: handle single-class edge cases gracefully
        acc = float(accuracy_score(y_test, y_pred))

        unique_classes = np.unique(y_test)
        if len(unique_classes) < 2:
            # Single-class test set: warn and compute F1 with zero_division
            print(f"  [Warning] Region {test_region} contains only class {unique_classes[0]} in ground truth.")
            f1 = float(f1_score(y_test, y_pred, zero_division=0))
        else:
            f1 = float(f1_score(y_test, y_pred, zero_division=0))

        loro_results.append({
            "Region": test_region,
            "Accuracy": acc,
            "F1": f1,
        })
        print(f"  Tested on {test_region:<20} (n={len(test_idx):3d}) -> Accuracy: {acc:.4f}, F1: {f1:.4f}")

    # 6. Create results DataFrame and sort by Accuracy ascending
    results_df = pd.DataFrame(loro_results)
    sorted_df = results_df.sort_values(by="Accuracy", ascending=True).reset_index(drop=True)

    # 7. Save results to outputs/loro_results.csv with columns: Region, Accuracy, F1
    sorted_df.to_csv(output_path, index=False)
    print(f"\n[OK] Results saved to: {output_path.resolve()}")

    # 8. Print sorted table and identify the two weakest regions
    print("\n" + "=" * 70)
    print("Leave-One-Region-Out (LORO) Results - Sorted by Accuracy")
    print("=" * 70)
    print(sorted_df.to_string(index=True))
    print("=" * 70)

    weakest_1 = sorted_df.iloc[0]
    weakest_2 = sorted_df.iloc[1]
    print(f"\nTwo Weakest Generalization Regions:")
    print(f"  1. {weakest_1['Region']} -> Accuracy: {weakest_1['Accuracy']:.4f}, F1: {weakest_1['F1']:.4f}")
    print(f"  2. {weakest_2['Region']} -> Accuracy: {weakest_2['Accuracy']:.4f}, F1: {weakest_2['F1']:.4f}")
    print(f"\nMean Accuracy across all regions: {sorted_df['Accuracy'].mean():.4f} +/- {sorted_df['Accuracy'].std():.4f}")
    print(f"Mean F1 across all regions:       {sorted_df['F1'].mean():.4f} +/- {sorted_df['F1'].std():.4f}")
    print("=" * 70)


if __name__ == "__main__":
    run_loro()
