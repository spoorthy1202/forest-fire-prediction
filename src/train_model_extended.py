"""
Machine Learning Training Pipeline for Extended Algerian Forest Fires Dataset.

Key components:
1. Loads cleaned dataset from dataset/processed/forest_fires_cleaned.csv.
2. Chronologically sorts observations by DATE to prevent temporal leakage.
3. Feature engineering: extracts 'month' from DATE, encodes 'Region' deterministically,
   and excludes 'DATE', 'FWI' (leaky proxy), and 'Classes' (target).
4. Validates 4 Scikit-Learn pipelines with StandardScaler using TimeSeriesSplit (5 splits):
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - XGBoost (XGBClassifier)
5. Evaluates Accuracy, F1-Score, Precision, Recall, and ROC-AUC across time-series splits.
6. Selects best model based on mean CV F1-score and trains on the full dataset.
7. Saves:
   - Trained model: models/fire_classifier_extended.joblib
   - Feature metadata: models/metadata_extended.json
   - Metrics comparison: outputs/model_comparison_metrics_extended.csv
   - SHAP summary plot: outputs/shap_summary_extended.png
   - ROC curve plot: outputs/roc_curve_extended.png
   - Confusion matrix plot: outputs/confusion_matrix_extended.png
"""

import json
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def run_pipeline():
    # 0. Ensure target folders exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    print("=" * 70)
    print("Extended Algerian Forest Fires - Model Training Pipeline")
    print("=" * 70)

    # 1. Load Data
    data_path = Path("dataset/processed/forest_fires_cleaned.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {data_path.resolve()}")

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns.")

    # 2. Chronological Ordering for TimeSeriesSplit
    # Crucial for time series validation: strictly sort by DATE to prevent temporal leakage
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.sort_values(by=["DATE", "Region"]).reset_index(drop=True)

    # 3. Feature Engineering: extract month from DATE and encode Region
    df["month"] = df["DATE"].dt.month

    # Deterministic alphabetical mapping for Region
    unique_regions = sorted(df["Region"].unique())
    region_map = {reg: idx for idx, reg in enumerate(unique_regions)}
    df["Region"] = df["Region"].map(region_map)

    # 4. Define Features (X) and Target (y)
    # Features: weather indices, month, region (exclude DATE, FWI, Classes)
    X = df.drop(columns=["Classes", "FWI", "DATE"])
    y = df["Classes"].astype(int)

    feature_names = X.columns.tolist()
    print(f"\nFeature set ({len(feature_names)} features): {feature_names}")
    print(f"Target distribution: {y.value_counts().to_dict()}")

    # 5. Build Pipelines with StandardScaler
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", DecisionTreeClassifier(random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]),
        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss")),
        ]),
    }

    # 6. TimeSeriesSplit Cross-Validation
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scoring = ["accuracy", "f1", "precision", "recall", "roc_auc"]
    results = {}

    print(f"\nEvaluating models using TimeSeriesSplit (n_splits={n_splits})...")
    for name, pipeline in models.items():
        cv_res = cross_validate(pipeline, X, y, cv=tscv, scoring=scoring)
        results[name] = {
            "Accuracy": float(np.mean(cv_res["test_accuracy"])),
            "Accuracy_Std": float(np.std(cv_res["test_accuracy"])),
            "F1-Score": float(np.mean(cv_res["test_f1"])),
            "F1_Std": float(np.std(cv_res["test_f1"])),
            "Precision": float(np.mean(cv_res["test_precision"])),
            "Precision_Std": float(np.std(cv_res["test_precision"])),
            "Recall": float(np.mean(cv_res["test_recall"])),
            "Recall_Std": float(np.std(cv_res["test_recall"])),
            "ROC-AUC": float(np.mean(cv_res["test_roc_auc"])),
            "ROC_AUC_Std": float(np.std(cv_res["test_roc_auc"])),
        }

    # Format and save comparison table
    metrics_df = pd.DataFrame(results).T
    metrics_csv_path = Path("outputs/model_comparison_metrics_extended.csv")
    metrics_df.to_csv(metrics_csv_path)

    print("\n" + "=" * 70)
    print("Time-Series Cross-Validation Model Comparison Table")
    print("=" * 70)
    display_cols = ["Accuracy", "F1-Score", "Precision", "Recall", "ROC-AUC"]
    print(metrics_df[display_cols].map(lambda x: f"{x:.4f}").to_string())
    print("=" * 70)

    # 7. Select Best Model based on Time-Series CV F1-Score
    best_model_name = metrics_df["F1-Score"].idxmax()
    best_f1 = metrics_df.loc[best_model_name, "F1-Score"]
    print(f"\nBest Model: {best_model_name} (Mean CV F1-Score = {best_f1:.4f})")

    # 8. Out-of-Fold Evaluation for Best Model Plots
    best_pipeline = models[best_model_name]
    oof_y_true = []
    oof_y_pred = []
    oof_y_proba = []

    for train_idx, test_idx in tscv.split(X):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]

        best_pipeline.fit(X_tr, y_tr)
        oof_y_true.extend(y_te)
        oof_y_pred.extend(best_pipeline.predict(X_te))
        oof_y_proba.extend(best_pipeline.predict_proba(X_te)[:, 1])

    oof_y_true = np.array(oof_y_true)
    oof_y_pred = np.array(oof_y_pred)
    oof_y_proba = np.array(oof_y_proba)

    oof_auc = roc_auc_score(oof_y_true, oof_y_proba)
    print(f"\nOut-of-Fold Evaluation ({best_model_name}):")
    print(f"Overall OOF ROC-AUC: {oof_auc:.4f}")
    print("\nClassification Report (Out-of-Fold):")
    print(classification_report(oof_y_true, oof_y_pred, target_names=["Not Fire (0)", "Fire (1)"]))

    # 9. Train Best Model on the Full Dataset & Save
    import joblib

    print(f"Fitting final {best_model_name} model on full dataset (1,215 observations)...")
    final_model = models[best_model_name]
    final_model.fit(X, y)

    model_path = Path("models/fire_classifier_extended.joblib")
    joblib.dump(final_model, model_path)
    print(f"[OK] Saved model to: {model_path.resolve()}")

    # 10. Save Metadata
    metadata_path = Path("models/metadata_extended.json")
    metadata = {
        "features": feature_names,
        "target_map": {"0": "Not Fire", "1": "Fire"},
        "region_map": region_map,
        "best_model_name": best_model_name,
        "cv_strategy": f"TimeSeriesSplit(n_splits={n_splits})",
        "metrics": {col: round(float(metrics_df.loc[best_model_name, col]), 4) for col in display_cols},
        "all_model_metrics": metrics_df[display_cols].to_dict(orient="index"),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[OK] Saved metadata to: {metadata_path.resolve()}")

    # 11. Save Plots (ROC Curve, Confusion Matrix)
    # ROC Curve
    fpr, tpr, _ = roc_curve(oof_y_true, oof_y_proba)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#d62728", lw=2, label=f"{best_model_name} (OOF AUC = {oof_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title(f"ROC Curve - {best_model_name} (Time-Series Cross-Validation)", fontsize=12, weight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    roc_path = Path("outputs/roc_curve_extended.png")
    plt.savefig(roc_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[OK] Saved ROC curve to: {roc_path.resolve()}")

    # Confusion Matrix
    cm = confusion_matrix(oof_y_true, oof_y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Fire (0)", "Fire (1)"],
        yticklabels=["Not Fire (0)", "Fire (1)"],
        cbar=True,
    )
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("Actual Class", fontsize=11)
    plt.title(f"Confusion Matrix - {best_model_name}\n(Time-Series CV Out-of-Fold, n=1,010)", fontsize=12, weight="bold")
    cm_path = Path("outputs/confusion_matrix_extended.png")
    plt.savefig(cm_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[OK] Saved Confusion Matrix to: {cm_path.resolve()}")

    # 12. Explainability: SHAP Summary Plot
    print("\nGenerating SHAP summary plot for final model...")
    scaler = final_model.named_steps["scaler"]
    clf = final_model.named_steps["clf"]
    X_scaled = scaler.transform(X)

    if isinstance(clf, (RandomForestClassifier, DecisionTreeClassifier, XGBClassifier)):
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_scaled)
        if isinstance(shap_values, list):
            shap_vals_to_plot = shap_values[1]
        elif len(getattr(shap_values, "shape", ())) == 3:
            shap_vals_to_plot = shap_values[:, :, 1]
        else:
            shap_vals_to_plot = shap_values
    else:
        explainer = shap.LinearExplainer(clf, X_scaled)
        shap_vals_to_plot = explainer.shap_values(X_scaled)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals_to_plot, X_scaled, feature_names=feature_names, show=False)
    shap_path = Path("outputs/shap_summary_extended.png")
    plt.savefig(shap_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[OK] Saved SHAP summary plot to: {shap_path.resolve()}")

    print("\n" + "=" * 70)
    print("Extended Model Training Complete Successfully!")
    print("=" * 70)

    return final_model, metadata, metrics_df


if __name__ == "__main__":
    run_pipeline()
