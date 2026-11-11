<div align="center">

# Fire-Day Classification from Fire-Weather Indices

### An interactive XGBoost dashboard for forest-fire classification

<p>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/XGBoost-Classifier-F7931E" alt="XGBoost">
  <img src="https://img.shields.io/badge/Status-Academic%20Project-blue" alt="Project status">
</p>

<p><strong>Predict fire-day occurrence using raw weather and Fire Weather Index indicators. Honest, temporally-validated ML.</strong></p>

</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Dataset and Input Features](#dataset-and-input-features)
- [Machine Learning Model](#machine-learning-model)
- [Evaluation & Honest Results](#evaluation--honest-results)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Running the Application](#running-the-application)
- [Running Tests](#running-tests)
- [Limitations](#limitations)
- [Contributors](#contributors)

## Overview

This project is a machine learning system for **same-day fire classification** using the Extended Algerian Forest Fires dataset. Given environmental and fire-weather observations, the model classifies whether the day is associated with a fire event.

The project includes:
- A robust preprocessing pipeline.
- Rigorous evaluation using **TimeSeriesSplit** (temporal) and **Leave-One-Region-Out** (spatial) validation.
- An ablation study comparing FWI indices vs. raw weather.
- An interactive **Streamlit web application** for exploration and live prediction.

The extended dataset contains **1,215 observations** across **10 region-year groups** (2012, 2017, 2021).

> **Note:** This project is for academic and educational use. It is not an operational wildfire warning system.

## Key Features

- **Interactive Dashboard:** Four-page Streamlit interface (Overview, EDA, Model Performance, Prediction).
- **Honest Evaluation:** TimeSeriesSplit and Leave-One-Region-Out validation, no temporal leakage.
- **Ablation Study:** Proves the incremental value of FWI indices over raw weather.
- **Dynamic Metrics:** Model performance page loads real metrics from `outputs/model_comparison_metrics_extended.csv`.
- **Live Prediction:** Enter 11 features and receive a fire/non-fire classification with probability.

## Dataset and Input Features

The project uses the **Extended Algerian Forest Fires dataset** (1,215 observations, 10 regions, 2012/2017/2021, June–September).

The model uses **11 input features**:

| Feature | Description |
|---|---|
| Temperature | Air temperature (°C) |
| RH | Relative humidity (%) |
| Ws | Wind speed (km/h) |
| Rain | Rainfall (mm) |
| FFMC | Fine Fuel Moisture Code |
| DMC | Duff Moisture Code |
| DC | Drought Code |
| ISI | Initial Spread Index |
| BUI | Buildup Index |
| Region | Region-year group identifier (0–9) |
| month | Month (6–9) |

**Excluded (leakage):** `FWI`, `DATE`, `Classes`.

The target is binary:
- `0` — Not Fire
- `1` — Fire

## Machine Learning Model

- **Algorithm:** XGBoost (XGBClassifier)
- **Preprocessing:** StandardScaler wrapped in a Scikit-Learn Pipeline
- **Saved model:** `models/fire_classifier_extended.joblib`
- **Metadata:** `models/metadata_extended.json`
- **Output:** Binary classification + probability

The pipeline guarantees that the scaler is only fit on training folds during cross-validation.

## Evaluation & Honest Results

We used **5-Fold TimeSeriesSplit** (chronological) to prevent temporal leakage. Random splits produce fake 100% accuracy; our honest number is:

| Model | Accuracy | F1-Score | Recall (Fire) | ROC-AUC |
|---|---|---|---|---|
| **XGBoost (final)** | **86.73%** | **86.55%** | **97.61%** | **0.9318** |
| Random Forest | 84.95% | 85.27% | 98.65% | 0.9656 |
| Decision Tree | 84.46% | 84.50% | 96.79% | 0.8452 |
| Logistic Regression | 80.99% | 81.92% | 100.00% | 0.9766 |

**Standard deviations** are sample standard deviations (`ddof=1`, pandas default).

### Ablation Study (Raw Weather Only)

Removing the 5 FWI indices and training on only raw weather (Temperature, RH, Ws, Rain, Region, month) gives:

| Model | Accuracy | F1-Score |
|---|---|---|
| XGBoost (raw weather only) | **78.91% ± 7.40%** | **78.35% ± 11.08%** |

**Conclusion:** The FWI indices add ~8 percentage points, but raw weather alone still beats the 52% majority baseline by a wide margin.

### Leave-One-Region-Out (LORO) Validation

Training on 9 regions and testing on the 10th, repeated 10 times:

| Region | Accuracy | F1 |
|---|---|---|
| Sidi Bel Abass 2012 | 65.57% | 0.6379 |
| Bejaia 2012 | 67.21% | 0.4872 |
| Tipaza 2017 | 84.43% | 0.7397 |
| All 2021 regions | 91–95% | 0.90–0.95 |

**Conclusion:** The model generalizes well to unseen 2021 regions, but fails on 2012 regions. A threshold analysis suggests the 2012 labels use a different FFMC cutoff (~80) than the 2021 labels (~88). This is a **label-definition shift**, not a weather distribution shift.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.11 | Core language |
| XGBoost | Final classifier |
| Scikit-learn | Pipelines, TimeSeriesSplit, metrics |
| Pandas / NumPy | Data manipulation |
| Streamlit | Interactive dashboard |
| Plotly | Interactive charts |
| SHAP | Model explainability |
| Joblib | Model persistence |

## Repository Structure

```text
forest-fire-prediction/
│
├── app/
│   ├── app_forest_background.py  # Streamlit entry point
│   ├── core.py                   # Model loading, feature order, prediction
│   ├── assets/                   # Background images, styling
│   └── styles.css
│
├── dataset/
│   ├── The extended Algerian forest fires dataset.xlsx  # Raw
│   └── processed/
│       └── forest_fires_cleaned.csv                     # Cleaned (1,215 rows)
│
├── models/
│   ├── fire_classifier_extended.joblib   # Trained XGBoost pipeline
│   └── metadata_extended.json            # Feature order, metrics, versions
│
├── src/
│   ├── preprocess.py            # Data cleaning
│   ├── train_model_extended.py  # Training (XGBoost, TimeSeriesSplit)
│   ├── ablation_no_fwi.py       # Ablation study
│   ├── loro_validation.py       # Leave-One-Region-Out
│   └── eda.py                   # EDA plots
│
├── outputs/
│   ├── model_comparison_metrics_extended.csv
│   ├── ablation_no_fwi_metrics.csv
│   ├── loro_results.csv
│   ├── roc_curve_extended.png
│   ├── confusion_matrix_extended.png
│   ├── shap_summary_extended.png
│   └── figures/                 # EDA plots
│
├── tests/
│   └── test_preprocess.py       # 11 passing tests
│
├── requirements.txt
└── README.md
