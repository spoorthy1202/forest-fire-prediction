# 🔥 Algerian Forest Fire Prediction – Streamlit App

A Streamlit dashboard for predicting forest fires from weather and fuel-moisture data.

- **Classification:** fire / not fire (tuned XGBoost)
- **Regression:** Fire Weather Index, FWI (tuned Random Forest)
- **Dataset:** UCI *Algerian Forest Fires* (Bejaia & Sidi Bel-Abbes, Jun–Sep 2012, 243 rows)

> **Credits:** the data cleaning, model training and saved models come from the open-source
> project by **Aravind Selvam** – https://github.com/aravind-selvam/forest-fire-prediction.
> This repository re-packages those models as a Streamlit dashboard (replacing the original
> Flask + MongoDB + Heroku deployment).

## Pages
| Page | What it shows |
|---|---|
| Project Overview | Problem, dataset, pipeline, input features and their training ranges |
| EDA | Class balance, feature distributions, correlation heatmap, month/region patterns |
| Model Performance | Model comparison tables, confusion matrix, predicted-vs-actual FWI, feature importance |
| Fire Prediction | Input form with validation → fire/no-fire result, fire probability, predicted FWI and hazard level |

## Model inputs (order matters)
`Temperature, Ws (wind speed), FFMC, DMC, ISI`

## Project structure
```
app.py                 Streamlit dashboard
core.py                model/data loading, validation, prediction helpers
models/                classification.pkl (XGBoost), regression.pkl (Random Forest)  [bz2 pickles]
data/                  Algerian_forest_fires_dataset_CLEANED.csv
tests/                 pytest tests (prediction sanity, validation, notebook-metric reproduction)
requirements.txt       pinned versions
```

## Setup
The pickled models were created with **scikit-learn 1.0.2 / XGBoost 1.x**, so use
**Python 3.9 or 3.10** and the pinned versions.

```bash
python3.10 -m venv venv
venv\Scripts\activate            # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```
Run the tests: `python -m pytest -q`

## Deploy (Streamlit Community Cloud)
1. Push this folder to a GitHub repository.
2. On share.streamlit.io choose **New app**, select the repo, main file `app.py`.
3. Under **Advanced settings** set **Python version = 3.10**.
4. Deploy.

## How the models are integrated
- The original Flask app re-fitted a `StandardScaler` on the dataset (loaded from MongoDB) at
  startup because the scaler was never saved. Here the same scaler is fitted on the same 5
  columns of the cleaned CSV, so no database is needed.
- On the **Model Performance** page the saved models are re-run on the notebook's exact test
  splits. They reproduce the notebook: classifier 95.9 % accuracy (precision/recall as in the
  notebook), regressor R² = 0.9787, MAE = 0.6205.

## Input validation
- Values are limited to physically sensible ranges (e.g. FFMC 0–101, wind ≥ 0).
- Missing/invalid values show an error and no prediction is made.
- Values outside the training range show a warning.

## Known limitations
- The classifier relies mainly on FFMC (a switch near FFMC ≈ 80), so borderline days can be
  misclassified (9 of 243 rows). Classifier and regressor are independent and may disagree.
- The "FWI > 15 = high hazard" threshold is the original author's choice.
- Tiny dataset (243 days, one season) – results will not generalise to other regions/years.
- The scaler is fitted on all rows, whereas the notebook fitted it on the training split only
  (kept as in the original app; the effect is small).
