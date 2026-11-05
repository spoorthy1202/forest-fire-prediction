# Forest Fire Streamlit integration

## Files
- `app.py`: Streamlit interface
- `core.py`: loads metadata, dataset and fitted model pipeline

## Expected layout
Place these files together in your app folder, with:
- `models/fire_classifier.joblib`
- `models/metadata.json`
- `data/Algerian_forest_fires_dataset_CLEANED.csv`

The loader also checks a few parent-folder `models/`, `data/` and `dataset/` locations.

## Install and run
Use Python 3.10 and install the compatible versions from your team's requirements file.

```powershell
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Important
The app uses the Random Forest scikit-learn Pipeline saved by `src/train_model.py`.
The StandardScaler is already part of that Pipeline. Do not scale the input again.
The app currently implements classification only. The training script supplied does
not train or save an FWI regression model.
