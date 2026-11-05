import pandas as pd
import numpy as np
import joblib
import json
import os
import shap
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# 0. Ensure folders exist
os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# 1. Load Data
print("Loading data...")
df = pd.read_csv('dataset/Algerian_forest_fires_dataset_CLEANED.csv')
df.columns = df.columns.str.strip()

# 2. Define Features and Target (13 Features)
X = df.drop(columns=['Classes', 'FWI'])
y = df['Classes'].str.strip().str.lower().map({'not fire': 0, 'fire': 1}) 

feature_names = X.columns.tolist()
with open('models/metadata.json', 'w') as f:
    json.dump({"features": feature_names, "target_map": {"0": "Not Fire", "1": "Fire"}}, f)

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Build Pipelines
models = {
    "Logistic Regression": Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, random_state=42))]),
    "Decision Tree": Pipeline([('scaler', StandardScaler()), ('clf', DecisionTreeClassifier(random_state=42))]),
    "Random Forest": Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier(random_state=42))])
}

# 5. Evaluate Models (Save to CSV)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
results = {}

for name, pipeline in models.items():
    cv_results = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring)
    results[name] = {
        'Accuracy': np.mean(cv_results['test_accuracy']),
        'Accuracy_Std': np.std(cv_results['test_accuracy']),
        'F1-Score': np.mean(cv_results['test_f1']),
        'F1_Std': np.std(cv_results['test_f1']),
        'ROC-AUC': np.mean(cv_results['test_roc_auc'])
    }

# Save comparison metrics to CSV for the report
metrics_df = pd.DataFrame(results).T
metrics_df.to_csv('outputs/model_comparison_metrics.csv')
print("\n--- Model Comparison (Saved to CSV) ---")
print(metrics_df[['Accuracy', 'F1-Score', 'ROC-AUC']])

# 6. Hyperparameter Tuning
param_grid = {'clf__n_estimators': [50, 100], 'clf__max_depth': [4, 6, None], 'clf__min_samples_split': [2, 4]}
grid_search = GridSearchCV(models["Random Forest"], param_grid, cv=cv, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Best CV F1-Score: {grid_search.best_score_:.4f}")

# 7. Final Evaluation & Risk Assessment
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("\n--- Final Test Set Evaluation ---")
print(classification_report(y_test, y_pred))

# Risk Assessment Mapping
def get_risk_level(prob):
    if prob < 0.3: return "Low Risk"
    elif prob < 0.7: return "Moderate Risk"
    else: return "High Risk"

# Demonstrate risk assessment on the first 5 test samples
print("\n--- Sample Risk Assessment ---")
for i in range(5):
    prob = y_proba[i]
    print(f"Sample {i+1}: Probability = {prob:.2f} -> {get_risk_level(prob)}")

# 8. Save Plots (ROC, CM, SHAP)
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba):.4f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve - Test Set')
plt.legend()
plt.savefig('outputs/roc_curve.png')
plt.close()

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig('outputs/confusion_matrix.png')
plt.close()

# SHAP
rf_clf = best_model.named_steps['clf']
X_train_scaled = best_model.named_steps['scaler'].transform(X_train)
explainer = shap.TreeExplainer(rf_clf)
shap_values = explainer.shap_values(X_train_scaled)
shap_vals_to_plot = shap_values[:, :, 1] if len(shap_values.shape) == 3 else shap_values[1]
plt.figure()
shap.summary_plot(shap_vals_to_plot, X_train_scaled, feature_names=feature_names, show=False)
plt.savefig('outputs/shap_summary.png', bbox_inches='tight')
plt.close()

# 9. Save Model
joblib.dump(best_model, 'models/fire_classifier.joblib')
print("\nSUCCESS: Model, metrics CSV, and plots saved.")