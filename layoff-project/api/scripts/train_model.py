# scripts/train_model.py

import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from api.preprocess.preprocess_pipeline import build_preprocessing_pipeline
import os
from api.scripts.performance_artifacts import (
    save_performance_artifacts,
)

# ----------------------------------------
# Load Data
# ----------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default=None, help="Path to CSV to train on")
args = parser.parse_args()


# Load data
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# csv_path = os.path.join(BASE_DIR, "data", "layoffs.csv")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "data" / "layoffs.csv"

if args.data:
    print("Retraining with new dataset")
    candidate = Path(args.data)
    csv_path = candidate if candidate.is_absolute() else (PROJECT_ROOT / args.data)
else:
    print("Retraining with  dataset layoff")
    csv_path = DEFAULT_CSV

df = pd.read_csv(csv_path)
print(f"Training on: {csv_path}")


# ----------------------------------------
# retraining the new uploaded dataset
# ----------------------------------------
def retrain_new_dataset():
    return


# ----------------------------------------
# Create layoff_severity target
# ----------------------------------------


def clean_percentage(val):
    if pd.isna(val):
        return np.nan
    return float(str(val).replace("%", ""))


df["perc_laid_off"] = df["percentage_laid_off"].apply(clean_percentage)

# If no layoffs and missing percentage → set to 0.0
df.loc[
    (df["perc_laid_off"].isna()) & (df["total_laid_off"].isna()), "perc_laid_off"
] = 0.0


def severity_class(p, total):
    if pd.isna(p):
        return 3  # Unknown
    elif p <= 10:
        return 0  # Low
    elif p <= 50:
        return 1  # Medium
    else:
        return 2  # High


df["layoff_severity"] = df.apply(
    lambda row: severity_class(row["perc_laid_off"], row["total_laid_off"]), axis=1
)

print(df["layoff_severity"].value_counts())

# ----------------------------------------
# Select Features
# ----------------------------------------

features = [
    "total_laid_off",
    "perc_laid_off",
    "funds_raised",
    "industry",
    "country",
    "stage",
]

X = df[features]
y = df["layoff_severity"]

# ----------------------------------------
# Train/Validation/Test Split
# ----------------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Train shape: {X_train.shape}")
print(f"Validation shape: {X_val.shape}")
print(f"Test shape: {X_test.shape}")

# ----------------------------------------
# Prepare Pipeline
# ----------------------------------------

preprocessor = build_preprocessing_pipeline()

models = {
    "LogisticRegression": LogisticRegression(
        max_iter=500, class_weight="balanced", multi_class="multinomial"
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        objective="multi:softprob",
        num_class=4,
        eval_metric="mlogloss",
        use_label_encoder=False,
    ),
}

best_model_name = None
best_pipeline = None
best_score = 0.0

for name, model in models.items():
    print(f"\nTraining: {name}")

    pipeline = Pipeline([("preprocessing", preprocessor), ("classifier", model)])

    pipeline.fit(X_train, y_train)

    y_pred_val = pipeline.predict(X_val)
    report = classification_report(y_val, y_pred_val, output_dict=True)
    macro_f1 = report["macro avg"]["f1-score"]

    print(f"Macro F1 on validation: {macro_f1:.4f}")

    if macro_f1 > best_score:
        best_score = macro_f1
        best_model_name = name
        best_pipeline = pipeline

print(f"\n✅ Best model: {best_model_name} (Macro F1: {best_score:.4f})")

# Evaluate on test set
y_pred_test = best_pipeline.predict(X_test)
print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_pred_test))

print("\nConfusion Matrix on Test Set:")
print(confusion_matrix(y_test, y_pred_test))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
MODEL_DIR = os.path.join(BASE_DIR, "trained_model")
model_path = os.path.join(MODEL_DIR, "layoff_pipeline.joblib")
joblib.dump(best_pipeline, model_path)
print(f"Model saved to {model_path}")


classes = np.sort(y.unique())
class_labels = ["Low (≤10%)", "Medium (10-50%)", "High (>50%)", "Unknown severity"]


metrics = save_performance_artifacts(
    y_test=y_test,
    y_pred_test=y_pred_test,
    classes=classes,
    class_labels=class_labels,
    best_model=best_model_name,
)


# Save pipeline
# joblib.dump(best_pipeline, "../api/trained_model/layoff_pipeline.joblib")
# print("\n✅ Pipeline saved as api/trained_model/layoff_pipeline.joblib")
