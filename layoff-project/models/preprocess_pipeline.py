# models/pipeline.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# ------------------------------------------
# Custom Transformer for funds_raised
# ------------------------------------------


class CleanFundsTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to clean funds_raised column:
    "$234" → 234.0
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        cleaned_series = X.iloc[:, 0].apply(self._clean_value)
        return cleaned_series.to_frame()

    def _clean_value(self, val):
        if pd.isna(val):
            return np.nan
        return float(str(val).replace("$", "").replace(",", ""))


# ------------------------------------------
# Function to build preprocessing pipeline
# ------------------------------------------


def build_preprocessing_pipeline():
    numeric_features = ["total_laid_off", "perc_laid_off"]

    categorical_features = ["industry", "country", "stage"]

    # Numeric pipeline
    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    # Funds pipeline
    funds_pipeline = Pipeline(
        [
            ("cleaner", CleanFundsTransformer()),
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical pipeline
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("funds", funds_pipeline, ["funds_raised"]),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor
