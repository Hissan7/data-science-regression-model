import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def make_preprocessor(df: pd.DataFrame):
    """
    Create a preprocessing pipeline for the CW1 dataset.
    Assumes 'outcome' is the target variable.
    """

    # Identify columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove target from features
    if "outcome" in numeric_cols:
        numeric_cols.remove("outcome")

    # Pipelines
    numeric_pipeline = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # Column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols)
        ]
    )

    return preprocessor
