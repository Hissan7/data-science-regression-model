import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score

from preprocessing import make_preprocessor

from xgboost import XGBRegressor


def load_data(path: str):
    df = pd.read_csv(path)
    X = df.drop(columns=["outcome"])
    y = df["outcome"].astype(float)
    return df, X, y


def evaluate_xgboost(df, X, y, random_state=123):
    """
    Cross-validated R^2 evaluation for XGBoost regression.
    Uses the same preprocessing + CV setup as other model scripts for fair comparison.
    """

    preprocessor = make_preprocessor(df)

    model = XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1
    )

    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])

    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="r2")

    print("\nXGBoost Regressor results:")
    print("-" * 40)
    print(f"R2 mean: {scores.mean():.4f}")
    print(f"R2 std : {scores.std():.4f}")

    return scores.mean(), scores.std()


def main():
    df, X, y = load_data("data/CW1_train.csv")
    evaluate_xgboost(df, X, y)


if __name__ == "__main__":
    main()
