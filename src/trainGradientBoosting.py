import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor

from preprocessing import make_preprocessor


def load_data(path: str):
    df = pd.read_csv(path)
    X = df.drop(columns=["outcome"])
    y = df["outcome"].astype(float)
    return df, X, y


def evaluate_gradient_boosting(df, X, y, random_state=123):
    """
    Evaluate Gradient Boosting Regressor using cross-validated R^2.
    """

    preprocessor = make_preprocessor(df)

    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,  #5 fold
        max_depth=3,
        subsample=1.0,
        random_state=random_state
    )

    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])

    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

    scores = cross_val_score(pipe, X, y, cv=cv, scoring="r2")

    mean_r2 = scores.mean()
    std_r2 = scores.std()

    print("\nGradient Boosting Regressor results:")
    print("-" * 45)
    print(f"R2 mean: {mean_r2:.4f}")
    print(f"R2 std : {std_r2:.4f}")

    return mean_r2, std_r2


def main():
    df, X, y = load_data("data/CW1_train.csv")
    evaluate_gradient_boosting(df, X, y)


if __name__ == "__main__":
    main()
