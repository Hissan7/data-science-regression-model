import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score
import joblib

from preprocessing import make_preprocessor


def load_data(path: str):
    df = pd.read_csv(path)
    X = df.drop(columns=["outcome"])
    y = df["outcome"].astype(float)
    return df, X, y


def evaluate_models(df, X, y, random_state=123):
    preprocessor = make_preprocessor(df)

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=random_state),
        "Lasso": Lasso(alpha=0.001, random_state=random_state, max_iter=5000),
        "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=random_state, max_iter=5000),
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

    results = []
    for name, model in models.items():
        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", model)
        ])
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="r2")
        results.append((name, scores.mean(), scores.std()))
        print(f"{name:15s} | R2 mean: {scores.mean():.4f} | std: {scores.std():.4f}")

    results.sort(key=lambda x: x[1], reverse=True)
    best_name, best_mean, best_std = results[0]
    print("\nBest baseline:", best_name, f"(mean R2={best_mean:.4f}, std={best_std:.4f})")

    return best_name


def fit_and_save_best(df, X, y, best_name: str, random_state=123):
    preprocessor = make_preprocessor(df)

    model_map = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=random_state),
        "Lasso": Lasso(alpha=0.001, random_state=random_state, max_iter=5000),
        "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=random_state, max_iter=5000),
    }

    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", model_map[best_name])
    ])

    pipe.fit(X, y)
    joblib.dump(pipe, "models/baseline_model.joblib")
    print("Saved model to models/baseline_model.joblib")


def main():
    df, X, y = load_data("data/CW1_train.csv")
    best_name = evaluate_models(df, X, y)
    fit_and_save_best(df, X, y, best_name)


if __name__ == "__main__":
    main()
