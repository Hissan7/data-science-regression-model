import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from preprocessing import make_preprocessor


def load_data(path: str):
    df = pd.read_csv(path)
    X = df.drop(columns=["outcome"])
    y = df["outcome"].astype(float)
    return df, X, y


def evaluate_tree_ensembles(df, X, y, random_state=123):
    """
    Compare tree-based ensemble regression models using cross-validated R^2.
    """

    preprocessor = make_preprocessor(df)

    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=-1
        ),
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

    results = []
    print("\nTree-based ensemble results:")
    print("-" * 40)

    for name, model in models.items():
        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", model)
        ])

        scores = cross_val_score(pipe, X, y, cv=cv, scoring="r2")
        mean_r2 = scores.mean()
        std_r2 = scores.std()

        results.append((name, mean_r2, std_r2))
        print(f"{name:15s} | R2 mean: {mean_r2:.4f} | std: {std_r2:.4f}")

    results.sort(key=lambda x: x[1], reverse=True)
    best_name, best_mean, best_std = results[0]

    print("\nBest tree ensemble:")
    print(f"{best_name} (mean R2={best_mean:.4f}, std={best_std:.4f})")

    return results


def main():
    df, X, y = load_data("data/CW1_train.csv")
    evaluate_tree_ensembles(df, X, y)


if __name__ == "__main__":
    main()
