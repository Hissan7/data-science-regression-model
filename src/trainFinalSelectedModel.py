import os
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor

from preprocessing import make_preprocessor


TRAIN_PATH = "data/CW1_train.csv"
TEST_PATH = "data/CW1_test.csv"
SUBMISSION_PATH = "submissions/CW1_submission.csv"


def load_train_data(path: str):
    df = pd.read_csv(path)
    X = df.drop(columns=["outcome"])
    y = df["outcome"].astype(float)
    return df, X, y


def load_test_data(path: str):
    return pd.read_csv(path)


def print_cv_metrics(train_df, X_train, y_train, random_state=123):
    """Print 5-fold cross-validated R^2 for the final selected model."""
    preprocessor = make_preprocessor(train_df)

    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=1.0,
        random_state=random_state,
    )

    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])

    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2")

    mean_r2 = scores.mean()
    std_r2 = scores.std()

    print("\nGradient Boosting Regressor results:")
    print("-" * 45)
    print(f"R2 mean: {mean_r2:.4f}")
    print(f"R2 std : {std_r2:.4f}")

    return mean_r2, std_r2


def main():
    # Load training data
    train_df, X_train, y_train = load_train_data(TRAIN_PATH)

    # 1) Print CV performance (same as trainGradientBoosting.py)
    print_cv_metrics(train_df, X_train, y_train, random_state=123)

    # 2) Fit final model on ALL training data
    preprocessor = make_preprocessor(train_df)
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=1.0,
        random_state=123
    )

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    # 3) Predict test and write submission
    test_df = load_test_data(TEST_PATH)
    test_predictions = pipeline.predict(test_df)

    submission = pd.DataFrame({"outcome": test_predictions})

    # Ensure folder exists even if gitignored
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)

    print(f"Submission file written to: {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
