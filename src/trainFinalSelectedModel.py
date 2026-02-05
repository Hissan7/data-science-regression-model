import pandas as pd
from sklearn.pipeline import Pipeline
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
    df = pd.read_csv(path)
    return df


def main():
    # Load training data
    train_df, X_train, y_train = load_train_data(TRAIN_PATH)

    # Build preprocessing + model pipeline
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

    # Train final model 
    pipeline.fit(X_train, y_train)

    # Load test data
    test_df = load_test_data(TEST_PATH)

    # Predict outcomes
    test_predictions = pipeline.predict(test_df)

    submission = pd.DataFrame({
        "outcome": test_predictions
    })

    submission.to_csv(SUBMISSION_PATH, index=False)

    print(f"Submission file written to: {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()

