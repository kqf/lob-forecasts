import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score, classification_report

from forecasts.data import files, read_single, to_classification
from forecasts.model import build_model
from forecasts.timer import timer


def remove_nans(X, y, dt) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    # Downsample the missing data examples for now
    isnan = np.isnan(X).any(axis=(-3, -2, -1)) | np.isnan(y)
    return (
        X.compress(~isnan, 0),
        y.compress(~isnan, 0),
        dt.loc[~isnan],
    )


def build_eval_dataset(scaler) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    first_test_day = next(iter(files(subset="test")))
    features, labels, dt = read_single(first_test_day)
    X, y, dt = to_classification(scaler.transform(features), labels, dt)
    X, y, dt = remove_nans(X, y, dt)
    return X, y, dt


LABEL_MAPPING = {
    0: 1,
    1: -1,
    2: 0,
}


def main():
    with timer("Read the normalization"):
        scaler = load("data/scaler.pickle")

    with timer("Build the test set"):
        X_test_, y_test_, dt = build_eval_dataset(scaler)

    model = build_model(
        num_classes=3,
        batch_size=64,
        train_split=None,
    )
    model.initialize()
    model.load_params(f_params="data/best.pt")

    y_pred = model.predict(X_test_)
    print()
    print("accuracy_score:", accuracy_score(y_test_, y_pred))
    print(classification_report(y_test_, y_pred, digits=4))
    dt["Predictions"] = y_pred
    dt["Predictions"] = dt["Predictions"].map(LABEL_MAPPING)
    print(dt)
    dt.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()