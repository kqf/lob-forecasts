import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score, classification_report

from forecasts.data import COLUMNS, files, read_single, to_classification
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


def build_eval_dataset(
    scaler,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, str]:
    first_test_day = next(iter(files(subset="test")))
    features, labels, dt = read_single(first_test_day)
    X, y, dt = to_classification(scaler.transform(features), labels, dt)
    X, y, dt = remove_nans(X, y, dt)
    return X, y, dt, first_test_day


LABEL_MAPPING = {
    0: 1,
    1: -1,
    2: 0,
}


def to_csv(fname, dt: pd.DataFrame) -> None:
    df = pd.read_csv(fname, header=None, names=COLUMNS)
    df["Date_time"] = pd.to_datetime(
        df["Date_time"],
        format="%Y%m%d-%H:%M:%S.%f",
    )
    merged = pd.merge(df, dt, on="Date_time", how="left")
    merged["Predictions"] = merged["Predictions"].fillna(0)
    merged.to_csv(f"{fname.stem}-predictions.csv", index=False)


def main():
    with timer("Read the normalization"):
        scaler = load("data/scaler.pickle")

    with timer("Build the test set"):
        X_test_, y_test_, dt, fname = build_eval_dataset(scaler)

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
    to_csv(fname, dt)


if __name__ == "__main__":
    main()
