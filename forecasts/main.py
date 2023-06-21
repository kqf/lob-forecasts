from functools import partial

import numpy as np
import skorch
from imblearn.under_sampling import RandomUnderSampler
from joblib import dump, load
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler

from forecasts.data import files, read_single, to_classification
from forecasts.model import build_model
from forecasts.timer import timer


def no_downsample(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return X, y


def downsample(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(X) == 0:
        return X, y

    shape = X.shape
    X_res, y = RandomUnderSampler().fit_resample(
        X.reshape(shape[0], -1),
        y,
    )

    return X_res.reshape(-1, *shape[1:]), y


def remove_nans(X, y):
    # Downsample the missing data examples for now
    isnan = np.isnan(X).any(axis=(-3, -2, -1)) | np.isnan(y)
    return (
        X.compress(~isnan, 0),
        y.compress(~isnan, 0),
    )


def build_dataset(
    subset,
    scaler,
    downsample=downsample,
) -> tuple[np.ndarray, np.ndarray]:
    XX, yy = np.empty((0, 1, 10, 20), dtype=np.float32), np.empty(
        (0), dtype=np.int64
    )
    for file in files(subset=subset):
        features, labels, dt = read_single(file)
        X, y = to_classification(scaler.transform(features), labels, dt)
        X, y = remove_nans(X, y)
        X, y = downsample(X, y)

        XX = np.append(XX, X, axis=0)
        yy = np.append(yy, y)

    np.save(f"data/X_{subset}.npy", XX)
    np.save(f"data/y_{subset}.npy", yy)
    print("Saved")

    return np.load(f"data/X_{subset}.npy"), np.load(f"data/y_{subset}.npy")


def train_split(X, y, X_valid, y_valid):
    return X, skorch.dataset.Dataset(X=X_valid, y=y_valid)


def main():
    scaler = MinMaxScaler()
    with timer("Learn the normalization"):
        for file in files(subset="train"):
            features, *_ = read_single(file)
            scaler.partial_fit(features)
        dump(scaler, "data/scaler.pickle")
        scaler = load("data/scaler.pickle")

    with timer("Build the train set"):
        X_train, y_train = build_dataset("train", scaler)

    with timer("Build the valid set"):
        X_valid, y_valid = build_dataset("valid", scaler)

    with timer("Build the test set"):
        # No downsampling for test set to simulate realistic scenario
        X_test_, y_test_ = build_dataset(
            "test",
            scaler,
            downsample=no_downsample,
        )

    model = build_model(
        num_classes=3,
        batch_size=64,
        train_split=partial(train_split, X_valid=X_valid, y_valid=y_valid),
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test_)
    print()
    print("accuracy_score:", accuracy_score(y_test_, y_pred))
    print(classification_report(y_test_, y_pred, digits=4))


if __name__ == "__main__":
    main()
