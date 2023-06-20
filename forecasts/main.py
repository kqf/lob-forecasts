import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler

from forecasts.data import files, read_single, to_classification
from forecasts.timer import timer


def downsample(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(X) == 0:
        return X, y

    shape = X.shape
    X_res, y_res = RandomUnderSampler().fit_resample(
        X.reshape(shape[0], -1),
        y,
    )

    return X_res.reshape(-1, *shape[1:]), y_res


def build_valid(
    subset,
    scaler,
    downsample=downsample,
) -> tuple[np.ndarray, np.ndarray]:
    xx, yy = [], []
    for file in files(subset=subset):
        features, labels, dt = read_single(file)
        X, y = to_classification(scaler.transform(features), labels, dt)
        X_res, y_res = downsample(X, y)
        xx.append(X_res)
        yy.append(y_res)

    return np.concatenate(xx), np.concatenate(yy)


def main():
    scaler = MinMaxScaler()
    with timer("Learn the normalization"):
        for file in files(subset="train"):
            features, *_ = read_single(file)
            scaler.partial_fit(features)
            break

    with timer("Normalize the features"):
        for file in files():
            features, labels, dt = read_single(file)
            X, y = to_classification(scaler.transform(features), labels, dt)
            X_res, y_res = downsample(X, y)
            print("~", X_res.shape, y_res.shape)
            break

    with timer("Build the valid set"):
        X_valid, y_valid = build_valid(
            "valid",
            scaler,
        )
        print(X_valid.shape, y_valid.shape)

    with timer("Build the test set"):
        X_test, y_test = build_valid(
            "test",
            scaler,
            # No downsampling for test set to simulate realistic scenario
            downsample=lambda X, y: (X, y),
        )
        print(X_test.shape, y_test.shape)


if __name__ == "__main__":
    main()
