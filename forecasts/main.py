import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler

from forecasts.data import files, read_single, to_classification
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
    X_res, y_res = RandomUnderSampler().fit_resample(
        X.reshape(shape[0], -1),
        y,
    )

    return X_res.reshape(-1, *shape[1:]), y_res


def build_dataset(
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

    X, y = np.concatenate(xx), np.concatenate(yy)
    np.save(f"data/X_{subset}.npy", X)
    np.save(f"data/y_{subset}.npy", y)

    return np.load(f"data/X_{subset}.npy"), np.load(f"data/y_{subset}.npy")


def main():
    scaler = MinMaxScaler()
    with timer("Learn the normalization"):
        for file in files(subset="train"):
            features, *_ = read_single(file)
            scaler.partial_fit(features)

    with timer("Normalize the features"):
        X_train, y_train = build_dataset("train", scaler)

    with timer("Build the valid set"):
        X_valid, y_valid = build_dataset("valid", scaler)

    with timer("Build the test set"):
        # No downsampling for test set to simulate realistic scenario
        X_test, y_test = build_dataset(
            "test",
            scaler,
            downsample=no_downsample,
        )
    print(X_train.shape, y_train.shape)
    print(X_valid.shape, y_valid.shape)
    print(X_test.shape, y_test.shape)


if __name__ == "__main__":
    main()
