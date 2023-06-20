from collections import Counter

from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler

from forecasts.data import files, read_single, to_classification
from forecasts.timer import timer


def main():
    scaler = MinMaxScaler()
    with timer("Learn the normalization"):
        for file in files():
            features, *_ = read_single(file)
            scaler.partial_fit(features)

    with timer("Normalize the features"):
        for file in files():
            features, labels, dt = read_single(file)
            X, y = to_classification(scaler.transform(features), labels, dt)
            shape = X.shape
            X_res, y_res = RandomUnderSampler().fit_resample(
                X.reshape(shape[0], -1),
                y,
            )
            print(Counter(y_res))
            print("~", X_res.reshape(-1, *shape[1:]).shape, y_res.shape)
            continue
    # print(df.head())


if __name__ == "__main__":
    main()
