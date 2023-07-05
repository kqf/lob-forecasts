from forecasts.data import files, read_single, to_classification


def test_data(dataset):
    filename = next(iter(files("train", directory=dataset)))
    features, labels, ticks = read_single(filename)
    n_obj = len(features)
    assert features.shape == (n_obj, 20)
    assert labels.shape == (n_obj,)
    assert ticks.shape == (n_obj,)

    print(features.shape, labels.shape, ticks.shape)
    X, y, dt = to_classification(features, labels, ticks)
    print(filename, X.shape, y.shape, dt.shape)
    assert X.shape == (len(X), 1, 10, 20)
    assert y.shape == (len(y),)
    assert dt.shape == (len(dt), 1)
