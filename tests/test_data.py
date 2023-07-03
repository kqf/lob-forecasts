from forecasts.data import files, read_single, to_classification


def test_data(dataset):  # sourcery skip: extract-duplicate-method
    filename = next(iter(files("train", directory=dataset)))
    features, labels, ticks = read_single(filename)
    assert features.shape == (1491, 1, 10, 20)
    assert labels.shape == (1491,)
    assert ticks.shape == (1491, 1)

    print(features.shape, labels.shape, ticks.shape)
    X, y, dt = to_classification(features, labels, ticks)
    print(filename, X.shape, y.shape, dt.shape)
    assert X.shape == (1491, 1, 10, 20)
    assert y.shape == (1491,)
    assert dt.shape == (1491, 1)
