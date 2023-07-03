from forecasts.data import files, read_single, to_classification


def test_data(dataset):
    filename = next(iter(files("train", directory=dataset)))
    features, labels, ticks = read_single(filename)
    X, y, dt = to_classification(features, labels, ticks)
    print(filename, X.shape, y.shape, dt.shape)
    assert X.shape == (1491, 1, 10, 20)
    assert y.shape == (1491,)
    assert dt.shape == (1491, 1)
