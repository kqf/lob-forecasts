from forecasts.data import files, read_single


def test_data(dataset):
    filename = next(iter(files("train", directory=dataset)))
    X, y, dt = read_single(filename)
    print(filename, X.shape, y.shape, dt.shape)
    print(dataset)
