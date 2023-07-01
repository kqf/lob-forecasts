from forecasts.data import files


def test_data(dataset):
    filename = next(iter(files("train", directory=dataset)))
    print(filename)
    print(dataset)
