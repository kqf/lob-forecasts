from forecasts.data import files


def test_data(dataset):
    for file in files(directory=dataset):
        print(file)

    print("d")
    print(dataset)
