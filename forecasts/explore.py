from forecasts.data import files, read_single
from forecasts.timer import timer


def main():
    with timer("Load the raw data"):
        for file in files():
            df = read_single(file)
            continue
    print(df.head())


if __name__ == "__main__":
    main()
