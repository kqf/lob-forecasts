from forecasts.data import build_raw_data
from forecasts.timer import timer


def main():
    with timer("Load the raw data"):
        df = build_raw_data("data/EURUSD/")
    print(df.head())


if __name__ == "__main__":
    main()
