import pandas as pd


def fake(
    start="2023-06-27 06:06:00",
    stop="2023-06-27 10:02:00",
) -> pd.DataFrame:
    datetime_range = pd.date_range(start=start, end=stop, freq="30L")
    # Create a DataFrame with the datetime column
    df = pd.DataFrame({"Date_time": datetime_range})
    df["idx"] = df.index
    df["currency"] = "EURUSD"
    return df
