import pathlib

import numpy as np
import pandas as pd
import pytest

from forecasts.data import COLUMNS, FEATURES


def legacy_fake(path):
    np.savetxt(path, np.random.rand(149, 10000))


def fake(
    start="2023-06-27 06:06:00",
    stop="2023-06-27 10:02:00",
) -> pd.DataFrame:
    datetime_range = pd.date_range(start=start, end=stop, freq="30L")
    # Create a DataFrame with the datetime column
    df = pd.DataFrame({"Date_time": datetime_range})
    df["idx"] = df.index
    df["currency"] = "EURUSD"
    for c in COLUMNS:
        if c in df.columns:
            continue
        df[c] = np.nan

    for c in FEATURES:
        df[c] = np.random.rand(*df.index.values.shape)
    return df


@pytest.fixture
def fake_dataset(tmp_path: pathlib.Path) -> pathlib.Path:
    datadir = tmp_path / " legacy-data"
    datadir.mkdir(exist_ok=True)
    legacy_fake(datadir / "Train_Dst_NoAuction_DecPre_CF_7.txt")
    legacy_fake(datadir / "Test_Dst_NoAuction_DecPre_CF_7.txt")
    legacy_fake(datadir / "Test_Dst_NoAuction_DecPre_CF_8.txt")
    legacy_fake(datadir / "Test_Dst_NoAuction_DecPre_CF_9.txt")
    return datadir


@pytest.fixture
def dataset(tmp_path: pathlib.Path) -> pathlib.Path:
    datadir = tmp_path / "data"
    datadir.mkdir(exist_ok=True)
    df = fake()
    df.to_csv(datadir / "data.csv", index=False, header=None)
    return datadir
