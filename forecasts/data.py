from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import tqdm

COLUMNS = [
    "idx",
    "currency",
    "tick",
    "P_a1",
    "P_a2",
    "P_a3",
    "P_a4",
    "P_a5",
    "exclude_8",
    "exclude_9",
    "exclude_10",
    "exclude_11",
    "exclude_12",
    "P_b1",
    "P_b2",
    "P_b3",
    "P_b4",
    "P_b5",
    "exclude_18",
    "exclude_19",
    "exclude_20",
    "exclude_21",
    "exclude_22",
    "V_a1",
    "V_a2",
    "V_a3",
    "V_a4",
    "V_a5",
    "exclude_28",
    "exclude_29",
    "exclude_30",
    "exclude_31",
    "exclude_32",
    "V_b1",
    "V_b2",
    "V_b3",
    "V_b4",
    "V_b5",
    "exclude_38",
    "exclude_39",
    "exclude_40",
    "exclude_41",
    "exclude_42",
    "exclude_43",
    "exclude_44",
    "exclude_45",
    "exclude_46",
    "exclude_47",
]

FEATURES = [
    "P_a1",
    "V_a1",
    "P_b1",
    "V_b1",
    "P_a2",
    "V_a2",
    "P_b2",
    "V_b2",
    "P_a3",
    "V_a3",
    "P_b3",
    "V_b3",
    "P_a4",
    "V_a4",
    "P_b4",
    "V_b4",
    "P_a5",
    "V_a5",
    "P_b5",
    "V_b5",
]


LABEL_MAPPING = {
    "up": 0,
    "down": 1,
    "stationary": 2,
}


def label(l_t: float, alpha: float) -> str:
    if l_t > alpha:
        return "up"
    if l_t < -alpha:
        return "down"
    return "stationary"


def files(
    subset: Literal["train"] | Literal["valid"] | Literal["test"] = "train",
    directory: str = "data/EURUSD/",
    n_valid: int = 5,
    n_test: int = 1,
):
    path = Path(directory)
    # Sort the files by date
    files = sorted(path.glob("*.csv"), key=lambda x: x.stem.split("_")[1])
    sets = {
        "train": files[: -n_valid - n_test],
        "valid": files[-n_valid - n_test : -n_valid],
        "test": files[-n_valid:],
    }
    for file in tqdm.tqdm(sets[subset]):
        yield file


def read_single(path, horizon: int = 10, alpha=0.000015):
    usecols = [c for c in COLUMNS[2:] if not c.startswith("exclude")]
    df = pd.read_csv(
        path,
        header=None,
        names=COLUMNS,
        usecols=usecols,
        low_memory=True,
    )
    df["tick"] = pd.to_datetime(df["tick"], format="%Y%m%d-%H:%M:%S.%f")
    # Calculate the mid prices
    df["mid"] = (df["P_a1"] + df["P_b1"]) / 2.0

    # Rolling average of previous "horizon" ticks
    df["m_minus"] = df["mid"].rolling(window=horizon).mean()
    # Rolling average of next "horizon" ticks
    df["m_plus"] = df["mid"].rolling(window=horizon).mean().shift(-horizon + 1)
    df = df.loc[df["m_minus"].notna() & df["m_plus"].notna()]

    # Calculate the labels
    df["l_t"] = (df["m_plus"] - df["m_minus"]) / df["m_minus"]
    df["label"] = df["l_t"].apply(partial(label, alpha=alpha))

    return df[FEATURES], df["label"].map(LABEL_MAPPING), df["tick"]


def to_classification(
    X: np.ndarray,
    y: np.ndarray,
    ticks: pd.Series,
    T: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    # Select relevant columns
    df = np.array(X)
    dY = np.array(y)

    # Form the lag-features
    N, D = df.shape
    dt = ticks[T - 1 : N]
    dataX = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        dataX[i - T] = df[i - T : i, :]

    dataY = dY[T - 1 : N]
    x, y = dataX[:, None], dataY

    dt = ticks[T - 1 : N]
    # Filter by time as the last step
    morning = (dt.dt.hour >= 7) & (dt.dt.hour <= 10)
    afternoon = (dt.dt.hour >= 13) & (dt.dt.hour <= 16)
    index = morning | afternoon
    return x[index].astype(np.float32), y[index].astype(np.int64)
