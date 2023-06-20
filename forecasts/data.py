from functools import partial
from pathlib import Path

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
]


def label(l_t: float, alpha: float) -> str:
    if l_t > alpha:
        return "up"
    if l_t < -alpha:
        return "down"
    return "stationary"


def files(directory: str = "data/EURUSD/"):
    path = Path(directory)
    for file in tqdm.tqdm(path.glob("*20230310_book_update.csv")):
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
    ticks = df["tick"]
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
    print(df["label"].value_counts())

    # Filter by time
    morning = (ticks.dt.hour >= 7) & (ticks.dt.hour <= 10)
    afternoon = (ticks.dt.hour >= 13) & (ticks.dt.hour <= 16)
    print(f"After removing invalid timing {len(df)=}")
    return df.loc[morning | afternoon]


def to_sequences_classification(
    X: np.ndarray,
    y: np.ndarray,
    T: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    # Select relevant columns
    df = np.array(X)
    dY = np.array(y)

    # Form the lag-features
    N, D = df.shape
    dataY = dY[T - 1 : N]
    dataX = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        dataX[i - T] = df[i - T : i, :]
    x, y = dataX[:, None], dataY
    return x.astype(np.float32), y.astype(np.int64)
