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


def read_single(path):
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
    df["mid"] = (df["P_a1"] + df["P_b1"]) / 2.0

    # Filter by time
    morning = (ticks.dt.hour >= 7) & (ticks.dt.hour <= 10)
    afternoon = (ticks.dt.hour >= 13) & (ticks.dt.hour <= 16)
    return df[morning | afternoon]


def build_raw_data(directory: str) -> pd.DataFrame:
    path = Path(directory)

    dfs = [
        read_single(file)
        for file in tqdm.tqdm(path.glob("*20230310_book_update.csv"))
    ]
    combined = pd.concat(dfs, ignore_index=True)
    # Return the combined DataFrame
    return combined


def build_data(
    data=Path("data/"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dec_data = np.loadtxt(data / "Train_Dst_NoAuction_DecPre_CF_7.txt")
    dec_train = dec_data[:, : int(np.floor(dec_data.shape[1] * 0.8))]
    dec_val = dec_data[:, int(np.floor(dec_data.shape[1] * 0.8)) :]

    dec_test1 = np.loadtxt(data / "Test_Dst_NoAuction_DecPre_CF_7.txt")
    dec_test2 = np.loadtxt(data / "Test_Dst_NoAuction_DecPre_CF_8.txt")
    dec_test3 = np.loadtxt(data / "Test_Dst_NoAuction_DecPre_CF_9.txt")
    dec_test = np.hstack((dec_test1, dec_test2, dec_test3))

    print(dec_train.shape, dec_val.shape, dec_test.shape)
    return dec_train, dec_val, dec_test


def to_classification(
    data: np.ndarray,
    T: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    # Select relevant columns
    df = np.array(data[:40, :].T)
    dY = np.array(data[-5:, :].T)

    # Form the lag-features
    N, D = df.shape
    dataY = dY[T - 1 : N]
    dataX = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        dataX[i - T] = df[i - T : i, :]
    x, y = dataX[:, None], dataY[:, -1] - 1
    return x.astype(np.float32), y.astype(np.int64)
