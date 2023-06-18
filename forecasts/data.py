from pathlib import Path

import numpy as np
import torch


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


def data_classification(
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
    return dataX[:, None], dataY[:, -1] - 1


class LobDataset(torch.utils.data.Dataset):
    def __init__(self, data: np.ndarray, T: int = 100):
        self.x, self.y = data_classification(data, T)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        return self.x[index].astype(np.float32), self.y[index].astype(np.int64)
