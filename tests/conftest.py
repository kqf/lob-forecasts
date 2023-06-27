import pathlib

import numpy as np
import pytest

from forecasts.fake import fake


def legacy_fake(path):
    np.savetxt(path, np.random.rand(149, 10000))


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
    datadir = tmp_path / " data"
    datadir.mkdir(exist_ok=True)
    df = fake()
    df.to_csv(datadir, index=False, header=None)
    return datadir
