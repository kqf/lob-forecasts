import pytest
import torch

from forecasts.model import DeepLob


@pytest.fixture
def batch(device):
    return torch.ones((64, 1, 100, 40), device=device)


@pytest.mark.parametrize(
    "device",
    [
        torch.device("cpu"),
        # torch.device("cuda"),
    ],
)
def test_model(batch):
    model = DeepLob(y_len=3)
    assert model(batch).shape == (64, 3)
