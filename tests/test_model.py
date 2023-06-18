import pytest
import torch

from forecasts.model import DeepLob


@pytest.fixture
def batch(device, batch_size):
    return torch.ones((batch_size, 1, 100, 40), device=device)


@pytest.mark.parametrize(
    "device",
    [
        torch.device("cpu"),
        # torch.device("cuda"),
    ],
)
@pytest.mark.parametrize(
    "batch_size",
    [
        16,
        32,
        64,
    ],
)
def test_model(batch, device, batch_size):
    model = DeepLob(num_classes=3).to(device)
    assert model(batch).shape == (batch_size, 3)
