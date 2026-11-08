import torch

from djepa.losses import (
    covariance_decorrelation_loss,
    lejepa_loss,
    sigreg_loss,
)


def test_sigreg_loss_is_finite():
    torch.manual_seed(0)
    z = torch.randn(8, 16)
    value = sigreg_loss(z, num_slices=32)
    assert value.ndim == 0
    assert torch.isfinite(value)


def test_lejepa_loss_returns_three_terms():
    z1 = torch.randn(8, 16)
    z2 = z1.clone()
    total, invariance, regulariser = lejepa_loss(
        z1, z2, sigreg_weight=0.02, num_slices=32
    )
    assert torch.isclose(invariance, torch.tensor(0.0))
    assert regulariser.ndim == 0
    assert torch.isfinite(total)


def test_covariance_decorrelation_is_nonnegative():
    torch.manual_seed(1)
    a = torch.randn(16, 8)
    b = torch.randn(16, 8)
    value = covariance_decorrelation_loss(a, b)
    assert value.ndim == 0
    assert value >= 0.0
