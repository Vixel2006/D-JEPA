import torch

from djepa.config import ModelConfig
from djepa.evaluation.alignment import solve_procrustes_centered
from djepa.evaluation.kernels import evaluate_kernel_ridge_r2
from djepa.evaluation.probes import (
    build_probe_features,
    build_subspace_specs,
)


def test_kernel_ridge_recovers_linear_signal():
    torch.manual_seed(0)
    x = torch.randn(200, 6)
    weights = torch.randn(6, 3)
    y = x @ weights
    r2 = evaluate_kernel_ridge_r2(
        x[:140], y[:140], x[140:], y[140:], kernel_type="poly2"
    )
    assert r2 > 0.9


def test_procrustes_recovers_rotation():
    torch.manual_seed(0)
    text = torch.randn(128, 8)
    q, _ = torch.linalg.qr(torch.randn(8, 8))
    vision = text @ q
    similarity = solve_procrustes_centered(vision, text)
    assert similarity > 0.99


def test_probe_features_match_declared_dimensions():
    torch.manual_seed(0)
    config = ModelConfig()
    specs = build_subspace_specs(config)
    batch = 4
    c1 = torch.randn(batch, config.component_dim)
    c2 = torch.randn(batch, config.component_dim)
    u1 = torch.randn(batch, config.component_dim)
    u2 = torch.randn(batch, config.component_dim)
    s = torch.randn(batch, config.synergy_dim)
    random_vector = torch.randn(batch, config.component_dim)

    features = build_probe_features(
        {"lejepa": torch.randn(batch, 2 * config.projection_dim)},
        {"c1": c1, "c2": c2, "u1": u1, "u2": u2, "s": s},
        random_vector,
    )
    assert list(features) == [spec.name for spec in specs]
    for spec in specs:
        assert features[spec.name].shape == (batch, spec.dim)
