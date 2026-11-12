"""Fast regression tests for the synthetic pipeline and PID helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from pidjepa.data import make_synthetic_bundle
from pidjepa.evaluation import (
    LinearProbe,
    estimate_flow_pid,
    evaluate_pid_match,
    linear_cka,
    minimum_mutual_information_pid,
    mutual_knn,
    representation_diagnostics,
    ridge_r2,
    run_seed_experiment,
)
from pidjepa.models import (
    ARM_CONFIGS,
    TokenEncoder,
    arm_family,
    build_model,
)
from pidjepa.training import encode_dataset, extract_raw_features, pretrain_arm


def small_args() -> SimpleNamespace:
    """Return tiny but structurally complete benchmark arguments."""
    return SimpleNamespace(
        block_dim=4,
        hidden_dim=16,
        transformer_dim=16,
        transformer_layers=1,
        attention_heads=2,
        learning_rate=1e-3,
        pretrain_steps=2,
        batch_size=8,
        sigreg_weight=0.05,
        invariance_weight=1.0,
        cross_modal_weight=1.0,
        reconstruction_weight=1.0,
        independence_weight=0.2,
        sigreg_slices=8,
        probe_epochs=2,
        pid_dim=4,
        pid_steps=2,
        no_flow_pid=True,
    )


@pytest.fixture(scope="module")
def bundle():
    return make_synthetic_bundle(
        n_samples=96,
        seed=0,
        n_views=3,
        n_shared=3,
        n_private=3,
        n_nuisance=2,
        vision_dim=8,
        text_dim=6,
    )


def test_synthetic_bundle_schema(bundle):
    assert bundle.kind == "vec"
    assert bundle.has_ground_truth
    assert bundle.vision_views.shape == (96, 3, 8)
    assert bundle.text_views.shape == (96, 3, 6)
    assert set(bundle.tasks) == {"shared", "priv_v", "priv_t", "synergy"}
    assert set(bundle.factors) == {"shared", "priv_v", "priv_t"}
    assert set(bundle.split) == {"train", "val", "test"}
    assert sum(len(indices) for indices in bundle.split.values()) == 96
    assert torch.isfinite(bundle.vision_views).all()
    assert torch.isfinite(bundle.text_views).all()


@pytest.mark.parametrize("arm_name", list(ARM_CONFIGS))
def test_every_arm_pretrains_and_encodes(bundle, arm_name):
    args = small_args()
    config = ARM_CONFIGS[arm_name]
    model = build_model(bundle, config, args)
    pretrain_arm(model, config, bundle, args, "cpu", _sigreg(args), arm_name)
    features = encode_dataset(model, bundle, "cpu")

    expected_keys = (
        {"z1", "z2"} if arm_family(arm_name) == "alignment" else {"c1", "c2", "u1", "u2"}
    )
    assert set(features) == expected_keys
    for value in features.values():
        assert value.shape[0] == 96
        assert torch.isfinite(value).all()


def test_raw_features_are_concatenated(bundle):
    raw = extract_raw_features(bundle)
    assert raw["h1"].shape == (96, 8)
    assert raw["h2"].shape == (96, 6)


def test_token_encoder_masks_padding():
    encoder = TokenEncoder(
        in_dim=8,
        n_tokens=5,
        model_dim=16,
        n_layers=1,
        n_heads=2,
        hidden_dim=16,
        out_dim=8,
    )
    tokens = torch.randn(4, 5, 8)
    mask = torch.ones(4, 5, dtype=torch.bool)
    mask[0, 3:] = False
    output = encoder(tokens, mask)
    assert output.shape == (4, 8)
    assert torch.isfinite(output).all()


def test_probe_and_representation_metrics():
    torch.manual_seed(0)
    x = torch.randn(64, 8)
    y = x + 0.1 * torch.randn(64, 8)
    labels = torch.randint(0, 3, (64,))

    assert linear_cka(x, y) > 0.5
    assert 0.0 <= mutual_knn(x, y) <= 1.0
    assert ridge_r2(x[:48], y[:48], x[48:], y[48:]) > 0.5

    std, effective_rank = representation_diagnostics(x)
    assert std > 0
    assert effective_rank > 0

    probe = LinearProbe(in_dim=8, n_classes=3, device="cpu", hidden_dim=16, seed=0)
    probe.fit(x[:48], labels[:48], x[48:], labels[48:], epochs=2)
    assert probe.predict(x[48:]).shape == (16,)


def test_pid_helpers():
    pid = minimum_mutual_information_pid(1.0, 0.5, 2.0)
    assert pid["R"] == 0.5
    assert pid["U1"] == 0.5
    assert pid["U2"] == 0.0
    assert pid["S"] == 1.0

    matched, fraction = evaluate_pid_match(pid, "S", reference_total=2.0)
    assert matched is True
    assert fraction == pytest.approx(0.5)


def test_flow_pid_runs():
    torch.manual_seed(0)
    a_train, b_train = torch.randn(64, 2), torch.randn(64, 2)
    target_train = a_train[:, 0] + b_train[:, 0]
    a_test, b_test = torch.randn(32, 2), torch.randn(32, 2)
    target_test = a_test[:, 0] + b_test[:, 0]

    pid = estimate_flow_pid(
        a_train,
        b_train,
        target_train,
        a_test,
        b_test,
        target_test,
        steps=2,
        device="cpu",
        seed=0,
        hidden_dim=8,
        n_layers=1,
    )
    assert {"I1", "I2", "I12", "R", "U1", "U2", "S", "null_I12"} <= set(pid)


def test_run_seed_experiment_smoke(bundle):
    args = small_args()
    output = run_seed_experiment(seed=0, args=args, bundle=bundle, device="cpu")
    assert ("djepa", "joint", "shared") in output["acc"]
    assert ("alignment", "joint", "shared") in output["acc"]
    assert ("raw", "joint", "shared") in output["acc"]
    assert ("djepa", "stream", "shared", "all") in output["ppid"]
    assert ("djepa", "stream", "shared") in output["fpid"] or args.no_flow_pid
    assert ("djepa", "c1", "shared") in output["idmat"]
    assert ("raw", "h1", "shared") in output["idmat"]


def test_report_and_report_figures(tmp_path, bundle):
    pytest.importorskip("seaborn")
    pytest.importorskip("sklearn")
    from pidjepa.figures import make_report_plots
    from pidjepa.reporting import aggregate_results

    args = small_args()
    output = run_seed_experiment(seed=0, args=args, bundle=bundle, device="cpu")
    summary = aggregate_results([output], bundle)
    assert "acc" in summary
    assert "ppid" in summary
    make_report_plots(summary, str(tmp_path))
    assert (tmp_path / "fig2_modality_gap.png").exists()
    assert (tmp_path / "fig3_factor_dissection_heatmap.png").exists()
    assert (tmp_path / "fig6_probe_pid.png").exists()


def test_cli_end_to_end(tmp_path):
    pytest.importorskip("seaborn")
    pytest.importorskip("sklearn")
    from pidjepa.cli import main

    exit_code = main(
        [
            "--mode",
            "synthetic",
            "--seeds",
            "1",
            "--pretrain-steps",
            "1",
            "--batch-size",
            "8",
            "--synthetic-samples",
            "64",
            "--block-dim",
            "4",
            "--hidden-dim",
            "16",
            "--transformer-dim",
            "16",
            "--transformer-layers",
            "1",
            "--attention-heads",
            "2",
            "--probe-epochs",
            "1",
            "--pid-steps",
            "2",
            "--no-flow-pid",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert exit_code == 0
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "fig6_probe_pid.png").exists()


def _sigreg(args):
    from pidjepa.models import SIGReg

    return SIGReg(n_slices=args.sigreg_slices)
