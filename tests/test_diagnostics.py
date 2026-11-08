import torch

from djepa.diagnostics import (
    compute_calibration,
    confusion_matrix,
    expected_calibration_error,
    linear_cka,
    spectral_metrics,
)


def test_linear_cka_is_one_for_identical_representations():
    torch.manual_seed(0)
    x = torch.randn(64, 8)
    assert abs(linear_cka(x, x) - 1.0) < 1e-5


def test_spectral_metrics_prefers_low_rank_structure():
    torch.manual_seed(0)
    latent = torch.randn(128, 1)
    rank_one = latent @ torch.randn(1, 16)
    isotropic = torch.randn(128, 16)
    low_rank = spectral_metrics("low", rank_one)
    full_rank = spectral_metrics("full", isotropic)
    assert low_rank.effective_rank < full_rank.effective_rank
    assert low_rank.participation_ratio < full_rank.participation_ratio


def test_expected_calibration_error_is_small_when_confident_and_correct():
    logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
    labels = torch.tensor([0, 1])
    assert expected_calibration_error(logits, labels) < 0.01


def test_confusion_matrix_and_calibration_summary():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]])
    labels = torch.tensor([0, 1, 1])
    matrix = confusion_matrix(logits.argmax(-1), labels, num_classes=2)
    assert matrix == [[1, 0], [1, 1]]

    rows = compute_calibration(
        logits={"probe": torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]])},
        labels=labels,
        num_classes=2,
        conditions=("probe",),
    )
    assert len(rows) == 1
    assert rows[0].condition == "probe"
    assert 0.0 <= rows[0].ece <= 1.0
