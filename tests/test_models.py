import torch

from djepa.evaluation.probes import CapacityMatchedTaskHead
from djepa.models.projectors import LeJEPAProjector, SynergyHead, make_predictor


def test_lejepa_projector_shape():
    projector = LeJEPAProjector(384, 256)
    output = projector(torch.randn(4, 384))
    assert output.shape == (4, 256)


def test_capacity_matched_head_shape():
    head = CapacityMatchedTaskHead(512, bottleneck=128, num_classes=10)
    output = head(torch.randn(4, 512))
    assert output.shape == (4, 10)


def test_predictor_and_synergy_shapes():
    predictor = make_predictor(512, 256, 384)
    assert predictor(torch.randn(4, 512)).shape == (4, 384)

    synergy = SynergyHead(512, 512, 128)
    assert synergy(torch.randn(4, 512)).shape == (4, 128)
