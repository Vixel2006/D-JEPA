from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("matplotlib")

from djepa.config import FigureConfig
from djepa.diagnostics import (
    Diagnostics,
    compute_alignment_pairs,
    compute_calibration,
    compute_cka_pairs,
    compute_spectral_diagnostics,
    confusion_matrix,
)
from djepa.evaluation.probes import (
    DJEPA_C,
    DJEPA_C_U1,
    DJEPA_C_U2,
    DJEPA_C_U1_U2,
    DJEPA_FULL,
    LEJEPA,
)
from djepa.figures import FigureData, generate_all_figures
from djepa.results import PIDRow


def test_figure_suite_writes_artifacts(tmp_path: Path):
    torch.manual_seed(0)
    num = 20
    dim = 8
    representations = {
        "z1": torch.randn(num, dim),
        "z2": torch.randn(num, dim),
        "c1": torch.randn(num, dim),
        "c2": torch.randn(num, dim),
        "c": torch.randn(num, dim),
        "u1": torch.randn(num, dim),
        "u2": torch.randn(num, dim),
        "s": torch.randn(num, dim),
    }
    names = [LEJEPA, DJEPA_C, DJEPA_C_U1, DJEPA_C_U2,
             DJEPA_C_U1_U2, DJEPA_FULL]
    accuracy = {
        name: {
            category: (50.0 + index, 1.0)
            for index, category in enumerate(
                ("counting", "attribute", "spatial_relation", "overall")
            )
        }
        for name in names
    }
    categories = ["counting", "attribute", "spatial_relation"]
    records = []
    for index in range(num):
        category = categories[index % len(categories)]
        records.append(
            {
                "idx": index,
                "row": index,
                "category": category,
                "question": f"question {index}",
                "ground_truth": "2",
                "lejepa": "1" if index % 2 else "2",
                "c_only": "2",
                "u1_only": "2",
                "u2_only": "1",
                "both_u1_u2": "2",
                "full": "2",
                "is_win": index % 2 == 1,
                "is_loss": index % 4 == 0,
            }
        )
    logits = {
        LEJEPA: torch.randn(num, 4),
        DJEPA_C: torch.randn(num, 4),
        DJEPA_C_U1_U2: torch.randn(num, 4),
        DJEPA_FULL: torch.randn(num, 4),
    }
    labels = torch.arange(num) % 4
    diagnostics = Diagnostics(
        spectral=compute_spectral_diagnostics(representations),
        cka=compute_cka_pairs(
            representations,
            [("c1", "c2"), ("u1", "u2"), ("c1", "u1")],
        ),
        alignment=compute_alignment_pairs(
            representations, [("c1", "c2"), ("u1", "u2")]
        ),
        calibration=compute_calibration(logits, labels, 4),
        swaps={
            "u1": {**{c: 10.0 for c in categories}, "overall": 10.0},
            "u2": {**{c: 5.0 for c in categories}, "overall": 5.0},
        },
        confusion={
            LEJEPA: confusion_matrix(
                logits[LEJEPA].argmax(-1), labels, 4
            ),
            DJEPA_FULL: confusion_matrix(
                logits[DJEPA_FULL].argmax(-1), labels, 4
            ),
        },
    )
    data = FigureData(
        accuracy=accuracy,
        categories=categories,
        subspace_names=names,
        embeddings={
            "c": np.random.randn(num, 2 * dim),
            "c+u1+u2": np.random.randn(num, 4 * dim),
            "full": np.random.randn(num, 5 * dim),
        },
        representations=representations,
        records=records,
        sample_images={index: np.zeros((8, 8, 3)) for index in range(num)},
        diagnostics=diagnostics,
        loss_history={
            "step": [1, 2, 3],
            "loss_lejepa": [3.0, 2.5, 2.2],
            "loss_djepa": [4.0, 3.0, 2.5],
            "djepa_cross": [1.0, 0.8, 0.6],
        },
        seeds=(42,),
        answer_vocabulary=["1", "2", "3", "4"],
    )
    pid_rows = [
        PIDRow("u1", 0.5, 0.1, 0.6, 0.2, 0.0, 0.1, 0.3, "Unq1"),
        PIDRow("s", 0.2, 0.2, 0.5, 0.0, 0.0, 0.2, 0.3, "Syn"),
    ]

    paths = generate_all_figures(
        data, tmp_path, FigureConfig(save_pdf=False, tsne_max_iter=250),
        pid_rows=pid_rows,
    )

    assert len(paths) >= 8
    assert (tmp_path / "fig01_probe_accuracy_heatmap.png").exists()
    assert (tmp_path / "fig07_spectral_diagnostics.png").exists()
    assert (tmp_path / "fig10_pid_decomposition.png").exists()
