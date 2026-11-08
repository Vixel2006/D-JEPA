"""D-JEPA CLEVR benchmark.

A clean, modular implementation of the CLEVR experiment described in the
accompanying manuscript.  The package is intentionally import-light: heavy
backbones (DINOv2, DistilBERT) are only loaded when a model is instantiated.
"""

from .config import (
    DataConfig,
    FigureConfig,
    ModelConfig,
    OptimizationConfig,
    RunConfig,
)
from .diagnostics import Diagnostics

__all__ = [
    "DataConfig",
    "Diagnostics",
    "FigureConfig",
    "ModelConfig",
    "OptimizationConfig",
    "RunConfig",
]
