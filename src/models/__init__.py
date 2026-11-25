"""Small Flax encoders and projection heads."""

from .audio import AudioEncoder
from .projection import ProjectionHead
from .djepa import DJEPA
from .sigreg import SIGReg
from .text import TextEncoder
from .transformer import TransformerEncoder
from .vision import VisionEncoder

__all__ = [
    "AudioEncoder",
    "DJEPA",
    "ProjectionHead",
    "SIGReg",
    "TextEncoder",
    "TransformerEncoder",
    "VisionEncoder",
]
