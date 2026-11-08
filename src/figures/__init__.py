"""Figure generation for the D-JEPA CLEVR benchmark."""

from .paper import (
    CATEGORY_COLORS,
    CATEGORY_LABELS,
    FigureData,
    generate_all_figures,
    shorten_condition,
)

__all__ = [
    "CATEGORY_COLORS",
    "CATEGORY_LABELS",
    "FigureData",
    "generate_all_figures",
    "shorten_condition",
]
