"""JAX loss functions for representation learning."""

from .representation import (
    loss_alignment,
    loss_covariance,
    loss_cross_covariance,
    loss_sigreg,
)

__all__ = ["loss_alignment", "loss_covariance", "loss_cross_covariance", "loss_sigreg"]
