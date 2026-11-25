"""Shared-weight LeJEPA encoder for paired multimodal views."""

from typing import Any

import jax
from flax import linen as nn

from .projection import ProjectionHead


def _run_encoder(encoder: nn.Module, inputs: Any) -> jax.Array:
    if isinstance(inputs, dict):
        return encoder(**inputs)
    if isinstance(inputs, tuple):
        return encoder(*inputs)
    return encoder(inputs)


class DJEPA(nn.Module):
    """Encode context and target views with the same parameters.

    ``encoders`` maps modality names to encoder modules. Inputs are dictionaries
    with one batch per modality, with corresponding examples in the same order.
    """

    encoders: dict[str, nn.Module]
    projection_dim: int = 128

    def setup(self) -> None:
        self.modalities = tuple(self.encoders)
        self.online_encoders = {
            name: encoder.clone(name=f"online_{name}_encoder")
            for name, encoder in self.encoders.items()
        }
        self.online_unique = {
            name: ProjectionHead(self.projection_dim, name=f"online_{name}_unique")
            for name in self.modalities
        }
        self.online_redundant = {
            name: ProjectionHead(self.projection_dim, name=f"online_{name}_redundant")
            for name in self.modalities
        }

    def __call__(
        self, context: dict[str, Any], target: dict[str, Any]
    ) -> dict[str, jax.Array | dict[str, jax.Array]]:
        context_encoded = {
            name: _run_encoder(self.online_encoders[name], context[name])
            for name in self.modalities
        }
        context_unique = {
            name: self.online_unique[name](context_encoded[name])
            for name in self.modalities
        }
        context_redundant = {
            name: self.online_redundant[name](context_encoded[name])
            for name in self.modalities
        }

        target_encoded = {
            name: _run_encoder(self.online_encoders[name], target[name])
            for name in self.modalities
        }
        target_unique = {
            name: self.online_unique[name](target_encoded[name])
            for name in self.modalities
        }
        target_redundant = {
            name: self.online_redundant[name](target_encoded[name])
            for name in self.modalities
        }
        return {
            "context_unique": context_unique,
            "context_redundant": context_redundant,
            "target_unique": target_unique,
            "target_redundant": target_redundant,
        }
