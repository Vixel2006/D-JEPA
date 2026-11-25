"""A small paired dataset with known shared and modality-private factors."""

from __future__ import annotations

import numpy as np

from ..dataset import PreparedDataset, make_split


def make_synthetic_dataset(
    n_samples: int = 1024,
    seed: int = 0,
    modalities: tuple[str, ...] = ("vision", "text", "audio"),
) -> PreparedDataset:
    """Generate two noisy views per modality and continuous probe targets."""
    rng = np.random.default_rng(seed)
    shared = rng.normal(size=(n_samples, 4)).astype(np.float32)
    private = {
        name: rng.normal(size=(n_samples, 4)).astype(np.float32) for name in modalities
    }
    context, target = {}, {}

    for name in modalities:
        latent = np.concatenate((shared, private[name]), axis=-1)
        if name == "text":
            # Each sequence position represents a quantized latent coordinate.
            coordinates = np.tile(np.arange(8), 2)

            def make_text_view():
                bins = np.clip(
                    ((np.tanh(latent[:, coordinates]) + 1.0) * 16).astype(np.int32), 0, 31
                )
                return bins + coordinates[None, :] * 32

        elif name == "vision":
            projection = rng.normal(size=(8, 3)).astype(np.float32) / np.sqrt(8)

            def make_vision_view():
                pixels = np.tanh(latent @ projection)
                image = np.broadcast_to(pixels[:, None, None, :], (n_samples, 32, 32, 3))
                return (image + 0.1 * rng.normal(size=image.shape)).astype(np.float32)

        elif name == "audio":
            projection = rng.normal(size=(8, 8)).astype(np.float32) / np.sqrt(8)

            def make_audio_view():
                frames = np.tanh(latent @ projection)[:, None, :]
                frames = np.broadcast_to(frames, (n_samples, 32, 8))
                return (frames + 0.1 * rng.normal(size=frames.shape)).astype(np.float32)

        else:
            raise ValueError(f"No synthetic input generator for modality {name!r}")

        if name == "text":
            context[name], target[name] = make_text_view(), make_text_view()
        elif name == "vision":
            context[name], target[name] = make_vision_view(), make_vision_view()
        else:
            context[name], target[name] = make_audio_view(), make_audio_view()

    factors = {"shared": shared, **{f"private_{name}": values for name, values in private.items()}}
    train_indices, test_indices = make_split(n_samples, seed)
    return PreparedDataset(context, target, factors, train_indices, test_indices)
