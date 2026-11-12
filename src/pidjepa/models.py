"""Representation models, losses, and the experimental-arm registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from pidjepa.data import Bundle


class SIGReg(nn.Module):
    """Sliced Epps–Pulley regularizer for non-contrastive representation learning.

    The regularizer encourages each one-dimensional projection of a batch to
    match a standard Gaussian characteristic function.  It is the anti-collapse
    term used by both the alignment and D-JEPA arms.
    """

    def __init__(self, n_slices: int = 256, max_frequency: float = 3.0, n_points: int = 17):
        super().__init__()
        if n_points % 2 == 0:
            raise ValueError("n_points must be odd")
        frequencies = torch.linspace(0.0, max_frequency, n_points)
        step = max_frequency / (n_points - 1)
        characteristic_function = (-0.5 * frequencies**2).exp()
        quadrature_weights = torch.full((n_points,), 2 * step)
        quadrature_weights[[0, -1]] = step
        self.register_buffer("frequencies", frequencies)
        self.register_buffer("target", characteristic_function)
        self.register_buffer("weights", quadrature_weights * characteristic_function)
        self.n_slices = n_slices

    def forward(self, representations: torch.Tensor) -> torch.Tensor:
        representations = representations.float()
        batch_size = representations.size(0)
        directions = torch.randn(
            representations.size(1), self.n_slices, device=representations.device
        )
        directions = directions / directions.norm(dim=0, keepdim=True)
        projected = (representations @ directions).unsqueeze(-1) * self.frequencies
        error = (projected.cos().mean(0) - self.target).square() + projected.sin().mean(0).square()
        return ((error @ self.weights) * batch_size).mean()


def cross_correlation_loss(
    private: torch.Tensor, shared: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """Penalize linear correlation between a private fibre and a shared code."""
    batch_size = private.size(0)
    private = (private - private.mean(0)) / (private.std(0) + eps)
    shared = (shared - shared.mean(0)) / (shared.std(0) + eps)
    correlation = private.T @ shared / (batch_size - 1)
    return correlation.pow(2).sum() / max(private.size(1), shared.size(1))


def make_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    """Two-hidden-layer MLP used by vector encoders and predictors."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, out_dim),
    )


def pool_tokens(tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Mean-pool a token sequence, ignoring padded positions when a mask is given."""
    tokens = tokens.float()
    if mask is None:
        return tokens.mean(1)
    weights = mask.unsqueeze(-1).float()
    return (tokens * weights).sum(1) / weights.sum(1).clamp_min(1.0)


class VectorEncoder(nn.Module):
    """MLP encoder for pre-computed vector features (e.g. synthetic data)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.network = make_mlp(in_dim, hidden_dim, out_dim)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        del mask  # Vector features are already pooled.
        return self.network(inputs.float())


class TokenEncoder(nn.Module):
    """Transformer encoder for token sequences such as BERT/DINOv2 outputs."""

    def __init__(
        self,
        in_dim: int,
        n_tokens: int,
        model_dim: int,
        n_layers: int,
        n_heads: int,
        hidden_dim: int,
        out_dim: int,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(in_dim)
        self.input_projection = nn.Linear(in_dim, model_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.position_embedding = nn.Parameter(torch.randn(1, n_tokens + 1, model_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            model_dim,
            n_heads,
            dim_feedforward=2 * model_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, n_layers, enable_nested_tensor=False
        )
        self.head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        tokens = self.input_projection(self.input_norm(tokens.float()))
        batch_size, token_count = tokens.shape[:2]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        tokens = tokens + self.position_embedding[:, : token_count + 1]

        padding_mask = None
        if mask is not None:
            cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=tokens.device)
            padding_mask = torch.cat([cls_mask, ~mask], dim=1)

        hidden = self.transformer(tokens, src_key_padding_mask=padding_mask)
        return self.head(hidden[:, 0])


class MultimodalModel(nn.Module):
    """Two-stream encoder with an optional shared/private representation split.

    ``architecture="alignment"`` produces a single joint embedding per stream
    (``z1``/``z2``).  ``architecture="djepa"`` splits each stream into a shared
    code (``c1``/``c2``) and a private fibre (``u1``/``u2``).
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        text_encoder: nn.Module,
        block_dim: int,
        hidden_dim: int,
        architecture: str,
        uses_reconstruction: bool,
        vision_target_dim: int,
        text_target_dim: int,
    ):
        super().__init__()
        self.architecture = architecture
        self.block_dim = block_dim
        self.uses_reconstruction = uses_reconstruction
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        if uses_reconstruction:
            self.vision_predictor = make_mlp(2 * block_dim, hidden_dim, vision_target_dim)
            self.text_predictor = make_mlp(2 * block_dim, hidden_dim, text_target_dim)

    def split_representation(
        self, representation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split an embedding into ``(shared code, private fibre)``."""
        return representation[:, : self.block_dim], representation[:, self.block_dim :]

    @staticmethod
    def normalized_target(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Layer-normalize a pooled target used by the reconstruction objective."""
        target = values.float() if values.dim() == 2 else pool_tokens(values, mask)
        return F.layer_norm(target, (target.size(-1),))

    @torch.no_grad()
    def encode_features(
        self,
        vision_inputs: torch.Tensor,
        text_inputs: torch.Tensor,
        text_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return the named representation blocks used by downstream probes."""
        vision_representation = self.vision_encoder(vision_inputs)
        text_representation = self.text_encoder(text_inputs, text_mask)
        if self.architecture == "alignment":
            return {"z1": vision_representation, "z2": text_representation}

        vision_shared, vision_private = self.split_representation(vision_representation)
        text_shared, text_private = self.split_representation(text_representation)
        return {
            "c1": vision_shared,
            "u1": vision_private,
            "c2": text_shared,
            "u2": text_private,
        }


# =============================================================================
# Experimental-arm registry
# =============================================================================
ARM_CONFIGS = {
    "alignment": {"architecture": "alignment", "reconstruct": False, "independence": False},
    "alignment_rec": {"architecture": "alignment", "reconstruct": True, "independence": False},
    "djepa": {"architecture": "djepa", "reconstruct": True, "independence": True},
    "djepa_noindep": {"architecture": "djepa", "reconstruct": True, "independence": False},
    "djepa_norec": {"architecture": "djepa", "reconstruct": False, "independence": True},
}

#: Named feature groups used by the capacity-matched probes.
FEATURE_SETS = {
    "raw": {"v": ["h1"], "t": ["h2"], "joint": ["h1", "h2"]},
    "alignment": {"v": ["z1"], "t": ["z2"], "joint": ["z1", "z2"]},
    "djepa": {
        "v": ["c1", "u1"],
        "t": ["c2", "u2"],
        "joint": ["c1", "c2", "u1", "u2"],
        "c": ["c1", "c2"],
        "u": ["u1", "u2"],
        "c1": ["c1"],
        "c2": ["c2"],
        "u1": ["u1"],
        "u2": ["u2"],
    },
}

#: Source/target triples for the PID estimators: (name, source A, source B, joint).
PID_STREAM_PAIRS = {
    "raw": [("stream", "v", "t", "joint")],
    "alignment": [("stream", "v", "t", "joint")],
    "djepa": [
        ("stream", "v", "t", "joint"),
        ("c-pair", "c1", "c2", "c"),
        ("u-pair", "u1", "u2", "u"),
    ],
}

#: Qualitative expectations for the synthetic ground-truth benchmark.
EXPECTED_COMPONENTS = {
    "shared": {"stream": "R", "c-pair": "R", "u-pair": "none"},
    "priv_v": {"stream": "U1", "c-pair": "none", "u-pair": "U1"},
    "priv_t": {"stream": "U2", "c-pair": "none", "u-pair": "U2"},
    "synergy": {"stream": "S", "c-pair": "none", "u-pair": "S"},
    "product": {"stream": "S", "c-pair": "none", "u-pair": "S"},
}


def arm_family(arm: str) -> str:
    """Map a concrete arm name to its feature-set family."""
    return "raw" if arm == "raw" else ARM_CONFIGS[arm]["architecture"]


def build_model(bundle: Bundle, config: dict, args) -> MultimodalModel:
    """Construct the two-stream model for one experimental arm."""
    block_dim, hidden_dim = args.block_dim, args.hidden_dim
    if bundle.kind == "vec":
        vision_dim, text_dim = bundle.vision_views.size(-1), bundle.text_views.size(-1)
        vision_encoder = VectorEncoder(vision_dim, hidden_dim, 2 * block_dim)
        text_encoder = VectorEncoder(text_dim, hidden_dim, 2 * block_dim)
        vision_target_dim, text_target_dim = vision_dim, text_dim
    else:
        _, _, vision_tokens, vision_dim = bundle.vision_views.shape
        _, _, text_tokens, text_dim = bundle.text_views.shape
        encoder_kwargs = dict(
            model_dim=args.transformer_dim,
            n_layers=args.transformer_layers,
            n_heads=args.attention_heads,
            hidden_dim=hidden_dim,
            out_dim=2 * block_dim,
        )
        vision_encoder = TokenEncoder(vision_dim, vision_tokens, **encoder_kwargs)
        text_encoder = TokenEncoder(text_dim, text_tokens, **encoder_kwargs)
        vision_target_dim, text_target_dim = vision_dim, text_dim

    return MultimodalModel(
        vision_encoder=vision_encoder,
        text_encoder=text_encoder,
        block_dim=block_dim,
        hidden_dim=hidden_dim,
        architecture=config["architecture"],
        uses_reconstruction=config["reconstruct"],
        vision_target_dim=vision_target_dim,
        text_target_dim=text_target_dim,
    )
