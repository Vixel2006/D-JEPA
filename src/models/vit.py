from dataclasses import dataclass
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import chex
from .components import TransformerBlock


@dataclass
class ViTConfig:
    img_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    num_classes: int = 1000
    d_model: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    use_rope: bool = True
    rope_base: float = 10000.0
    use_slots: bool = False
    num_slots: int = 7
    d_slot: int = 64
    slot_iterations: int = 3
    dropout: float = 0.0


class ViT(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        config: ViTConfig | None = None,
        **kwargs,
    ):
        if config is None:
            config = ViTConfig(**kwargs)
        else:
            assert not kwargs, "Cannot specify both config and kwargs"

        chex.assert_scalar_positive(config.img_size)
        chex.assert_scalar_positive(config.patch_size)
        chex.assert_scalar_positive(config.in_channels)
        chex.assert_scalar_positive(config.num_classes)
        chex.assert_scalar_positive(config.d_model)
        chex.assert_scalar_positive(config.num_layers)
        chex.assert_scalar_positive(config.num_heads)
        assert (
            config.d_model % config.num_heads == 0
        ), f"d_model {config.d_model} must be divisible by num_heads {config.num_heads}"
        assert (
            config.img_size % config.patch_size == 0
        ), f"img_size {config.img_size} must be divisible by patch_size {config.patch_size}"

        self.config = config
        self.img_size = config.img_size
        self.patch_size = config.patch_size
        self.num_patches = (config.img_size // config.patch_size) ** 2
        self.d_model = config.d_model
        self.num_layers = config.num_layers
        self.num_classes = config.num_classes
        self.use_slots = config.use_slots

        self.patch_embed = nnx.Conv(
            config.in_channels,
            config.d_model,
            kernel_size=(config.patch_size, config.patch_size),
            strides=(config.patch_size, config.patch_size),
            rngs=rngs,
        )

        if config.use_slots:
            seq_len = config.num_slots
        else:
            seq_len = self.num_patches + 1

        self.cls_token = nnx.Variable(
            jax.random.normal(rngs.params(), (1, 1, config.d_model))
        )

        max_seq_len = max(seq_len, 2048)
        self.pos_embed = nnx.Variable(
            jax.random.normal(rngs.params(), (1, max_seq_len, config.d_model))
        )

        self.blocks = nnx.List(
            [
                TransformerBlock(
                    rngs=rngs,
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    use_rope=config.use_rope,
                    max_seq_len=max_seq_len,
                    rope_base=config.rope_base,
                    use_slots=config.use_slots,
                    num_slots=config.num_slots,
                    d_slot=config.d_slot,
                    slot_iterations=config.slot_iterations,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.norm = nnx.BatchNorm(config.d_model, rngs=rngs)
        self.head = nnx.Linear(config.d_model, config.num_classes, rngs=rngs)
        self.dropout = config.dropout

    def __call__(self, x: jax.Array) -> jax.Array:
        chex.assert_rank(x, 4)
        batch_size, h, w, c = x.shape
        chex.assert_equal(h, self.img_size)
        chex.assert_equal(w, self.img_size)

        x = self.patch_embed(x)
        chex.assert_shape(
            x,
            (
                batch_size,
                self.img_size // self.patch_size,
                self.img_size // self.patch_size,
                self.d_model,
            ),
        )

        x = x.reshape(batch_size, self.num_patches, self.d_model)
        chex.assert_shape(x, (batch_size, self.num_patches, self.d_model))

        cls_token = jnp.broadcast_to(
            self.cls_token.value, (batch_size, 1, self.d_model)
        )
        x = jnp.concatenate([cls_token, x], axis=1)
        chex.assert_shape(x, (batch_size, self.num_patches + 1, self.d_model))

        seq_len = x.shape[1]
        x = x + self.pos_embed.value[:, :seq_len]
        chex.assert_shape(x, (batch_size, seq_len, self.d_model))

        for block in self.blocks:
            x = block(x)

        if self.use_slots:
            x = x.mean(
                axis=1
            )
        else:
            x = x[:, 0]

        x = self.norm(x)
        x = self.head(x)
        chex.assert_shape(x, (batch_size, self.num_classes))
        return x