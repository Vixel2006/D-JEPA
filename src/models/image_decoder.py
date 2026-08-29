from dataclasses import dataclass
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import chex
from .components import TransformerBlock


@dataclass
class ImageDecoderConfig:
    joint_d_model: int = 512
    img_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    d_model: int = 768
    num_layers: int = 4
    num_heads: int = 12
    mlp_ratio: float = 4.0
    use_rope: bool = True
    rope_base: float = 10000.0
    max_seq_len: int = 2048
    dropout: float = 0.0


class ImageDecoder(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        config: ImageDecoderConfig | None = None,
        **kwargs,
    ):
        if config is None:
            config = ImageDecoderConfig(**kwargs)
        else:
            assert not kwargs, "Cannot specify both config and kwargs"

        chex.assert_scalar_positive(config.joint_d_model)
        chex.assert_scalar_positive(config.img_size)
        chex.assert_scalar_positive(config.patch_size)
        chex.assert_scalar_positive(config.in_channels)
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
        self.joint_d_model = config.joint_d_model
        self.img_size = config.img_size
        self.patch_size = config.patch_size
        self.in_channels = config.in_channels
        self.d_model = config.d_model
        self.num_layers = config.num_layers
        self.num_patches = (config.img_size // config.patch_size) ** 2

        self.latent_proj = nnx.Linear(config.joint_d_model, config.d_model, rngs=rngs)

        self.patch_pos_embed = nnx.Variable(
            jax.random.normal(rngs.params(), (1, self.num_patches, config.d_model))
        )

        self.blocks = nnx.List(
            [
                TransformerBlock(
                    rngs=rngs,
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    use_rope=config.use_rope,
                    max_seq_len=config.max_seq_len,
                    rope_base=config.rope_base,
                    use_slots=False,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.norm = nnx.LayerNorm(config.d_model, rngs=rngs)
        self.patch_pred = nnx.Linear(
            config.d_model,
            config.patch_size * config.patch_size * config.in_channels,
            rngs=rngs,
        )

    def __call__(self, joint_latent: jax.Array) -> jax.Array:
        chex.assert_rank(joint_latent, 2)
        batch_size, joint_d_model = joint_latent.shape
        chex.assert_equal(joint_d_model, self.joint_d_model)

        x = self.latent_proj(joint_latent)
        chex.assert_shape(x, (batch_size, self.d_model))

        x = jnp.broadcast_to(
            x[:, None, :], (batch_size, self.num_patches, self.d_model)
        )
        chex.assert_shape(x, (batch_size, self.num_patches, self.d_model))

        x = x + self.patch_pos_embed.value
        chex.assert_shape(x, (batch_size, self.num_patches, self.d_model))

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        chex.assert_shape(x, (batch_size, self.num_patches, self.d_model))

        patches = self.patch_pred(x)
        chex.assert_shape(
            patches,
            (
                batch_size,
                self.num_patches,
                self.patch_size * self.patch_size * self.in_channels,
            ),
        )

        patches = patches.reshape(
            batch_size,
            self.img_size // self.patch_size,
            self.img_size // self.patch_size,
            self.patch_size,
            self.patch_size,
            self.in_channels,
        )
        patches = patches.transpose(0, 1, 3, 2, 4, 5)
        img = patches.reshape(
            batch_size, self.img_size, self.img_size, self.in_channels
        )

        chex.assert_shape(
            img, (batch_size, self.img_size, self.img_size, self.in_channels)
        )
        return img