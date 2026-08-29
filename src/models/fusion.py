from dataclasses import dataclass
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import chex
from .components import TransformerBlock


@dataclass
class MultimodalFusionConfig:
    vit_d_model: int = 768
    bert_d_model: int = 768
    joint_d_model: int = 512
    num_heads: int = 8
    num_layers: int = 2
    mlp_ratio: float = 4.0
    use_rope: bool = True
    rope_base: float = 10000.0
    max_seq_len: int = 2048
    dropout: float = 0.1


class MultimodalFusion(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        config: MultimodalFusionConfig | None = None,
        **kwargs,
    ):
        if config is None:
            config = MultimodalFusionConfig(**kwargs)
        else:
            assert not kwargs, "Cannot specify both config and kwargs"

        assert (
            config.joint_d_model % config.num_heads == 0
        ), f"joint_d_model {config.joint_d_model} must be divisible by num_heads {config.num_heads}"

        self.config = config
        self.vit_d_model = config.vit_d_model
        self.bert_d_model = config.bert_d_model
        self.joint_d_model = config.joint_d_model
        self.num_layers = config.num_layers
        self.dropout = config.dropout

        self.vit_proj = nnx.Linear(config.vit_d_model, config.joint_d_model, rngs=rngs)
        self.bert_proj = nnx.Linear(
            config.bert_d_model, config.joint_d_model, rngs=rngs
        )

        self.pos_embed = nnx.Variable(
            jax.random.normal(rngs.params(), (1, 2, config.joint_d_model))
        )

        self.blocks = nnx.List(
            [
                TransformerBlock(
                    rngs=rngs,
                    d_model=config.joint_d_model,
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

        self.norm = nnx.LayerNorm(config.joint_d_model, rngs=rngs)
        self.out_proj = nnx.Linear(
            config.joint_d_model, config.joint_d_model, rngs=rngs
        )

    def __call__(
        self,
        vit_latent: jax.Array,
        bert_latent: jax.Array,
    ) -> jax.Array:
        chex.assert_rank(vit_latent, 2)
        chex.assert_rank(bert_latent, 2)

        batch_size_vit, vit_d_model = vit_latent.shape
        batch_size_bert, bert_d_model = bert_latent.shape

        chex.assert_equal(batch_size_vit, batch_size_bert)
        chex.assert_equal(vit_d_model, self.vit_d_model)
        chex.assert_equal(bert_d_model, self.bert_d_model)

        batch_size = batch_size_vit

        vit_proj = self.vit_proj(vit_latent)
        bert_proj = self.bert_proj(bert_latent)

        chex.assert_shape(vit_proj, (batch_size, self.joint_d_model))
        chex.assert_shape(bert_proj, (batch_size, self.joint_d_model))

        x = jnp.stack([vit_proj, bert_proj], axis=1)
        chex.assert_shape(x, (batch_size, 2, self.joint_d_model))

        x = x + self.pos_embed.value
        chex.assert_shape(x, (batch_size, 2, self.joint_d_model))

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        chex.assert_shape(x, (batch_size, 2, self.joint_d_model))

        x = x.mean(axis=1)
        chex.assert_shape(x, (batch_size, self.joint_d_model))

        joint_latent = self.out_proj(x)
        chex.assert_shape(joint_latent, (batch_size, self.joint_d_model))

        return joint_latent
