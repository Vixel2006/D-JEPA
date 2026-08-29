from dataclasses import dataclass
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import chex
from .components import TransformerBlock


@dataclass
class TextDecoderConfig:
    joint_d_model: int = 512
    vocab_size: int = 30522
    max_seq_len: int = 512
    d_model: int = 768
    num_layers: int = 4
    num_heads: int = 12
    mlp_ratio: float = 4.0
    use_rope: bool = True
    rope_base: float = 10000.0
    dropout: float = 0.0


class TextDecoder(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        config: TextDecoderConfig | None = None,
        **kwargs,
    ):
        if config is None:
            config = TextDecoderConfig(**kwargs)
        else:
            assert not kwargs, "Cannot specify both config and kwargs"

        assert (
            config.d_model % config.num_heads == 0
        ), f"d_model {config.d_model} must be divisible by num_heads {config.num_heads}"

        self.config = config
        self.joint_d_model = config.joint_d_model
        self.vocab_size = config.vocab_size
        self.max_seq_len = config.max_seq_len
        self.d_model = config.d_model
        self.num_layers = config.num_layers

        self.latent_proj = nnx.Linear(config.joint_d_model, config.d_model, rngs=rngs)

        self.token_embedding = nnx.Embed(config.vocab_size, config.d_model, rngs=rngs)
        self.pos_embedding = nnx.Variable(
            jax.random.normal(rngs.params(), (1, config.max_seq_len, config.d_model))
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
        self.lm_head = nnx.Linear(
            config.d_model, config.vocab_size, rngs=rngs, use_bias=False
        )

    def __call__(
        self,
        joint_latent: jax.Array,
        input_ids: jax.Array,
        attention_mask: jax.Array | None = None,
    ) -> jax.Array:
        chex.assert_rank(joint_latent, 2)
        chex.assert_rank(input_ids, 2)

        batch_size, joint_d_model = joint_latent.shape
        batch_size_ids, seq_len = input_ids.shape

        chex.assert_equal(batch_size, batch_size_ids)
        chex.assert_equal(joint_d_model, self.joint_d_model)
        assert (
            seq_len <= self.max_seq_len
        ), f"seq_len {seq_len} > max_seq_len {self.max_seq_len}"

        if attention_mask is not None:
            chex.assert_shape(attention_mask, (batch_size, seq_len))

        latent_proj = self.latent_proj(joint_latent)
        chex.assert_shape(latent_proj, (batch_size, self.d_model))

        token_emb = self.token_embedding(input_ids)
        chex.assert_shape(token_emb, (batch_size, seq_len, self.d_model))

        token_emb = token_emb + self.pos_embedding.value[:, :seq_len]
        chex.assert_shape(token_emb, (batch_size, seq_len, self.d_model))

        latent_expanded = jnp.broadcast_to(
            latent_proj[:, None, :], (batch_size, seq_len, self.d_model)
        )
        chex.assert_shape(latent_expanded, (batch_size, seq_len, self.d_model))

        x = token_emb + latent_expanded
        chex.assert_shape(x, (batch_size, seq_len, self.d_model))

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        chex.assert_shape(x, (batch_size, seq_len, self.d_model))

        logits = self.lm_head(x)
        chex.assert_shape(logits, (batch_size, seq_len, self.vocab_size))

        return logits