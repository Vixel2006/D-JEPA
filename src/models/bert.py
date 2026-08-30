from dataclasses import dataclass
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import chex
from .components import TransformerBlock


@dataclass
class BERTConfig:
    vocab_size: int = 30522
    max_seq_len: int = 512
    num_classes: int = 2
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


class BERT(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        config: BERTConfig | None = None,
        **kwargs,
    ):
        if config is None:
            config = BERTConfig(**kwargs)
        else:
            assert not kwargs, "Cannot specify both config and kwargs"

        assert (
            config.d_model % config.num_heads == 0
        ), f"d_model {config.d_model} must be divisible by num_heads {config.num_heads}"

        self.config = config
        self.vocab_size = config.vocab_size
        self.max_seq_len = config.max_seq_len
        self.num_classes = config.num_classes
        self.d_model = config.d_model
        self.num_layers = config.num_layers
        self.use_slots = config.use_slots

        self.token_embedding = nnx.Embed(config.vocab_size, config.d_model, rngs=rngs)
        self.pos_embedding = nnx.Variable(
            jax.random.normal(rngs.params(), (1, config.max_seq_len, config.d_model))
        )

        if config.use_slots:
            seq_len = config.num_slots
        else:
            seq_len = config.max_seq_len

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
                    use_slots=config.use_slots,
                    num_slots=config.num_slots,
                    d_slot=config.d_slot,
                    slot_iterations=config.slot_iterations,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.norm = nnx.LayerNorm(config.d_model, rngs=rngs)
        self.pooler = nnx.Linear(config.d_model, config.d_model, rngs=rngs)
        self.classifier = nnx.Linear(config.d_model, config.num_classes, rngs=rngs)
        self.dropout = config.dropout

    def __call__(
        self, input_ids: jax.Array, attention_mask: jax.Array | None = None
    ) -> jax.Array:
        chex.assert_rank(input_ids, 2)
        batch_size, seq_len = input_ids.shape
        assert (
            seq_len <= self.max_seq_len
        ), f"seq_len {seq_len} > max_seq_len {self.max_seq_len}"

        if attention_mask is not None:
            chex.assert_shape(attention_mask, (batch_size, seq_len))

        x = self.token_embedding(input_ids)
        chex.assert_shape(x, (batch_size, seq_len, self.d_model))

        x = x + self.pos_embedding.value[:, :seq_len]
        chex.assert_shape(x, (batch_size, seq_len, self.d_model))

        for block in self.blocks:
            x = block(x)

        if self.use_slots:
            x = x.mean(axis=1)
            chex.assert_shape(x, (batch_size, self.d_model))
        else:
            x = x[:, 0]
            chex.assert_shape(x, (batch_size, self.d_model))

        x = self.norm(x)
        x = self.pooler(x)
        x = jax.nn.gelu(x)
        x = self.classifier(x)
        chex.assert_shape(x, (batch_size, self.num_classes))
        return x

    def encode(
        self, input_ids: jax.Array, attention_mask: jax.Array | None = None
    ) -> jax.Array:
        """Encode text to latent representation (before classification head)."""
        chex.assert_rank(input_ids, 2)
        batch_size, seq_len = input_ids.shape
        assert (
            seq_len <= self.max_seq_len
        ), f"seq_len {seq_len} > max_seq_len {self.max_seq_len}"

        if attention_mask is not None:
            chex.assert_shape(attention_mask, (batch_size, seq_len))

        x = self.token_embedding(input_ids)
        chex.assert_shape(x, (batch_size, seq_len, self.d_model))

        x = x + self.pos_embedding.value[:, :seq_len]
        chex.assert_shape(x, (batch_size, seq_len, self.d_model))

        for block in self.blocks:
            x = block(x)

        if self.use_slots:
            x = x.mean(axis=1)
            chex.assert_shape(x, (batch_size, self.d_model))
        else:
            x = x[:, 0]
            chex.assert_shape(x, (batch_size, self.d_model))

        x = self.norm(x)
        chex.assert_shape(x, (batch_size, self.d_model))
        return x