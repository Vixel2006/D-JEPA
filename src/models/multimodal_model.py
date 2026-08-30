import jax
import jax.numpy as jnp
import flax.nnx as nnx
from .vit import ViT, ViTConfig
from .bert import BERT, BERTConfig
from .fusion import MultimodalFusion, MultimodalFusionConfig
from .image_decoder import ImageDecoder, ImageDecoderConfig
from .text_decoder import TextDecoder, TextDecoderConfig


class MultimodalModel(nnx.Module):
    """Combined ViT + BERT + Fusion + Decoders model for training with reconstruction."""

    def __init__(
        self,
        rngs: nnx.Rngs,
        vit_config: ViTConfig | None = None,
        bert_config: BERTConfig | None = None,
        fusion_config: MultimodalFusionConfig | None = None,
        image_decoder_config: ImageDecoderConfig | None = None,
        text_decoder_config: TextDecoderConfig | None = None,
    ):
        if vit_config is None:
            vit_config = ViTConfig()
        if bert_config is None:
            bert_config = BERTConfig()
        if fusion_config is None:
            fusion_config = MultimodalFusionConfig()
        if image_decoder_config is None:
            image_decoder_config = ImageDecoderConfig()
        if text_decoder_config is None:
            text_decoder_config = TextDecoderConfig()

        self.vit = ViT(rngs, vit_config)
        self.bert = BERT(rngs, bert_config)
        self.fusion = MultimodalFusion(rngs, fusion_config)
        self.image_decoder = ImageDecoder(rngs, image_decoder_config)
        self.text_decoder = TextDecoder(rngs, text_decoder_config)

    def __call__(
        self,
        images: jax.Array,
        input_ids: jax.Array,
        attention_mask: jax.Array | None = None,
    ) -> tuple:
        """Forward pass returning latents and reconstructions."""
        vit_latent = self.vit.encode(images)
        bert_latent = self.bert.encode(input_ids)
        joint_latent = self.fusion(vit_latent, bert_latent)

        recon_image = self.image_decoder(joint_latent)
        recon_text_logits = self.text_decoder(joint_latent, input_ids, attention_mask)

        return vit_latent, bert_latent, joint_latent, recon_image, recon_text_logits

    def encode_vit(self, images: jax.Array) -> jax.Array:
        return self.vit.encode(images)

    def encode_bert(self, input_ids: jax.Array) -> jax.Array:
        return self.bert.encode(input_ids)

    def decode_image(self, joint_latent: jax.Array) -> jax.Array:
        return self.image_decoder(joint_latent)

    def decode_text(
        self,
        joint_latent: jax.Array,
        input_ids: jax.Array,
        attention_mask: jax.Array | None = None,
    ) -> jax.Array:
        return self.text_decoder(joint_latent, input_ids, attention_mask)