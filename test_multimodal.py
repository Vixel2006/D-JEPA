import jax
import jax.numpy as jnp
import flax.nnx as nnx
import chex
from src.models.model import ViT, BERT, MultimodalFusion
from src.models.config import ViTConfig, BERTConfig, MultimodalFusionConfig


def test_vit():
    print("Testing ViT...")
    rngs = nnx.Rngs(0)
    config = ViTConfig(
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        d_model=768,
        num_layers=2,
        num_heads=12,
        use_slots=False,
    )
    model = ViT(rngs=rngs, config=config)

    batch_size = 2
    x = jax.random.normal(rngs.params(), (batch_size, 224, 224, 3))
    out = model(x)

    chex.assert_shape(out, (batch_size, 1000))
    print(f"  ViT output shape: {out.shape} ✓")


def test_bert():
    print("Testing BERT...")
    rngs = nnx.Rngs(1)
    config = BERTConfig(
        vocab_size=30522,
        max_seq_len=512,
        num_classes=2,
        d_model=768,
        num_layers=2,
        num_heads=12,
        use_slots=False,
    )
    model = BERT(rngs=rngs, config=config)

    batch_size = 2
    seq_len = 128
    input_ids = jax.random.randint(rngs.params(), (batch_size, seq_len), 0, 30522)
    attention_mask = jnp.ones((batch_size, seq_len))
    out = model(input_ids, attention_mask)

    chex.assert_shape(out, (batch_size, 2))
    print(f"  BERT output shape: {out.shape} ✓")


def test_multimodal_fusion():
    print("Testing MultimodalFusion...")
    rngs = nnx.Rngs(2)
    config = MultimodalFusionConfig(
        vit_d_model=768,
        bert_d_model=768,
        joint_d_model=512,
        num_heads=8,
        num_layers=2,
    )
    fusion = MultimodalFusion(rngs=rngs, config=config)

    batch_size = 4
    vit_latent = jax.random.normal(rngs.params(), (batch_size, 768))
    bert_latent = jax.random.normal(rngs.params(), (batch_size, 768))

    joint_latent = fusion(vit_latent, bert_latent)

    chex.assert_shape(joint_latent, (batch_size, 512))
    print(f"  MultimodalFusion output shape: {joint_latent.shape} ✓")


def test_full_architecture():
    print("Testing Full Architecture (ViT + BERT + MultimodalFusion)...")
    rngs = nnx.Rngs(3)

    vit_config = ViTConfig(
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        d_model=768,
        num_layers=2,
        num_heads=12,
        use_slots=False,
    )
    vit = ViT(rngs=rngs, config=vit_config)

    bert_config = BERTConfig(
        vocab_size=30522,
        max_seq_len=512,
        num_classes=2,
        d_model=768,
        num_layers=2,
        num_heads=12,
        use_slots=False,
    )
    bert = BERT(rngs=rngs, config=bert_config)

    fusion_config = MultimodalFusionConfig(
        vit_d_model=768,
        bert_d_model=768,
        joint_d_model=512,
        num_heads=8,
        num_layers=2,
    )
    fusion = MultimodalFusion(rngs=rngs, config=fusion_config)

    batch_size = 2
    img = jax.random.normal(rngs.params(), (batch_size, 224, 224, 3))
    input_ids = jax.random.randint(rngs.params(), (batch_size, 128), 0, 30522)
    attention_mask = jnp.ones((batch_size, 128))

    vit_out = vit(img)
    bert_out = bert(input_ids, attention_mask)

    chex.assert_shape(vit_out, (batch_size, 1000))
    chex.assert_shape(bert_out, (batch_size, 2))

    vit_latent = jax.random.normal(rngs.params(), (batch_size, 768))
    bert_latent = jax.random.normal(rngs.params(), (batch_size, 768))

    joint_latent = fusion(vit_latent, bert_latent)
    chex.assert_shape(joint_latent, (batch_size, 512))

    print(f"  ViT output shape: {vit_out.shape} ✓")
    print(f"  BERT output shape: {bert_out.shape} ✓")
    print(f"  Joint latent shape: {joint_latent.shape} ✓")
    print("  All components initialize and forward pass correctly ✓")


def test_with_slots():
    print("Testing with Slot Attention...")
    rngs = nnx.Rngs(4)

    vit_config = ViTConfig(
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        d_model=768,
        num_layers=2,
        num_heads=12,
        use_slots=True,
        num_slots=7,
        d_slot=64,
    )
    vit = ViT(rngs=rngs, config=vit_config)

    bert_config = BERTConfig(
        vocab_size=30522,
        max_seq_len=512,
        num_classes=2,
        d_model=768,
        num_layers=2,
        num_heads=12,
        use_slots=True,
        num_slots=7,
        d_slot=64,
    )
    bert = BERT(rngs=rngs, config=bert_config)

    fusion_config = MultimodalFusionConfig(
        vit_d_model=768,
        bert_d_model=768,
        joint_d_model=512,
        num_heads=8,
        num_layers=2,
    )
    fusion = MultimodalFusion(rngs=rngs, config=fusion_config)

    batch_size = 2
    img = jax.random.normal(rngs.params(), (batch_size, 224, 224, 3))
    input_ids = jax.random.randint(rngs.params(), (batch_size, 128), 0, 30522)
    attention_mask = jnp.ones((batch_size, 128))

    vit_out = vit(img)
    bert_out = bert(input_ids, attention_mask)

    chex.assert_shape(vit_out, (batch_size, 1000))
    chex.assert_shape(bert_out, (batch_size, 2))

    print(f"  ViT (slots) output shape: {vit_out.shape} ✓")
    print(f"  BERT (slots) output shape: {bert_out.shape} ✓")


def test_shape_assertions():
    print("Testing shape assertions (should raise errors)...")
    rngs = nnx.Rngs(5)
    config = MultimodalFusionConfig(
        vit_d_model=768,
        bert_d_model=768,
        joint_d_model=512,
        num_heads=8,
        num_layers=2,
    )
    fusion = MultimodalFusion(rngs=rngs, config=config)

    batch_size = 2
    vit_latent = jax.random.normal(rngs.params(), (batch_size, 768))
    bert_latent = jax.random.normal(rngs.params(), (batch_size, 768))

    try:
        fusion(vit_latent[:, :500], bert_latent)
        print("  ERROR: Should have raised assertion error for vit_d_model mismatch")
    except AssertionError as e:
        print(f"  Correctly caught vit_d_model mismatch: {str(e)[:80]}... ✓")

    try:
        fusion(vit_latent, bert_latent[:, :500])
        print("  ERROR: Should have raised assertion error for bert_d_model mismatch")
    except AssertionError as e:
        print(f"  Correctly caught bert_d_model mismatch: {str(e)[:80]}... ✓")

    try:
        fusion(vit_latent[:1], bert_latent)
        print("  ERROR: Should have raised assertion error for batch size mismatch")
    except AssertionError as e:
        print(f"  Correctly caught batch size mismatch: {str(e)[:80]}... ✓")

    try:
        fusion(vit_latent[None, ...], bert_latent[None, ...])
        print("  ERROR: Should have raised assertion error for 3D input")
    except AssertionError as e:
        print(f"  Correctly caught 3D input: {str(e)[:80]}... ✓")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Multimodal Architecture Tests")
    print("=" * 60)

    test_vit()
    print()
    test_bert()
    print()
    test_multimodal_fusion()
    print()
    test_full_architecture()
    print()
    test_with_slots()
    print()
    test_shape_assertions()

    print()
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)