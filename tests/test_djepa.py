"""Verification tests for D-JEPA architecture, losses, checkpointing, and probing."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
import pytest
from flax.training import train_state

from pidjepa.eval.probe import evaluate_downstream_probing
from pidjepa.losses.djepa import (
    DJEPACfg,
    compute_djepa_loss,
    loss_align,
    loss_ind,
    loss_rec,
    loss_sigreg,
)
from pidjepa.models.djepa import DJEPA, FiberPredictor, FusionHead, ModalityEncoder
from pidjepa.models.transformer import TransformerEncoder
from pidjepa.train.step import train_step
from pidjepa.utils.checkpoint import create_ckpt_manager, restore_checkpoint, save_checkpoint

# Dimensions for fast unit tests
B = 8
N_TOKENS = 4
D_MODEL = 32
D_C = 16
D_U = 16
D_TGT = 16


def build_test_djepa():
    """Create a lightweight DJEPA instance for fast testing."""
    t1 = TransformerEncoder(emb_dim=D_MODEL, n_heads=2, n_layers=1, mlp_dim=64, dropout_rate=0.0)
    t2 = TransformerEncoder(emb_dim=D_MODEL, n_heads=2, n_layers=1, mlp_dim=64, dropout_rate=0.0)
    enc1 = ModalityEncoder(transformer=t1, d_c=D_C, d_u=D_U)
    enc2 = ModalityEncoder(transformer=t2, d_c=D_C, d_u=D_U)
    pred1 = FiberPredictor(d_tgt=D_TGT, d_pred=32, n_heads=2, mlp_dim=64, n_layers=1)
    pred2 = FiberPredictor(d_tgt=D_TGT, d_pred=32, n_heads=2, mlp_dim=64, n_layers=1)
    fusion = FusionHead(num_classes=2, hidden_dim=32, n_heads=2)
    return DJEPA(encoder1=enc1, encoder2=enc2, pred1=pred1, pred2=pred2, fusion_head=fusion)


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


# ===================================================================
# Test 1: Output shapes (c_i: (B, d_c), u_i: (B, N_i, d_u)) & losses
# ===================================================================
def test_output_shapes_and_losses(key):
    model = build_test_djepa()
    k1, k2, k_init = jax.random.split(key, 3)

    x1 = jax.random.normal(k1, (B, N_TOKENS, D_MODEL))
    x2 = jax.random.normal(k2, (B, N_TOKENS, D_MODEL))

    variables = model.init(k_init, x1, x2)
    params = variables["params"]

    # 1. Modality 1 shapes
    c1, u1 = model.apply({"params": params}, x1, method="encode1")
    assert c1.shape == (B, D_C), f"Expected c1 shape {(B, D_C)}, got {c1.shape}"
    assert u1.shape == (B, N_TOKENS, D_U), f"Expected u1 shape {(B, N_TOKENS, D_U)}, got {u1.shape}"

    # 2. Modality 2 shapes
    c2, u2 = model.apply({"params": params}, x2, method="encode2")
    assert c2.shape == (B, D_C), f"Expected c2 shape {(B, D_C)}, got {c2.shape}"
    assert u2.shape == (B, N_TOKENS, D_U), f"Expected u2 shape {(B, N_TOKENS, D_U)}, got {u2.shape}"

    # 3. Masked context speedup (dropping tokens)
    keep_idx = jnp.zeros((B, 2), dtype=jnp.int32).at[:, 1].set(2)
    c1_ctx, u1_ctx = model.apply({"params": params}, x1, keep_indices=keep_idx, method="encode1")
    assert u1_ctx.shape == (B, 2, D_U)
    assert c1_ctx.shape == (B, D_C)

    # 4. Predictor output shape
    mask_pos = jax.random.normal(key, (B, 2, 8))
    pred_out = model.apply({"params": params}, u1_ctx, c1_ctx, mask_pos, method="predict1")
    assert pred_out.shape == (B, 2, D_TGT)

    # 5. Fusion head shapes across all 3 modes
    for mode in ("c_only", "u_only", "full"):
        logits = model.apply({"params": params}, c1, c2, u1, u2, mode=mode, method="fuse")
        assert logits.shape == (B, 2), f"Mode {mode} failed with shape {logits.shape}"

    # 6. Loss functions return finite scalars
    l_align = loss_align(c1, c2)
    assert l_align.shape == () and jnp.isfinite(l_align)

    l_sig_c = loss_sigreg(c1, key)
    assert l_sig_c.shape == () and jnp.isfinite(l_sig_c)

    l_sig_u = loss_sigreg(u1, key)
    assert l_sig_u.shape == () and jnp.isfinite(l_sig_u)

    l_ind = loss_ind(u1, c1)
    assert l_ind.shape == () and jnp.isfinite(l_ind)


# ===================================================================
# Test 2: stop_gradient on c_ctx and tgt_feat
# ===================================================================
def test_stop_gradient_c_and_tgt(key):
    u = jax.random.normal(key, (B, 2, D_U))
    c = jax.random.normal(key, (B, D_C))
    tgt = jax.random.normal(key, (B, 2, D_TGT))

    W = jax.random.normal(key, (D_U, D_TGT))
    pred_fn = lambda u_ctx, c_ctx, mask_pos=None: u_ctx @ W

    # Grad w.r.t. c_ctx must be zero
    grad_c = jax.grad(lambda c_: loss_rec(u, c_, tgt, pred_fn))(c)
    assert jnp.allclose(grad_c, 0.0), "Gradient w.r.t. c_ctx must be strictly zero"

    # Grad w.r.t. tgt_feat must be zero
    grad_tgt = jax.grad(lambda t_: loss_rec(u, c, t_, pred_fn))(tgt)
    assert jnp.allclose(grad_tgt, 0.0), "Gradient w.r.t. tgt_feat must be strictly zero"


# ===================================================================
# Test 3: 4 train_step updates, Orbax save & restore, parameter match,
#         and resumed step runs without NaNs
# ===================================================================
def test_orbax_checkpoint_and_resume(key, tmp_path):
    model = build_test_djepa()
    k_init, k_data, k_train = jax.random.split(key, 3)

    x1 = jax.random.normal(k_data, (B, N_TOKENS, D_MODEL))
    x2 = jax.random.normal(k_data, (B, N_TOKENS, D_MODEL))
    mask_pos = jax.random.normal(k_data, (B, N_TOKENS, 8))

    variables = model.init(k_init, x1, x2)
    params = variables["params"]
    target_params = jax.tree.map(jnp.copy, params)

    tx = optax.adam(1e-3)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    cfg = DJEPACfg()

    batch = {
        "x1": x1,
        "x2": x2,
        "x1_ctx": x1,
        "x2_ctx": x2,
        "x1_tgt": x1,
        "x2_tgt": x2,
        "mask_pos1": mask_pos,
        "mask_pos2": mask_pos,
    }

    # Run 4 train_steps
    step_key = k_train
    for i in range(4):
        step_key, s_key = jax.random.split(step_key)
        state, target_params, logs = train_step(state, target_params, batch, s_key, cfg)
        assert jnp.isfinite(logs["loss_total"]), f"Loss was NaN at step {i}"

    # Save Orbax checkpoint
    ckpt_dir = str(tmp_path / "test_ckpts")
    mngr = create_ckpt_manager(ckpt_dir, max_to_keep=2)

    save_checkpoint(
        mngr=mngr,
        step=4,
        model={"online": state.params, "target": target_params},
        optimizer=state.opt_state,
        key=step_key,
        epoch=1,
        metrics={"loss_total": float(logs["loss_total"])},
    )
    mngr.wait_until_finished()

    # Create fresh model + optimizer
    fresh_state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # Restore checkpoint
    restored = restore_checkpoint(
        mngr,
        model={"online": fresh_state.params, "target": target_params},
        optimizer=fresh_state.opt_state,
        step=4,
    )
    assert restored is not None, "Failed to restore checkpoint"

    # Assert step, epoch, key match
    assert restored["step"] == 4
    assert restored["epoch"] == 1
    assert jnp.array_equal(restored["key"], step_key)

    # Assert all online and target parameters match identically
    for k in state.params:
        assert jax.tree.all(
            jax.tree.map(jnp.array_equal, state.params[k], restored["model"]["online"][k])
        ), f"Mismatch in restored params for {k}"

    for k in target_params:
        assert jax.tree.all(
            jax.tree.map(jnp.array_equal, target_params[k], restored["model"]["target"][k])
        ), f"Mismatch in restored target params for {k}"

    # Assert optimizer state matches
    assert jax.tree.all(
        jax.tree.map(jnp.array_equal, state.opt_state, restored["optimizer"])
    ), "Optimizer state mismatch"

    # Resume training step from restored state
    resumed_state = fresh_state.replace(
        params=restored["model"]["online"],
        opt_state=restored["optimizer"],
        step=restored["step"],
    )
    resumed_target = restored["model"]["target"]

    step_key, next_s_key = jax.random.split(restored["key"])
    resumed_state, resumed_target, new_logs = train_step(
        resumed_state, resumed_target, batch, next_s_key, cfg
    )
    assert jnp.isfinite(new_logs["loss_total"]), "Resumed step resulted in NaN"
    assert new_logs["loss_total"] > 0.0


# ===================================================================
# Test 4: Downstream probe evaluation (c_only, u_only, full)
# ===================================================================
def test_downstream_probe(key):
    model = build_test_djepa()
    k_init, k_data, k_probe = jax.random.split(key, 3)

    x1 = jax.random.normal(k_data, (16, N_TOKENS, D_MODEL))
    x2 = jax.random.normal(k_data, (16, N_TOKENS, D_MODEL))
    y = jnp.array([0, 1] * 8)

    variables = model.init(k_init, x1, x2)
    params = variables["params"]

    dataset = {
        "train_x1": x1[:12],
        "train_x2": x2[:12],
        "train_y": y[:12],
        "val_x1": x1[12:],
        "val_x2": x2[12:],
        "val_y": y[12:],
    }

    results = evaluate_downstream_probing(
        model_apply_fn=model.apply,
        pretrained_params=params,
        dataset=dataset,
        key=k_probe,
        num_classes=2,
        modes=("c_only", "u_only", "full"),
        epochs=5,
    )

    for mode in ("c_only", "u_only", "full"):
        assert mode in results, f"Missing mode {mode} in probe results"
        metrics = results[mode]
        assert "accuracy" in metrics and 0.0 <= metrics["accuracy"] <= 1.0
        assert "f1" in metrics and 0.0 <= metrics["f1"] <= 1.0
        assert "auroc" in metrics
