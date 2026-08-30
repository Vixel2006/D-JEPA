import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import chex
from functools import partial
from pathlib import Path
from typing import Any

from src.losses import multimodal_kl_loss, full_multimodal_loss
from src.models.multimodal_model import MultimodalModel
from src.training.train_config import TrainConfig
from src.training.optimizer import create_optimizer

try:
    import orbax.checkpoint as ocp
    from orbax.checkpoint import PyTreeCheckpointer, CheckpointManager, CheckpointManagerOptions
    ORBAX_AVAILABLE = True
except ImportError:
    ORBAX_AVAILABLE = False

try:
    from tensorboardX import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    try:
        from torch.utils.tensorboard import SummaryWriter
        TENSORBOARD_AVAILABLE = True
    except ImportError:
        TENSORBOARD_AVAILABLE = False


@partial(jax.jit, static_argnames=(
    "kl_weight", "image_recon_weight", "text_recon_weight",
    "image_loss_type", "text_loss_type", "use_reconstruction", "tx"
))
def train_step(
    model: MultimodalModel,
    opt_state: optax.OptState,
    tx: optax.GradientTransformation,
    batch: tuple[jax.Array, jax.Array],
    kl_weight: float = 1.0,
    image_recon_weight: float = 1.0,
    text_recon_weight: float = 1.0,
    image_loss_type: str = "mse",
    text_loss_type: str = "cross_entropy",
    use_reconstruction: bool = True,
) -> tuple[MultimodalModel, optax.OptState, jax.Array, dict]:
    """
    Single training step computing KL(joint||image) + KL(joint||text) + reconstruction losses.

    Args:
        model: MultimodalModel containing ViT, BERT, Fusion, ImageDecoder, TextDecoder
        opt_state: optax optimizer state
        tx: optax gradient transformation
        batch: (images, input_ids) - images: (B, H, W, C), input_ids: (B, L)
        kl_weight: weight for KL loss
        image_recon_weight: weight for image reconstruction loss
        text_recon_weight: weight for text reconstruction loss
        image_loss_type: "mse" | "l1"
        text_loss_type: "cross_entropy"
        use_reconstruction: whether to use reconstruction losses

    Returns:
        new_model, new_opt_state, loss, metrics_dict
    """
    images, input_ids = batch

    def loss_fn(model: MultimodalModel):
        vit_latent = model.vit.encode(images)
        bert_latent = model.bert.encode(input_ids)
        joint_latent = model.fusion(vit_latent, bert_latent)

        if use_reconstruction:
            recon_image = model.image_decoder(joint_latent)
            recon_text_logits = model.text_decoder(joint_latent, input_ids)
        else:
            recon_image = None
            recon_text_logits = None

        loss, loss_dict = full_multimodal_loss(
            joint_mean=joint_latent,
            joint_logvar=jnp.zeros_like(joint_latent),
            image_mean=vit_latent,
            image_logvar=jnp.zeros_like(vit_latent),
            text_mean=bert_latent,
            text_logvar=jnp.zeros_like(bert_latent),
            recon_image=recon_image,
            target_image=images if use_reconstruction else None,
            recon_text_logits=recon_text_logits,
            target_text_ids=input_ids if use_reconstruction else None,
            kl_weight=kl_weight,
            image_recon_weight=image_recon_weight,
            text_recon_weight=text_recon_weight,
            image_loss_type=image_loss_type,
            text_loss_type=text_loss_type,
            reduction="mean",
        )
        return loss, (vit_latent, bert_latent, joint_latent, loss_dict)

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, (vit_latent, bert_latent, joint_latent, loss_dict)), grads = grad_fn(model)

    updates, new_opt_state = tx.update(grads, opt_state, nnx.state(model, nnx.Param))
    nnx.update(model, optax.apply_updates(nnx.state(model, nnx.Param), updates))

    metrics = {
        "loss": loss,
        "kl_loss": loss_dict.get("kl_loss", 0.0),
        "vit_latent_norm": jnp.mean(jnp.linalg.norm(vit_latent, axis=-1)),
        "bert_latent_norm": jnp.mean(jnp.linalg.norm(bert_latent, axis=-1)),
        "joint_latent_norm": jnp.mean(jnp.linalg.norm(joint_latent, axis=-1)),
    }
    if use_reconstruction:
        metrics["image_recon_loss"] = loss_dict.get("image_recon_loss", 0.0)
        metrics["text_recon_loss"] = loss_dict.get("text_recon_loss", 0.0)

    return model, new_opt_state, loss, metrics


@partial(jax.jit, static_argnames=(
    "kl_weight", "image_recon_weight", "text_recon_weight",
    "image_loss_type", "text_loss_type", "use_reconstruction"
))
def eval_step(
    model: MultimodalModel,
    batch: tuple[jax.Array, jax.Array],
    kl_weight: float = 1.0,
    image_recon_weight: float = 1.0,
    text_recon_weight: float = 1.0,
    image_loss_type: str = "mse",
    text_loss_type: str = "cross_entropy",
    use_reconstruction: bool = True,
) -> tuple[jax.Array, dict]:
    """
    Evaluation step without gradients.
    """
    images, input_ids = batch

    vit_latent = model.vit.encode(images)
    bert_latent = model.bert.encode(input_ids)
    joint_latent = model.fusion(vit_latent, bert_latent)

    if use_reconstruction:
        recon_image = model.image_decoder(joint_latent)
        recon_text_logits = model.text_decoder(joint_latent, input_ids)
    else:
        recon_image = None
        recon_text_logits = None

    loss, loss_dict = full_multimodal_loss(
        joint_mean=joint_latent,
        joint_logvar=jnp.zeros_like(joint_latent),
        image_mean=vit_latent,
        image_logvar=jnp.zeros_like(vit_latent),
        text_mean=bert_latent,
        text_logvar=jnp.zeros_like(bert_latent),
        recon_image=recon_image,
        target_image=images if use_reconstruction else None,
        recon_text_logits=recon_text_logits,
        target_text_ids=input_ids if use_reconstruction else None,
        kl_weight=kl_weight,
        image_recon_weight=image_recon_weight,
        text_recon_weight=text_recon_weight,
        image_loss_type=image_loss_type,
        text_loss_type=text_loss_type,
        reduction="mean",
    )

    metrics = {
        "eval_loss": loss,
        "kl_loss": loss_dict.get("kl_loss", 0.0),
        "vit_latent_norm": jnp.mean(jnp.linalg.norm(vit_latent, axis=-1)),
        "bert_latent_norm": jnp.mean(jnp.linalg.norm(bert_latent, axis=-1)),
        "joint_latent_norm": jnp.mean(jnp.linalg.norm(joint_latent, axis=-1)),
    }
    if use_reconstruction:
        metrics["image_recon_loss"] = loss_dict.get("image_recon_loss", 0.0)
        metrics["text_recon_loss"] = loss_dict.get("text_recon_loss", 0.0)

    return loss, metrics


class Trainer:
    def __init__(self, config: TrainConfig, model: MultimodalModel):
        self.config = config
        self.model = model

        self.tx = create_optimizer(config, model)
        self.opt_state = self.tx.init(nnx.state(model, nnx.Param))

        self.step = 0
        self.epoch = 0

        self._setup_logging()
        self._setup_checkpointing()

    def _setup_logging(self):
        """Setup TensorBoard logging."""
        self.writer = None
        if TENSORBOARD_AVAILABLE:
            log_dir = Path(self.config.tensorboard_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(log_dir))
            print(f"TensorBoard logging enabled at {log_dir}")
        else:
            print("TensorBoard not available. Install tensorboard or tensorboardX for logging.")

    def _setup_checkpointing(self):
        """Setup Orbax checkpointing."""
        self.checkpoint_manager = None
        if ORBAX_AVAILABLE:
            ckpt_dir = Path(self.config.checkpoint_dir).resolve()
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            checkpointer = PyTreeCheckpointer()
            options = CheckpointManagerOptions(
                max_to_keep=self.config.max_checkpoints,
                save_interval_steps=self.config.save_every,
            )
            self.checkpoint_manager = CheckpointManager(
                directory=str(ckpt_dir),
                checkpointers=checkpointer,
                options=options,
            )
            print(f"Orbax checkpointing enabled at {ckpt_dir}")
        else:
            print("Orbax not available. Checkpointing disabled.")

    def train_step(self, batch, use_reconstruction: bool = True):
        """Single training step."""
        self.step += 1

        self.model, self.opt_state, loss, metrics = train_step(
            self.model, self.opt_state, self.tx,
            batch,
            kl_weight=self.config.kl_weight,
            image_recon_weight=self.config.image_recon_weight,
            text_recon_weight=self.config.text_recon_weight,
            use_reconstruction=use_reconstruction,
        )

        if self.step % self.config.log_every == 0:
            self._log_metrics(metrics, prefix="train")

        return loss, metrics

    def evaluate(self, eval_loader, use_reconstruction: bool = True):
        """Run evaluation."""
        total_loss = 0.0
        total_metrics = {}
        num_batches = 0

        for batch in eval_loader:
            loss, metrics = eval_step(
                self.model, batch,
                kl_weight=self.config.kl_weight,
                image_recon_weight=self.config.image_recon_weight,
                text_recon_weight=self.config.text_recon_weight,
                use_reconstruction=use_reconstruction,
            )

            total_loss += loss
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v
            num_batches += 1

        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        avg_metrics["eval_loss"] = total_loss / num_batches

        self._log_metrics(avg_metrics, prefix="eval")
        return avg_metrics

    def _log_metrics(self, metrics: dict, prefix: str = ""):
        """Log metrics to console and TensorBoard."""
        prefix_str = f"{prefix}/" if prefix else ""
        metric_str = " | ".join(f"{prefix_str}{k}: {float(v):.4f}" for k, v in metrics.items())
        print(f"Step {self.step} | Epoch {self.epoch} | {metric_str}")

        if self.writer is not None:
            for k, v in metrics.items():
                self.writer.add_scalar(f"{prefix_str}{k}", float(v), self.step)
            self.writer.flush()

    def save_checkpoint(self, step: int | None = None):
        """Save model checkpoint using Orbax."""
        save_step = step if step is not None else self.step

        if self.checkpoint_manager is not None:
            state = {
                "model": nnx.state(self.model),
                "optimizer": nnx.state(self.opt_state),
                "step": self.step,
                "epoch": self.epoch,
            }
            self.checkpoint_manager.save(save_step, args=ocp.args.PyTreeSave(state))
            self.checkpoint_manager.wait_until_finished()
            print(f"Saved checkpoint at step {save_step}")
        else:
            import pickle
            ckpt_path = Path(self.config.checkpoint_dir) / f"checkpoint_step_{save_step}.pkl"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "model": nnx.state(self.model),
                "optimizer": nnx.state(self.opt_state),
                "step": self.step,
                "epoch": self.epoch,
            }
            with open(ckpt_path, "wb") as f:
                pickle.dump(state, f)
            print(f"Saved checkpoint to {ckpt_path}")

    def load_checkpoint(self, step: int | None = None):
        """Load model checkpoint using Orbax."""
        if self.checkpoint_manager is not None:
            if step is None:
                step = self.checkpoint_manager.latest_step()
                if step is None:
                    print("No checkpoint found")
                    return

            state = self.checkpoint_manager.restore(
                step,
                args=ocp.args.PyTreeRestore(None)
            )
            nnx.update(self.model, state["model"])
            self.opt_state = state["optimizer"]
            self.step = state["step"]
            self.epoch = state["epoch"]
            print(f"Loaded checkpoint from step {step}")
        else:
            import pickle
            if step is None:
                ckpt_files = list(Path(self.config.checkpoint_dir).glob("checkpoint_step_*.pkl"))
                if not ckpt_files:
                    print("No checkpoint found")
                    return
                ckpt_path = max(ckpt_files, key=lambda p: p.stat().st_mtime)
            else:
                ckpt_path = Path(self.config.checkpoint_dir) / f"checkpoint_step_{step}.pkl"

            if not ckpt_path.exists():
                print(f"Checkpoint not found: {ckpt_path}")
                return

            with open(ckpt_path, "rb") as f:
                state = pickle.load(f)
            nnx.update(self.model, state["model"])
            self.opt_state = state["optimizer"]
            self.step = state["step"]
            self.epoch = state["epoch"]
            print(f"Loaded checkpoint from {ckpt_path}")

    def close(self):
        """Close TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()