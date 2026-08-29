import jax
import jax.numpy as jnp
import chex
from functools import partial


def kl_divergence_gaussian(
    mean_p: jax.Array,
    logvar_p: jax.Array,
    mean_q: jax.Array,
    logvar_q: jax.Array,
) -> jax.Array:
    """
    KL divergence between two multivariate Gaussians with diagonal covariance.
    KL(q || p) = 0.5 * sum(log(var_p / var_q) + (var_q + (mean_q - mean_p)^2) / var_p - 1)

    Args:
        mean_p: (batch, d) - mean of distribution p (prior/modality)
        logvar_p: (batch, d) - log variance of distribution p
        mean_q: (batch, d) - mean of distribution q (joint/posterior)
        logvar_q: (batch, d) - log variance of distribution q

    Returns:
        kl: (batch,) - KL divergence per sample
    """
    chex.assert_equal_shape([mean_p, logvar_p, mean_q, logvar_q])
    chex.assert_rank(mean_p, 2)

    var_p = jnp.exp(logvar_p)
    var_q = jnp.exp(logvar_q)

    kl = 0.5 * jnp.sum(
        logvar_p - logvar_q
        + (var_q + (mean_q - mean_p) ** 2) / var_p
        - 1,
        axis=-1,
    )

    chex.assert_shape(kl, (mean_p.shape[0],))
    return kl


@partial(jax.jit, static_argnames=("reduction",))
def multimodal_kl_loss(
    joint_mean: jax.Array,
    joint_logvar: jax.Array,
    image_mean: jax.Array,
    image_logvar: jax.Array,
    text_mean: jax.Array,
    text_logvar: jax.Array,
    reduction: str = "mean",
) -> jax.Array:
    """
    KL(joint || image) + KL(joint || text)

    Args:
        joint_mean: (batch, d) - joint representation mean
        joint_logvar: (batch, d) - joint representation log variance
        image_mean: (batch, d) - image encoder latent mean
        image_logvar: (batch, d) - image encoder latent log variance
        text_mean: (batch, d) - text encoder latent mean
        text_logvar: (batch, d) - text encoder latent log variance
        reduction: "mean" | "sum" | "none"

    Returns:
        loss: scalar if reduction != "none" else (batch,)
    """
    chex.assert_rank(joint_mean, 2)
    chex.assert_rank(joint_logvar, 2)
    chex.assert_rank(image_mean, 2)
    chex.assert_rank(image_logvar, 2)
    chex.assert_rank(text_mean, 2)
    chex.assert_rank(text_logvar, 2)

    batch_size = joint_mean.shape[0]
    chex.assert_equal(batch_size, joint_logvar.shape[0])
    chex.assert_equal(batch_size, image_mean.shape[0])
    chex.assert_equal(batch_size, image_logvar.shape[0])
    chex.assert_equal(batch_size, text_mean.shape[0])
    chex.assert_equal(batch_size, text_logvar.shape[0])

    chex.assert_equal(joint_mean.shape[1], joint_logvar.shape[1])
    chex.assert_equal(image_mean.shape[1], image_logvar.shape[1])
    chex.assert_equal(text_mean.shape[1], text_logvar.shape[1])

    kl_joint_image = kl_divergence_gaussian(
        mean_p=image_mean,
        logvar_p=image_logvar,
        mean_q=joint_mean,
        logvar_q=joint_logvar,
    )
    chex.assert_shape(kl_joint_image, (batch_size,))

    kl_joint_text = kl_divergence_gaussian(
        mean_p=text_mean,
        logvar_p=text_logvar,
        mean_q=joint_mean,
        logvar_q=joint_logvar,
    )
    chex.assert_shape(kl_joint_text, (batch_size,))

    total_kl = kl_joint_image + kl_joint_text
    chex.assert_shape(total_kl, (batch_size,))

    if reduction == "mean":
        return jnp.mean(total_kl)
    elif reduction == "sum":
        return jnp.sum(total_kl)
    elif reduction == "none":
        return total_kl
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


@partial(jax.jit, static_argnames=("loss_type", "reduction"))
def reconstruction_loss(
    recon: jax.Array,
    target: jax.Array,
    loss_type: str = "mse",
    reduction: str = "mean",
) -> jax.Array:
    """
    Reconstruction loss for image or text.

    Args:
        recon: reconstructed output
        target: target output
        loss_type: "mse" | "l1" | "cross_entropy"
        reduction: "mean" | "sum" | "none"

    Returns:
        loss
    """
    chex.assert_equal_shape([recon, target])

    if loss_type == "mse":
        element_loss = jnp.mean((recon - target) ** 2, axis=tuple(range(1, recon.ndim)))
    elif loss_type == "l1":
        element_loss = jnp.mean(jnp.abs(recon - target), axis=tuple(range(1, recon.ndim)))
    elif loss_type == "cross_entropy":
        chex.assert_rank(target, 3)
        chex.assert_rank(recon, 3)
        log_probs = jax.nn.log_softmax(recon, axis=-1)
        element_loss = -jnp.sum(target * log_probs, axis=-1)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    if reduction == "mean":
        return jnp.mean(element_loss)
    elif reduction == "sum":
        return jnp.sum(element_loss)
    elif reduction == "none":
        return element_loss
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


@partial(jax.jit, static_argnames=("image_loss_type", "text_loss_type", "reduction",))
def full_multimodal_loss(
    joint_mean: jax.Array,
    joint_logvar: jax.Array,
    image_mean: jax.Array,
    image_logvar: jax.Array,
    text_mean: jax.Array,
    text_logvar: jax.Array,
    recon_image: jax.Array | None = None,
    target_image: jax.Array | None = None,
    recon_text_logits: jax.Array | None = None,
    target_text_ids: jax.Array | None = None,
    target_text_mask: jax.Array | None = None,
    kl_weight: float = 1.0,
    image_recon_weight: float = 1.0,
    text_recon_weight: float = 1.0,
    image_loss_type: str = "mse",
    text_loss_type: str = "cross_entropy",
    reduction: str = "mean",
) -> tuple[jax.Array, dict]:
    """
    Full multimodal loss: KL(joint||image) + KL(joint||text) + reconstruction losses.

    Args:
        joint_mean: (batch, d)
        joint_logvar: (batch, d)
        image_mean: (batch, d)
        image_logvar: (batch, d)
        text_mean: (batch, d)
        text_logvar: (batch, d)
        recon_image: (batch, H, W, C) - reconstructed image
        target_image: (batch, H, W, C) - target image
        recon_text_logits: (batch, seq_len, vocab) - text logits
        target_text_ids: (batch, seq_len) - target token ids
        target_text_mask: (batch, seq_len) - attention mask
        kl_weight: weight for KL terms
        image_recon_weight: weight for image reconstruction
        text_recon_weight: weight for text reconstruction
        image_loss_type: "mse" | "l1"
        text_loss_type: "cross_entropy"
        reduction: "mean" | "sum" | "none"

    Returns:
        total_loss: scalar
        loss_dict: dict with individual loss components
    """
    chex.assert_rank(joint_mean, 2)
    batch_size = joint_mean.shape[0]

    kl_loss = multimodal_kl_loss(
        joint_mean=joint_mean,
        joint_logvar=joint_logvar,
        image_mean=image_mean,
        image_logvar=image_logvar,
        text_mean=text_mean,
        text_logvar=text_logvar,
        reduction=reduction,
    )

    loss_dict = {"kl_loss": kl_loss}
    total_loss = kl_weight * kl_loss

    if recon_image is not None and target_image is not None:
        img_loss = reconstruction_loss(
            recon=recon_image,
            target=target_image,
            loss_type=image_loss_type,
            reduction=reduction,
        )
        loss_dict["image_recon_loss"] = img_loss
        total_loss = total_loss + image_recon_weight * img_loss

    if recon_text_logits is not None and target_text_ids is not None:
        if target_text_mask is not None:
            chex.assert_equal_shape([target_text_ids, target_text_mask])
            log_probs = jax.nn.log_softmax(recon_text_logits, axis=-1)
            target_one_hot = jax.nn.one_hot(target_text_ids, recon_text_logits.shape[-1])
            element_loss = -jnp.sum(target_one_hot * log_probs, axis=-1)
            masked_loss = element_loss * target_text_mask
            if reduction == "mean":
                text_loss = jnp.sum(masked_loss) / jnp.sum(target_text_mask)
            elif reduction == "sum":
                text_loss = jnp.sum(masked_loss)
            else:
                text_loss = masked_loss
        else:
            text_loss = reconstruction_loss(
                recon=recon_text_logits,
                target=jax.nn.one_hot(target_text_ids, recon_text_logits.shape[-1]),
                loss_type="cross_entropy",
                reduction=reduction,
            )
        loss_dict["text_recon_loss"] = text_loss
        total_loss = total_loss + text_recon_weight * text_loss

    loss_dict["total_loss"] = total_loss
    return total_loss, loss_dict


def sample_gaussian(mean: jax.Array, logvar: jax.Array, rng: jax.Array) -> jax.Array:
    """
    Reparameterization trick: sample from N(mean, exp(logvar)).

    Args:
        mean: (batch, d)
        logvar: (batch, d)
        rng: PRNG key

    Returns:
        sample: (batch, d)
    """
    chex.assert_equal_shape([mean, logvar])
    chex.assert_rank(mean, 2)

    std = jnp.exp(0.5 * logvar)
    eps = jax.random.normal(rng, mean.shape)
    sample = mean + eps * std

    chex.assert_shape(sample, mean.shape)
    return sample