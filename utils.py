"""
utils.py
========
Shared utilities for the "Sufficient Statistic Distillation for Multimodal
Alignment" experiments.

Contents
--------
1. Device / seeding / tensor helpers (reparameterization, Gaussian KL).
2. Generic training loops:
     - fit_vae:         beta-VAE teacher trained on (x, y) pairs.
     - fit_students:    unimodal encoders distilled toward the teacher's
                        posterior mean (MSE + KL regularizer).
3. Evaluation helpers:
     - encode_mus / encode_joint_mus: chunked, deterministic posterior means.
     - modality_gap / modality_gap_per_class.
     - r2_proxy:        linear-regression R^2 as an MI proxy.
     - correlation_mi:  Gaussian MI estimate from pairwise correlations.
     - estimate_mi_mine: MINE (Donsker-Varadhan lower bound) MI estimate.
     - cross_modal_retrieval / latent_dim_variance.
4. Plotting helpers (Agg backend, saved to disk):
     - plot_scatter, plot_tsne, plot_line, plot_grouped_bars, savefig.
"""

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


# --------------------------------------------------------------------------- #
# Device / seeding / core tensor helpers
# --------------------------------------------------------------------------- #
def get_device():
    """Return the best available device (CUDA if present)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed=0):
    """Seed all randomness for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reparameterize(mu, logvar):
    """Reparameterization trick: z = mu + exp(0.5 logvar) * eps."""
    std = torch.exp(0.5 * logvar)
    return mu + std * torch.randn_like(std)


def kl_gaussian_standard(mu, logvar):
    """KL(q(z|.) || N(0, I)).

    Mean over the batch, sum over latent dimensions:
        KL = 1/2 * sum_d (mu_d^2 + exp(logvar_d) - 1 - logvar_d)
    """
    return 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=1).mean()


# --------------------------------------------------------------------------- #
# Training loops
# --------------------------------------------------------------------------- #
def fit_vae(
    model,
    loader,
    optimizer,
    loss_fn,
    beta,
    epochs,
    device,
    desc="JointVAE",
    verbose=True,
):
    """Train a beta-VAE teacher.

    Parameters
    ----------
    loader : DataLoader yielding (x, y) batches.
    loss_fn : callable(out, x, y, beta) -> scalar loss, where `out` is the
        dict returned by ``model(x, y)``.
    """
    model.to(device)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        running, n = 0.0, 0
        pbar = tqdm(loader, desc=f"{desc} e{epoch}/{epochs}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x, y)
            loss = loss_fn(out, x, y, beta)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
            n += x.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        mean = running / max(n, 1)
        history.append(mean)
        if verbose:
            print(f"[{desc}] epoch {epoch}/{epochs}: loss = {mean:.4f}")
    return history


def fit_students(
    teacher,
    student_x,
    student_y,
    loader,
    optimizer,
    beta,
    epochs,
    device,
    desc="StudentDistill",
    verbose=True,
):
    """Distill the (frozen) teacher's posterior mean into two unimodal encoders.

    Loss:
        L = MSE(mu_x, mu_t) + MSE(mu_y, mu_t)
          + beta * (KL(q(z|x)||N(0,I)) + KL(q(z|y)||N(0,I)))

    where mu_t = teacher.mu_joint(x, y) is computed with no gradient.
    """
    teacher.eval()
    teacher.to(device)
    student_x.to(device)
    student_y.to(device)
    history = []
    for epoch in range(1, epochs + 1):
        student_x.train()
        student_y.train()
        running, n = 0.0, 0
        pbar = tqdm(loader, desc=f"{desc} e{epoch}/{epochs}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                mu_t = teacher.mu_joint(x, y)
            mu_x, logvar_x = student_x(x)
            mu_y, logvar_y = student_y(y)
            loss = (
                F.mse_loss(mu_x, mu_t)
                + F.mse_loss(mu_y, mu_t)
                + beta
                * (
                    kl_gaussian_standard(mu_x, logvar_x)
                    + kl_gaussian_standard(mu_y, logvar_y)
                )
            )
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
            n += x.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        mean = running / max(n, 1)
        history.append(mean)
        if verbose:
            print(f"[{desc}] epoch {epoch}/{epochs}: loss = {mean:.4f}")
    return history


# --------------------------------------------------------------------------- #
# Encoding / evaluation helpers
# --------------------------------------------------------------------------- #
@torch.no_grad()
def encode_mus(encoder, X, device, batch_size=256):
    """Deterministic posterior means (encoder(X)) over a chunked tensor.

    X : (N, ...) tensor of a single modality. Returns (N, d_z).
    """
    encoder.eval()
    encoder.to(device)
    mus = []
    for i in range(0, X.shape[0], batch_size):
        xb = X[i : i + batch_size].to(device)
        mu, _ = encoder(xb)
        mus.append(mu.detach().cpu())
    return torch.cat(mus, dim=0)


@torch.no_grad()
def encode_joint_mus(teacher, X, Y, device, batch_size=256):
    """Teacher posterior means mu_joint(X, Y) over a chunked pair of tensors."""
    teacher.eval()
    teacher.to(device)
    mus = []
    for i in range(0, X.shape[0], batch_size):
        xb = X[i : i + batch_size].to(device)
        yb = Y[i : i + batch_size].to(device)
        mus.append(teacher.mu_joint(xb, yb).detach().cpu())
    return torch.cat(mus, dim=0)


def _as_numpy(t):
    """Convert a torch tensor or array to a numpy array."""
    if torch.is_tensor(t):
        t = t.detach().cpu().numpy()
    return np.asarray(t)


def modality_gap(mu_a, mu_b):
    """Euclidean distance between the aggregate (all-sample) mean codes.

    This is the classic "modality gap" quantity: ||E[z_x] - E[z_y]||.
    """
    ma = mu_a.mean(dim=0)
    mb = mu_b.mean(dim=0)
    return float((ma - mb).norm())


def modality_gap_per_class(mu_x, mu_y, labels, num_classes):
    """Per-class modality gap and its mean.

    For each class c, the distance between the class-conditional mean code of
    modality x and of modality y. Returns (mean_gap, list_of_gaps).
    """
    labels = _as_numpy(labels)
    gaps = []
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            continue
        mx = mu_x[mask].mean(dim=0)
        my = mu_y[mask].mean(dim=0)
        gaps.append(float((mx - my).norm()))
    return float(np.mean(gaps)), gaps


def r2_proxy(z, s):
    """Linear-regression MI proxy.

    Fit a linear map s ~ z and return the mean R^2 (fraction of variance of
    the true latent s explained by the code z). Higher R^2 => z carries more
    (linear) information about s, i.e. higher I(z; s).
    """
    from sklearn.linear_model import LinearRegression

    z = _as_numpy(z)
    s = _as_numpy(s)
    lr = LinearRegression().fit(z, s)
    return float(lr.score(z, s))


def correlation_mi(z, s):
    """Gaussian MI from pairwise correlations: sum_ij -0.5 log(1 - rho_ij^2).

    A cheap, fully supervised "simple correlation" MI estimate for each
    (z_i, s_j) pair under a bivariate-Gaussian model.
    """
    z = _as_numpy(z)
    s = _as_numpy(s)
    total = 0.0
    for j in range(s.shape[1]):
        for i in range(z.shape[1]):
            rho = np.corrcoef(z[:, i], s[:, j])[0, 1]
            if abs(rho) < 1.0:
                total += -0.5 * math.log(max(1.0 - rho**2, 1e-12))
    return float(total)


class _MINENet(nn.Module):
    """Small MLP discriminator used by MINE."""

    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, z):
        return self.net(z)


def estimate_mi_mine(
    z, s, device=None, steps=500, batch_size=256, lr=1e-3, max_samples=4096, seed=0
):
    """MINE: Mutual Information Neural Estimation (Donsker-Varadhan bound).

    Trains a network T to maximise
        I_T = E[T(z, s)] - log E[exp(T(z, s'))]
    with (z, s) from the joint and (z, s') from the product of marginals.

    Note: returns a (lower-bound) estimate; noisy for small budgets. Use the
    cheaper ``r2_proxy`` / ``correlation_mi`` for stable headline numbers.
    """
    device = device or torch.device("cpu")
    z = torch.as_tensor(z, dtype=torch.float32)
    s = torch.as_tensor(s, dtype=torch.float32)
    n = min(z.shape[0], s.shape[0])
    if n > max_samples:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(n, generator=g)[:max_samples]
        z, s = z[idx], s[idx]
        n = z.shape[0]
    model = _MINENet(z.shape[1] + s.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def dv_bound(zb, sb):
        joint = torch.cat([zb, sb], dim=1)
        marg = torch.cat([zb, sb[torch.randperm(sb.shape[0])]], dim=1)
        tj = model(joint.to(device))
        tm = model(marg.to(device))
        return tj.mean() - torch.logsumexp(tm, dim=0) + math.log(zb.shape[0])

    for _ in range(steps):
        idx = torch.randperm(n)[:batch_size]
        mi = dv_bound(z[idx], s[idx])
        opt.zero_grad()
        (-mi).backward()
        opt.step()

    with torch.no_grad():
        idx = torch.randperm(n)[:batch_size]
        mi = dv_bound(z[idx], s[idx])
    return float(mi)


def cross_modal_retrieval(mu_img, mu_lab, labels, num_classes=10):
    """Bidirectional latent-space nearest-neighbour retrieval accuracy.

    Image -> Label : each test image retrieves the nearest class-mean label
                     embedding; accuracy = fraction with the correct class.
    Label -> Image : each class-mean label embedding retrieves its nearest
                     test-image code; accuracy = fraction of classes that
                     land inside their own class cluster.
    Returns (acc_img_to_lab, acc_lab_to_img).
    """
    labels = _as_numpy(labels)
    mu_img = mu_img if torch.is_tensor(mu_img) else torch.from_numpy(mu_img)
    mu_lab = mu_lab if torch.is_tensor(mu_lab) else torch.from_numpy(mu_lab)
    device = mu_img.device
    class_means = torch.stack(
        [mu_lab[labels == c].mean(dim=0) for c in range(num_classes)]
    ).to(device)

    dist = torch.cdist(mu_img, class_means)  # (N, 10)
    acc_img_to_lab = float((dist.argmin(dim=1).cpu().numpy() == labels).mean())

    dist_rev = torch.cdist(class_means, mu_img)  # (10, N)
    nearest = dist_rev.argmin(dim=1).cpu().numpy()
    acc_lab_to_img = float((labels[nearest] == np.arange(num_classes)).mean())
    return acc_img_to_lab, acc_lab_to_img


def latent_dim_variance(mu):
    """Per-dimension variance of the code means (latent "dimension usage")."""
    return np.asarray(mu.var(dim=0).detach().cpu().numpy())


# --------------------------------------------------------------------------- #
# Plotting helpers
# --------------------------------------------------------------------------- #
def savefig(fig, path):
    """Save a figure, creating the parent directory, and close it."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {path}")


def _style_ax(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)


def plot_scatter(
    points,
    colors,
    title,
    path,
    markers=None,
    xlabel="dim 1",
    ylabel="dim 2",
    alpha=0.6,
    s=14,
    colorbar_label="value",
):
    """Scatter several (N, 2) point sets sharing one scalar color array."""
    colors = np.asarray(colors)
    if markers is None:
        markers = ["o"] * len(points)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for pts, m in zip(points, markers):
        pts = np.asarray(pts)
        ax.scatter(
            pts[:, 0], pts[:, 1], c=colors, s=s, alpha=alpha, marker=m, cmap="viridis"
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _style_ax(ax, title)
    cb = fig.colorbar(ax.collections[0], ax=ax)
    cb.set_label(colorbar_label)
    savefig(fig, path)


def plot_tsne(
    points,
    labels,
    title,
    path,
    markers=("o", "^"),
    perplexity=30,
    seed=0,
    n_subsample=1000,
    modality_names=("modality 1", "modality 2"),
):
    """t-SNE of several point sets plotted together.

    points : list of (N_i, d) arrays (one per modality).
    labels : list of scalar/discrete labels aligned with each point set
             (e.g. digit class); colour by label, marker by modality.
    """
    from sklearn.manifold import TSNE

    rng = np.random.RandomState(seed)
    subs, lab_subs = [], []
    for p, l in zip(points, labels):
        p, l = np.asarray(p), np.asarray(l)
        if len(p) > n_subsample:
            idx = rng.choice(len(p), n_subsample, replace=False)
            p, l = p[idx], l[idx]
        subs.append(p)
        lab_subs.append(l)

    pts = np.vstack(subs)
    labels_all = np.concatenate(lab_subs)
    emb = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        init="pca",
        n_iter=1000,
    ).fit_transform(pts)

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    cmap = plt.get_cmap("tab10")
    offset = 0
    for sub, marker in zip(subs, markers):
        n_i = len(sub)
        ax.scatter(
            emb[offset : offset + n_i, 0],
            emb[offset : offset + n_i, 1],
            c=labels_all[offset : offset + n_i],
            s=10,
            alpha=0.6,
            marker=marker,
            cmap=cmap,
            vmin=0,
            vmax=max(labels_all.max(), 9),
        )
        offset += n_i
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker=m,
            color="w",
            markerfacecolor="gray",
            markersize=9,
            label=name,
        )
        for m, name in zip(markers, modality_names)
    ]
    ax.legend(handles=handles, loc="best", framealpha=0.9)
    _style_ax(ax, title)
    cbar = fig.colorbar(
        ax.collections[0], ax=ax, ticks=np.arange(0, max(labels_all.max(), 9) + 1)
    )
    cbar.set_label("class")
    savefig(fig, path)


def plot_line(
    x,
    ys,
    labels,
    title,
    path,
    xlabel="beta_student",
    ylabel="value",
    xscale="log",
    yscale=None,
    figsize=(7, 6),
):
    """Multiple lines over a shared x axis (betas are log-spaced)."""
    fig, ax = plt.subplots(figsize=figsize)
    x = np.asarray(x, dtype=float)
    for y, lab in zip(ys, labels):
        ax.plot(x, np.asarray(y), marker="o", label=lab)
    ax.set_xscale(xscale)
    if yscale:
        ax.set_yscale(yscale)
    ax.legend()
    _style_ax(ax, title, xlabel, ylabel)
    savefig(fig, path)


def plot_grouped_bars(
    groups,
    values,
    title,
    path,
    xlabel="beta_student",
    ylabel="per-dimension variance",
    width=0.8,
):
    """Grouped bar chart of per-dimension quantities across configurations."""
    values = np.asarray(values)
    n_groups, n_dims = values.shape
    fig, ax = plt.subplots(figsize=(max(7, 2.0 * n_dims), 6))
    x = np.arange(n_groups)
    bw = width / max(n_dims, 1)
    cmap = plt.get_cmap("tab20")
    for d in range(n_dims):
        ax.bar(
            x + (d - n_dims / 2.0 + 0.5) * bw,
            values[:, d],
            width=bw,
            label=f"dim {d + 1}",
            color=cmap(d % 20),
        )
    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in groups])
    ax.legend(ncol=min(n_dims, 6), fontsize=8, framealpha=0.9)
    _style_ax(ax, title, xlabel, ylabel)
    savefig(fig, path)
