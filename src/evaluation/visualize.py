import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np


def plot_tsne(
    img_embs: np.ndarray,
    txt_embs: np.ndarray,
    shared_latents: np.ndarray,
    save_path: str,
    subset_size: int = 500,
    seed: int = 42,
) -> None:
    """
    Runs t-SNE jointly on a subset of image and text embeddings, generating and saving
    two subplots: one colored by modality, and the other by the first ground-truth shared latent dim.
    """
    n_samples = min(subset_size, img_embs.shape[0])

    # Slice the subset
    img_sub = img_embs[:n_samples]
    txt_sub = txt_embs[:n_samples]
    s_sub = shared_latents[:n_samples]

    # Concatenate for joint t-SNE fitting
    # shape: (2 * n_samples, d_z)
    joint_embs = np.concatenate([img_sub, txt_sub], axis=0)

    # Fit t-SNE
    tsne = TSNE(n_components=2, random_state=seed, perplexity=30, max_iter=1000)
    embs_2d = tsne.fit_transform(joint_embs)

    img_2d = embs_2d[:n_samples]
    txt_2d = embs_2d[n_samples:]

    # Create the figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Subplot 1: Color by Modality
    ax1.scatter(
        img_2d[:, 0],
        img_2d[:, 1],
        c="#1f77b4",
        label="Image",
        alpha=0.6,
        edgecolors="none",
        s=25,
    )
    ax1.scatter(
        txt_2d[:, 0],
        txt_2d[:, 1],
        c="#ff7f0e",
        label="Text",
        alpha=0.6,
        edgecolors="none",
        s=25,
    )

    # Draw line connectors between pairs to visually inspect modality alignment/gap
    for i in range(min(50, n_samples)):  # draw a few lines so the plot is clean
        ax1.plot(
            [img_2d[i, 0], txt_2d[i, 0]],
            [img_2d[i, 1], txt_2d[i, 1]],
            color="gray",
            alpha=0.15,
            linestyle="--",
        )

    ax1.set_title("Joint t-SNE Colored by Modality", fontsize=14, fontweight="bold")
    ax1.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax1.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: Color by Shared Latent s_1 (semantic factor)
    # Concatenate the coloring values: we color both img and txt samples of pair i with s[i, 0]
    s_values = np.concatenate([s_sub[:, 0], s_sub[:, 0]], axis=0)

    sc = ax2.scatter(
        embs_2d[:, 0],
        embs_2d[:, 1],
        c=s_values,
        cmap="viridis",
        alpha=0.7,
        edgecolors="none",
        s=25,
    )

    fig.colorbar(sc, ax=ax2, label="Ground-truth Shared Latent $s_0$")
    ax2.set_title(
        "Joint t-SNE Colored by Shared Latent $s_0$", fontsize=14, fontweight="bold"
    )
    ax2.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax2.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()

    # Make sure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"t-SNE visualization saved to: {save_path}")
