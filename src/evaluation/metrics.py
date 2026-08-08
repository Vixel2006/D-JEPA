import torch
import numpy as np
from typing import Dict, Any, Tuple, Union
from torch.utils.data import DataLoader


def extract_embeddings(
    model_or_models: Union[torch.nn.Module, Tuple[torch.nn.Module, torch.nn.Module]],
    dataloader: DataLoader,
    device: torch.device = None,
    model_type: str = "clip",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts latent embeddings for the entire dataloader.

    Args:
        model_or_models: Either CLIPModel (for 'clip') or a tuple of (ImageStudent, TextStudent) (for 'ssd')
        dataloader: DataLoader containing SyntheticPairedData
        device: CPU or GPU
        model_type: 'clip' or 'ssd'

    Returns:
        img_embs: Numpy array of shape (N, d_z)
        txt_embs: Numpy array of shape (N, d_z)
        shared_latents: Numpy array of shape (N, latent_dim) of ground-truth shared latents
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_list = []
    txt_list = []
    s_list = []

    # Handle model setups
    if model_type == "clip":
        clip_model = model_or_models
        clip_model.to(device)
        clip_model.eval()
    elif model_type == "ssd":
        img_student, txt_student = model_or_models
        img_student.to(device)
        txt_student.to(device)
        img_student.eval()
        txt_student.eval()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    with torch.no_grad():
        for img_feat, txt_feat, s in dataloader:
            img_feat = img_feat.to(device)
            txt_feat = txt_feat.to(device)

            if model_type == "clip":
                img_emb, txt_emb = clip_model(img_feat, txt_feat)
            else:  # ssd student encoders
                img_emb, _ = img_student(img_feat)  # extract mean mu
                txt_emb, _ = txt_student(txt_feat)  # extract mean mu

            img_list.append(img_emb.cpu().numpy())
            txt_list.append(txt_emb.cpu().numpy())
            s_list.append(s.numpy())

    img_embs = np.concatenate(img_list, axis=0)
    txt_embs = np.concatenate(txt_list, axis=0)
    shared_latents = np.concatenate(s_list, axis=0)

    return img_embs, txt_embs, shared_latents


def compute_modality_gap(
    img_embs: np.ndarray, txt_embs: np.ndarray, labels: np.ndarray = None
) -> Dict[str, Any]:
    """
    Computes the average Euclidean distance between the mean latent vector of images
    and the mean latent vector of texts.
    """
    mean_img = np.mean(img_embs, axis=0)
    mean_txt = np.mean(txt_embs, axis=0)
    overall_gap = float(np.linalg.norm(mean_img - mean_txt))

    results = {"overall": overall_gap}

    if labels is not None:
        class_gaps = {}
        unique_classes = np.unique(labels)
        for cls in unique_classes:
            mask = labels == cls
            if np.sum(mask) > 0:
                mean_img_c = np.mean(img_embs[mask], axis=0)
                mean_txt_c = np.mean(txt_embs[mask], axis=0)
                class_gaps[str(cls)] = float(np.linalg.norm(mean_img_c - mean_txt_c))
        results["per_class"] = class_gaps

    return results


def compute_retrieval_metrics(
    img_embs: np.ndarray, txt_embs: np.ndarray
) -> Dict[str, float]:
    """
    Computes symmetric cross-modal retrieval metrics (Recall@1, @5, @10)
    using cosine similarity.
    """
    # L2 normalization to compute cosine similarity using dot product
    img_norm = img_embs / (np.linalg.norm(img_embs, axis=1, keepdims=True) + 1e-10)
    txt_norm = txt_embs / (np.linalg.norm(txt_embs, axis=1, keepdims=True) + 1e-10)

    # Similarity matrix: (num_images, num_texts)
    sim_matrix = np.dot(img_norm, txt_norm.T)
    num_samples = sim_matrix.shape[0]

    # 1. Image-to-Text (i2t) Retrieval
    i2t_ranks = np.argsort(-sim_matrix, axis=1)  # Sort in descending order
    # Index of ground truth matching text (which has the same index)
    i2t_gts = np.arange(num_samples)[:, None]
    i2t_matches = i2t_ranks == i2t_gts
    i2t_ranks_gt = np.argmax(
        i2t_matches, axis=1
    )  # Get index (rank) of the matching column

    # 2. Text-to-Image (t2i) Retrieval
    t2i_ranks = np.argsort(-sim_matrix.T, axis=1)
    t2i_gts = np.arange(num_samples)[:, None]
    t2i_matches = t2i_ranks == t2i_gts
    t2i_ranks_gt = np.argmax(t2i_matches, axis=1)

    def recall_at_k(ranks: np.ndarray, k: int) -> float:
        return float(np.mean(ranks < k) * 100.0)

    metrics = {
        "i2t_r1": recall_at_k(i2t_ranks_gt, 1),
        "i2t_r5": recall_at_k(i2t_ranks_gt, 5),
        "i2t_r10": recall_at_k(i2t_ranks_gt, 10),
        "t2i_r1": recall_at_k(t2i_ranks_gt, 1),
        "t2i_r5": recall_at_k(t2i_ranks_gt, 5),
        "t2i_r10": recall_at_k(t2i_ranks_gt, 10),
    }
    return metrics


def compute_latent_variance(img_embs: np.ndarray) -> float:
    """
    Computes the sum of variances of embeddings across all dimensions (higher = more information preserved).
    """
    variances = np.var(img_embs, axis=0)
    return float(np.sum(variances))


def get_retrieval_examples(
    img_embs: np.ndarray,
    txt_embs: np.ndarray,
    num_queries: int = 5,
    top_k: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Retrieves the top-k text matches for random image queries and vice versa.
    """
    rng = np.random.default_rng(seed)
    num_samples = img_embs.shape[0]
    query_indices = rng.choice(num_samples, size=num_queries, replace=False)

    # L2 normalize
    img_norm = img_embs / (np.linalg.norm(img_embs, axis=1, keepdims=True) + 1e-10)
    txt_norm = txt_embs / (np.linalg.norm(txt_embs, axis=1, keepdims=True) + 1e-10)

    sim_matrix = np.dot(img_norm, txt_norm.T)

    i2t_examples = []
    t2i_examples = []

    for idx in query_indices:
        # Image-to-Text retrieval
        similarities_i2t = sim_matrix[idx]
        sorted_indices_i2t = np.argsort(-similarities_i2t)[:top_k]
        sorted_scores_i2t = similarities_i2t[sorted_indices_i2t]
        i2t_examples.append(
            {
                "query_idx": int(idx),
                "retrieved_indices": sorted_indices_i2t.tolist(),
                "similarities": sorted_scores_i2t.tolist(),
            }
        )

        # Text-to-Image retrieval
        similarities_t2i = sim_matrix[:, idx]
        sorted_indices_t2i = np.argsort(-similarities_t2i)[:top_k]
        sorted_scores_t2i = similarities_t2i[sorted_indices_t2i]
        t2i_examples.append(
            {
                "query_idx": int(idx),
                "retrieved_indices": sorted_indices_t2i.tolist(),
                "similarities": sorted_scores_t2i.tolist(),
            }
        )

    return {"i2t": i2t_examples, "t2i": t2i_examples}
