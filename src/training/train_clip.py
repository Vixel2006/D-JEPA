import torch
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Any, Tuple
from torch.utils.data import DataLoader


def contrastive_loss(
    img_emb: torch.Tensor, txt_emb: torch.Tensor, temperature: float = 0.07
) -> torch.Tensor:
    """
    Computes symmetric InfoNCE contrastive loss between image and text embeddings.
    """
    # L2 normalize embeddings
    img_emb = F.normalize(img_emb, p=2, dim=-1)
    txt_emb = F.normalize(txt_emb, p=2, dim=-1)

    # Compute similarity logits
    # shape: (batch_size, batch_size)
    logits_per_image = torch.matmul(img_emb, txt_emb.t()) / temperature
    logits_per_text = logits_per_image.t()

    # Target indices represent matching pairs along the diagonal
    labels = torch.arange(img_emb.size(0), device=img_emb.device)

    # Compute cross entropy in both directions
    loss_img = F.cross_entropy(logits_per_image, labels)
    loss_txt = F.cross_entropy(logits_per_text, labels)

    return (loss_img + loss_txt) / 2.0


def train_clip(
    config: Dict[str, Any],
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device = None,
) -> torch.nn.Module:
    """
    Trains the CLIP baseline model using symmetric InfoNCE contrastive loss.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.train()

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get("lr", 1e-4),
        weight_decay=config.get("weight_decay", 1e-4),
    )

    num_epochs = config.get("epochs", 5)
    temperature = config.get("clip_temperature", 0.07)

    print(f"Starting CLIP training on {device} for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (img_feat, txt_feat, _) in enumerate(dataloader):
            img_feat = img_feat.to(device)
            txt_feat = txt_feat.to(device)

            optimizer.zero_grad()
            img_emb, txt_emb = model(img_feat, txt_feat)

            loss = contrastive_loss(img_emb, txt_emb, temperature)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Average Loss: {avg_loss:.4f}")

    return model
