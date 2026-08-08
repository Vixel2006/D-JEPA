import torch
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Any, Tuple
from torch.utils.data import DataLoader


def teacher_loss_fn(
    recon_img: torch.Tensor,
    x_img: torch.Tensor,
    recon_txt: torch.Tensor,
    x_txt: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the beta-VAE loss for the SSD Teacher.
    Loss = MSE(recon_img, x_img) + MSE(recon_txt, x_txt) + beta * KL(q(z|x,y) || N(0, I))
    """
    # Reconstruction losses (MSE)
    recon_img_loss = F.mse_loss(recon_img, x_img, reduction="mean")
    recon_txt_loss = F.mse_loss(recon_txt, x_txt, reduction="mean")

    # KL Divergence to Standard Normal: KL(N(mu, var) || N(0, I))
    # Sum over latent dims, average over batch
    kl_loss = -0.5 * torch.mean(
        torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    )

    total_loss = recon_img_loss + recon_txt_loss + beta * kl_loss
    return total_loss, recon_img_loss, recon_txt_loss, kl_loss


def train_ssd_teacher(
    config: Dict[str, Any],
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device | None = None,
) -> torch.nn.Module:
    """
    Trains the SSD Teacher model (beta-VAE).
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
    beta = config.get("beta_teacher", 0.5)

    print(
        f"Starting SSD Teacher training on {device} for {num_epochs} epochs with beta_teacher={beta}..."
    )

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_recon_img = 0.0
        epoch_recon_txt = 0.0
        epoch_kl = 0.0

        for _, (img_feat, txt_feat, _) in enumerate(dataloader):
            img_feat = img_feat.to(device)
            txt_feat = txt_feat.to(device)

            optimizer.zero_grad()
            recon_img, recon_txt, mu, logvar = model(img_feat, txt_feat)

            loss, recon_img_l, recon_txt_l, kl_l = teacher_loss_fn(
                recon_img, img_feat, recon_txt, txt_feat, mu, logvar, beta=beta
            )

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon_img += recon_img_l.item()
            epoch_recon_txt += recon_txt_l.item()
            epoch_kl += kl_l.item()

        avg_loss = epoch_loss / len(dataloader)
        avg_recon_img = epoch_recon_img / len(dataloader)
        avg_recon_txt = epoch_recon_txt / len(dataloader)
        avg_kl = epoch_kl / len(dataloader)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] - "
            f"Loss: {avg_loss:.4f} (Recon Img: {avg_recon_img:.4f}, "
            f"Recon Txt: {avg_recon_txt:.4f}, KL: {avg_kl:.4f})"
        )

    return model
