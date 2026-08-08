import torch
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Any, Tuple
from torch.utils.data import DataLoader


def student_loss_fn(
    mu_student: torch.Tensor,
    logvar_student: torch.Tensor,
    mu_teacher: torch.Tensor,
    beta: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes loss for a single student.
    Loss = MSE(mu_student, mu_teacher) + beta * KL(q_student(z|x) || N(0, I))
    """
    # MSE between student's mean and teacher's mean
    distill_loss = F.mse_loss(mu_student, mu_teacher, reduction="mean")

    # KL divergence to standard normal
    kl_loss = -0.5 * torch.mean(
        torch.sum(1 + logvar_student - mu_student.pow(2) - logvar_student.exp(), dim=-1)
    )

    total_loss = distill_loss + beta * kl_loss
    return total_loss, distill_loss, kl_loss


def train_ssd_students(
    config: Dict[str, Any],
    teacher: torch.nn.Module,
    img_student: torch.nn.Module,
    txt_student: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device = None,
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """
    Trains the unimodal student models (ImageStudent, TextStudent) by distilling the
    joint representation learned by the frozen teacher, with a KL-bottleneck constraint.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set teacher to eval and freeze parameters
    teacher = teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    img_student = img_student.to(device)
    txt_student = txt_student.to(device)

    img_student.train()
    txt_student.train()

    # Optimize both students jointly
    optimizer = optim.Adam(
        list(img_student.parameters()) + list(txt_student.parameters()),
        lr=config.get("lr", 1e-4),
        weight_decay=config.get("weight_decay", 1e-4),
    )

    num_epochs = config.get("epochs", 5)
    beta = config.get("beta_student", 0.1)

    print(
        f"Starting SSD Student distillation on {device} for {num_epochs} epochs with beta_student={beta}..."
    )

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_img_distill = 0.0
        epoch_img_kl = 0.0
        epoch_txt_distill = 0.0
        epoch_txt_kl = 0.0

        for batch_idx, (img_feat, txt_feat, _) in enumerate(dataloader):
            img_feat = img_feat.to(device)
            txt_feat = txt_feat.to(device)

            # 1. Get teacher targets (frozen, mean only)
            with torch.no_grad():
                mu_teacher, _ = teacher.encode(img_feat, txt_feat)

            # 2. Get student predictions
            optimizer.zero_grad()
            mu_img, logvar_img = img_student(img_feat)
            mu_txt, logvar_txt = txt_student(txt_feat)

            # 3. Compute student-specific losses
            img_loss, img_distill_l, img_kl_l = student_loss_fn(
                mu_img, logvar_img, mu_teacher, beta
            )
            txt_loss, txt_distill_l, txt_kl_l = student_loss_fn(
                mu_txt, logvar_txt, mu_teacher, beta
            )

            # 4. Joint backward and update
            total_loss = img_loss + txt_loss
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_img_distill += img_distill_l.item()
            epoch_img_kl += img_kl_l.item()
            epoch_txt_distill += txt_distill_l.item()
            epoch_txt_kl += txt_kl_l.item()

        avg_loss = epoch_loss / len(dataloader)
        avg_img_dist = epoch_img_distill / len(dataloader)
        avg_img_kl = epoch_img_kl / len(dataloader)
        avg_txt_dist = epoch_txt_distill / len(dataloader)
        avg_txt_kl = epoch_txt_kl / len(dataloader)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] - "
            f"Loss: {avg_loss:.4f} | "
            f"Img (Distill: {avg_img_dist:.4f}, KL: {avg_img_kl:.4f}) | "
            f"Txt (Distill: {avg_txt_dist:.4f}, KL: {avg_txt_kl:.4f})"
        )

    return img_student, txt_student
