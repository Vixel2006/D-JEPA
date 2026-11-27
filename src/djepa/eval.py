import torch
import torch.nn.functional as F
from torch import nn

from ..models import PredictorHead
from ..utils.evaluation import encode_vision


@torch.inference_mode()
def _projected_features(loader, vision_encoder, text_encoder, vision_projector,
                        text_projector, device, image_size):
    features, labels = [], []
    for batch in loader:
        context = batch["context"]
        images = context["vision"].to(device)
        text = context["text"].to(device)
        _, c_vision, u_vision = encode_vision(vision_encoder, images, image_size)
        _, c_text, u_text = text_encoder(text)
        _, z_vision = vision_projector(c_vision, u_vision)
        _, z_text = text_projector(c_text, u_text)
        features.append((z_vision, z_text))
        if "label" in batch:
            labels.extend(batch["label"])
    z_vision = torch.cat([pair[0] for pair in features])
    z_text = torch.cat([pair[1] for pair in features])
    return z_vision, z_text, labels


def train_mm_imdb(train_loader, validation_loader, *, vision_encoder, text_encoder,
                  vision_projector, text_projector, device, image_size,
                  epochs=20, learning_rate=1e-3, hidden_dim=256, batch_size=256,
                  temperature=0.07, wandb_run=None):
    """Train a multi-label genre predictor on paired projected features."""
    train_v, train_t, train_labels = _projected_features(
        train_loader, vision_encoder, text_encoder, vision_projector,
        text_projector, device, image_size,
    )
    val_v, val_t, val_labels = _projected_features(
        validation_loader, vision_encoder, text_encoder, vision_projector,
        text_projector, device, image_size,
    )
    classes = sorted({
        genre for label in train_labels
        for genre in str(label).split("|") if genre
    })
    if not classes:
        return {}
    lookup = {name: index for index, name in enumerate(classes)}

    def targets(labels):
        result = torch.zeros(len(labels), len(classes), device=device)
        for row, label in enumerate(labels):
            for genre in str(label).split("|"):
                if genre in lookup:
                    result[row, lookup[genre]] = 1
        return result

    train_x = torch.cat((train_v, train_t), dim=-1).to(device)
    val_x = torch.cat((val_v, val_t), dim=-1).to(device)
    y_train, y_val = targets(train_labels), targets(val_labels)
    head = PredictorHead(train_x.shape[-1], len(classes), hidden_dim).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(head(train_x), y_train)
        loss.backward()
        optimizer.step()

    head.eval()
    with torch.no_grad():
        logits = head(val_x)
        val_loss = loss_fn(logits, y_val)
        accuracy = ((logits.sigmoid() >= 0.5) == y_val.bool()).float().mean()
    result = {
        "evaluation/mm_imdb/train_loss": float(loss.item()),
        "evaluation/mm_imdb/validation_loss": float(val_loss.item()),
        "evaluation/mm_imdb/validation_accuracy": float(accuracy.item()),
    }
    if wandb_run is not None:
        wandb_run.log(result)
    print("Evaluation:", result)
    return result


def train_flickr30k(train_loader, validation_loader, *, vision_encoder, text_encoder,
                    vision_projector, text_projector, device, image_size,
                    epochs=20, learning_rate=1e-3, hidden_dim=256,
                    batch_size=256, temperature=0.07, wandb_run=None):
    """Train modality predictor heads with paired image/text contrastive loss."""
    train_v, train_t, _ = _projected_features(
        train_loader, vision_encoder, text_encoder, vision_projector,
        text_projector, device, image_size,
    )
    val_v, val_t, _ = _projected_features(
        validation_loader, vision_encoder, text_encoder, vision_projector,
        text_projector, device, image_size,
    )
    vision_head = PredictorHead(train_v.shape[-1], hidden_dim, hidden_dim).to(device)
    text_head = PredictorHead(train_t.shape[-1], hidden_dim, hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        [*vision_head.parameters(), *text_head.parameters()], lr=learning_rate
    )
    train_v, train_t = train_v.to(device), train_t.to(device)
    for _ in range(epochs):
        vision_head.train()
        text_head.train()
        order = torch.randperm(len(train_v), device=device)
        for indices in order.split(batch_size):
            if len(indices) < 2:
                continue
            optimizer.zero_grad(set_to_none=True)
            v = F.normalize(vision_head(train_v[indices]), dim=-1)
            t = F.normalize(text_head(train_t[indices]), dim=-1)
            logits = v @ t.T / temperature
            labels = torch.arange(len(indices), device=device)
            loss = (F.cross_entropy(logits, labels) +
                    F.cross_entropy(logits.T, labels)) / 2
            loss.backward()
            optimizer.step()

    vision_head.eval()
    text_head.eval()
    with torch.no_grad():
        logits = (F.normalize(vision_head(val_v.to(device)), dim=-1) @
                  F.normalize(text_head(val_t.to(device)), dim=-1).T / temperature)
        labels = torch.arange(len(logits), device=device)
        validation_loss = (F.cross_entropy(logits, labels) +
                           F.cross_entropy(logits.T, labels)) / 2
        image_recall = logits.argmax(dim=1).eq(labels).float().mean()
        text_recall = logits.T.argmax(dim=1).eq(labels).float().mean()
    result = {
        "evaluation/flickr30k/validation_loss": float(validation_loss.item()),
        "evaluation/flickr30k/image_to_text_recall@1": float(image_recall.item()),
        "evaluation/flickr30k/text_to_image_recall@1": float(text_recall.item()),
    }
    if wandb_run is not None:
        wandb_run.log(result)
    print("Evaluation:", result)
    return result


def evaluate_dataset(dataset, train_loader, validation_loader, **kwargs):
    """Dispatch to the small supervised task trainer for a supported dataset."""
    name = dataset.lower().replace("-", "_")
    trainers = {"mm_imdb": train_mm_imdb, "flickr30k": train_flickr30k}
    if name == "synthetic":
        return {}
    if name not in trainers:
        raise ValueError(f"No D-JEPA evaluation task is defined for {dataset!r}.")
    return trainers[name](train_loader, validation_loader, **kwargs)
