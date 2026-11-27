import torch
from torch import nn
from torch.nn import functional as F

from .evaluation import encode_vision


@torch.inference_mode()
def _features(loader, vision_encoder, text_encoder, factorcl_model, device, image_size):
    xs, labels = [], []
    for batch in loader:
        images = batch["context"].get("vision")
        text = batch["context"].get("text")
        if factorcl_model is not None:
            vision, language, *_ = factorcl_model.representations(
                images.to(device), text.to(device)
            )
        else:
            vision = encode_vision(vision_encoder, images.to(device), image_size)[0]
            language = text_encoder(text.to(device))[0]
        xs.append(torch.cat((vision, language), dim=-1).cpu())
        labels.extend(batch.get("label", []))
    return torch.cat(xs), labels


def evaluate_downstream(
    train_loader,
    validation_loader,
    *,
    dataset,
    vision_encoder=None,
    text_encoder=None,
    factorcl_model=None,
    device,
    image_size,
    epochs=20,
    learning_rate=1e-3,
    wandb_run=None,
):
    """Train one compact linear probe for MM-IMDb, or paired retrieval for Flickr."""
    if dataset.lower().replace("-", "_") == "flickr30k":
        return _retrieval(
            train_loader,
            validation_loader,
            vision_encoder,
            text_encoder,
            factorcl_model,
            device,
            image_size,
            epochs,
            learning_rate,
            wandb_run,
        )

    train_x, train_labels = _features(
        train_loader, vision_encoder, text_encoder, factorcl_model, device, image_size
    )
    val_x, val_labels = _features(
        validation_loader,
        vision_encoder,
        text_encoder,
        factorcl_model,
        device,
        image_size,
    )
    if not train_labels or not any(str(label) for label in train_labels):
        return {}

    # MM-IMDb genres are pipe-separated multi-label targets.
    classes = sorted(
        {part for label in train_labels for part in str(label).split("|") if part}
    )
    if len(classes) > 1:
        lookup = {name: i for i, name in enumerate(classes)}

        def targets(values):
            result = torch.zeros(len(values), len(classes))
            for row, label in enumerate(values):
                for name in str(label).split("|"):
                    if name in lookup:
                        result[row, lookup[name]] = 1
            return result

        y_train, y_val = targets(train_labels), targets(val_labels)
        head = nn.Linear(train_x.shape[1], len(classes)).to(device)
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        values = sorted(set(train_labels))
        lookup = {value: i for i, value in enumerate(values)}
        y_train = torch.tensor([lookup[v] for v in train_labels])
        y_val = torch.tensor([lookup.get(v, -1) for v in val_labels])
        head = nn.Linear(train_x.shape[1], len(values)).to(device)
        loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate)
    train_x, val_x = train_x.to(device), val_x.to(device)
    y_train, y_val = y_train.to(device), y_val.to(device)
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
        if y_train.ndim == 2:
            score = ((logits.sigmoid() >= 0.5) == y_val.bool()).float().mean()
            metric = "accuracy"
        else:
            valid = y_val >= 0
            score = (logits[valid].argmax(-1) == y_val[valid]).float().mean()
            metric = "accuracy"
    result = {
        "downstream/genre_train_loss": float(loss.item()),
        "downstream/genre_validation_loss": float(val_loss.item()),
        f"downstream/genre_validation_{metric}": float(score.item()),
    }
    if wandb_run is not None:
        wandb_run.log(result)
    print("Downstream:", result)
    return result


def _retrieval(
    train_loader,
    validation_loader,
    vision_encoder,
    text_encoder,
    factorcl_model,
    device,
    image_size,
    epochs,
    learning_rate,
    wandb_run,
):
    # Reuse paired pretrained embeddings and learn a lightweight alignment head.
    def paired(loader):
        vs, ts = [], []
        with torch.inference_mode():
            for batch in loader:
                image, text = (
                    batch["context"]["vision"].to(device),
                    batch["context"]["text"].to(device),
                )
                if factorcl_model is not None:
                    v, t, *_ = factorcl_model.representations(image, text)
                else:
                    v = encode_vision(vision_encoder, image, image_size)[0]
                    t = text_encoder(text)[0]
                vs.append(v)
                ts.append(t)
        return torch.cat(vs), torch.cat(ts)

    train_v, train_t = paired(train_loader)
    val_v, val_t = paired(validation_loader)
    v_head, t_head = (
        nn.Linear(train_v.shape[-1], 256).to(device),
        nn.Linear(train_t.shape[-1], 256).to(device),
    )
    optimizer = torch.optim.AdamW(
        [*v_head.parameters(), *t_head.parameters()], lr=learning_rate
    )
    for _ in range(epochs):
        epoch_loss = 0.0
        for indices in torch.randperm(len(train_v), device=device).split(256):
            if len(indices) < 2:
                continue
            optimizer.zero_grad(set_to_none=True)
            scores = (
                F.normalize(v_head(train_v[indices]), dim=-1)
                @ F.normalize(t_head(train_t[indices]), dim=-1).T
                / 0.07
            )
            labels = torch.arange(len(scores), device=device)
            loss = (
                F.cross_entropy(scores, labels) + F.cross_entropy(scores.T, labels)
            ) / 2
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    with torch.no_grad():
        scores = (
            F.normalize(v_head(val_v), dim=-1)
            @ F.normalize(t_head(val_t), dim=-1).T
            / 0.07
        )
        labels = torch.arange(len(scores), device=device)
        val_loss = (
            F.cross_entropy(scores, labels) + F.cross_entropy(scores.T, labels)
        ) / 2
        v2t = scores.argmax(1).eq(labels).float().mean()
        t2v = scores.T.argmax(1).eq(labels).float().mean()
    result = {
        "downstream/retrieval_train_loss": float(
            epoch_loss / max(1, (len(train_v) + 255) // 256)
        ),
        "downstream/retrieval_validation_loss": float(val_loss.item()),
        "downstream/image_to_text_recall@1": float(v2t.item()),
        "downstream/text_to_image_recall@1": float(t2v.item()),
    }
    if wandb_run is not None:
        wandb_run.log(result)
    print("Downstream:", result)
    return result
