import torch
import torch.nn as nn
from .ssd_students import ViTImageEncoder, TransformerTextEncoder


class ImageEncoder(nn.Module):
    """
    Transformer encoder for image features, using the same ViTImageEncoder
    backbone as the SSD models to ensure a fair comparison.
    """

    def __init__(
        self,
        input_dim: int = 512,
        d_model: int = 256,
        d_z: int = 128,
        seq_len: int = 4,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.encoder = ViTImageEncoder(
            input_dim=input_dim,
            d_model=d_model,
            d_z=d_z,
            seq_len=seq_len,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, return_all_tokens: bool = False) -> torch.Tensor:
        if return_all_tokens:
            batch_size = x.size(0)
            x_proj = (
                self.encoder.proj(x)
                .view(batch_size, self.encoder.seq_len, self.encoder.d_model)
            )
            cls_tokens = self.encoder.cls_token.expand(batch_size, -1, -1)
            x_seq = torch.cat((cls_tokens, x_proj), dim=1)
            x_seq = x_seq + self.encoder.pos_embed
            return self.encoder.transformer_encoder(x_seq)

        mu, _ = self.encoder(x)
        return mu


class TextEncoder(nn.Module):
    """
    Transformer encoder for text features, using the same TransformerTextEncoder
    backbone as the SSD models to ensure a fair comparison.
    """

    def __init__(
        self,
        input_dim: int = 384,
        d_model: int = 256,
        d_z: int = 128,
        seq_len: int = 4,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.encoder = TransformerTextEncoder(
            input_dim=input_dim,
            d_model=d_model,
            d_z=d_z,
            seq_len=seq_len,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, return_all_tokens: bool = False) -> torch.Tensor:
        if return_all_tokens:
            batch_size = x.size(0)
            x_proj = (
                self.encoder.proj(x)
                .view(batch_size, self.encoder.seq_len, self.encoder.d_model)
            )
            cls_tokens = self.encoder.cls_token.expand(batch_size, -1, -1)
            x_seq = torch.cat((cls_tokens, x_proj), dim=1)
            x_seq = x_seq + self.encoder.pos_embed
            return self.encoder.transformer_encoder(x_seq)

        mu, _ = self.encoder(x)
        return mu


class CLIPModel(nn.Module):
    """
    CLIP-style model containing both image and text encoders.
    """

    def __init__(self, image_encoder: ImageEncoder, text_encoder: TextEncoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

    def forward(self, img_feat: torch.Tensor, txt_feat: torch.Tensor):
        img_emb = self.image_encoder(img_feat)
        txt_emb = self.text_encoder(txt_feat)
        return img_emb, txt_emb
