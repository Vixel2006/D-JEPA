import torch
import torch.nn as nn
from .ssd_students import ViTImageDecoder, TransformerTextDecoder


class SSDTeacher(nn.Module):
    """
    Sufficient Statistic Distillation (SSD) Teacher model with MulT-like architecture.
    It takes both image and text features, applies Conv1D, adds positional encodings,
    performs bidirectional cross-attention, fuses the modalities, and estimates
    a joint posterior q(z|x,y) using a Variational Autoencoder (VAE) framework
    with a ViT decoder for images and a Transformer decoder for text.
    """

    def __init__(
        self,
        img_in_dim: int = 512,
        txt_in_dim: int = 384,
        d_model: int = 256,
        d_z: int = 128,
        nhead: int = 4,
        num_layers: int = 2,  # Kept for signature compatibility
        dropout: float = 0.1,
        seq_len: int = 4,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_z = d_z
        self.seq_len = seq_len

        # Define number of input channels for Conv1D
        assert img_in_dim % seq_len == 0, f"img_in_dim ({img_in_dim}) must be divisible by seq_len ({seq_len})"
        assert txt_in_dim % seq_len == 0, f"txt_in_dim ({txt_in_dim}) must be divisible by seq_len ({seq_len})"
        self.img_in_channels = img_in_dim // seq_len
        self.txt_in_channels = txt_in_dim // seq_len

        # Conv1D layers to project/align modal features to d_model
        self.img_conv = nn.Conv1d(
            in_channels=self.img_in_channels,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.txt_conv = nn.Conv1d(
            in_channels=self.txt_in_channels,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        # Positional encodings
        self.img_pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.txt_pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))

        # Cross-Attention Layers
        # Image tokens query Text tokens
        self.img_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.img_attn_norm = nn.LayerNorm(d_model)

        # Text tokens query Image tokens
        self.txt_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.txt_attn_norm = nn.LayerNorm(d_model)

        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )

        # Latent heads for joint posterior q(z|x,y)
        self.fc_mu = nn.Linear(d_model, d_z)
        self.fc_logvar = nn.Linear(d_model, d_z)

        # ViT Image Decoder for reconstruction
        self.img_decoder = ViTImageDecoder(
            d_z=d_z,
            d_model=d_model,
            out_dim=img_in_dim,
            seq_len=seq_len,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Transformer Text Decoder for reconstruction
        self.txt_decoder = TransformerTextDecoder(
            d_z=d_z,
            d_model=d_model,
            out_dim=txt_in_dim,
            seq_len=seq_len,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.img_pos_embed, std=0.02)
        nn.init.normal_(self.txt_pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.img_conv.weight)
        if self.img_conv.bias is not None:
            nn.init.constant_(self.img_conv.bias, 0.0)
        nn.init.xavier_uniform_(self.txt_conv.weight)
        if self.txt_conv.bias is not None:
            nn.init.constant_(self.txt_conv.bias, 0.0)

    def encode(self, x_img: torch.Tensor, x_txt: torch.Tensor):
        """
        Encodes both modalities jointly to output mu and logvar.
        """
        batch_size = x_img.size(0)

        # Reshape flat features: (batch, in_dim) -> (batch, in_channels, seq_len)
        x_img_seq = x_img.view(batch_size, self.img_in_channels, self.seq_len)
        x_txt_seq = x_txt.view(batch_size, self.txt_in_channels, self.seq_len)

        # Conv1D projection: (batch, in_channels, seq_len) -> (batch, d_model, seq_len)
        img_feats = self.img_conv(x_img_seq)
        txt_feats = self.txt_conv(x_txt_seq)

        # Permute to (batch, seq_len, d_model)
        img_feats = img_feats.permute(0, 2, 1)
        txt_feats = txt_feats.permute(0, 2, 1)

        # Add positional encoding
        img_tokens = img_feats + self.img_pos_embed
        txt_tokens = txt_feats + self.txt_pos_embed

        # Bidirectional cross-attention with residual connection and layer norm
        # 1. Image attends to Text
        img_attended, _ = self.img_cross_attn(
            query=img_tokens, key=txt_tokens, value=txt_tokens
        )
        img_attended = self.img_attn_norm(img_attended + img_tokens)

        # 2. Text attends to Image
        txt_attended, _ = self.txt_cross_attn(
            query=txt_tokens, key=img_tokens, value=img_tokens
        )
        txt_attended = self.txt_attn_norm(txt_attended + txt_tokens)

        # Mean pooling over sequence dimension
        pooled_img = img_attended.mean(dim=1)  # (batch, d_model)
        pooled_txt = txt_attended.mean(dim=1)  # (batch, d_model)

        # Concatenate and fuse
        fused = torch.cat([pooled_img, pooled_txt], dim=-1)  # (batch, 2 * d_model)
        fused_feat = self.fusion_mlp(fused)  # (batch, d_model)

        # Latent distributions
        mu = self.fc_mu(fused_feat)
        logvar = self.fc_logvar(fused_feat)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) during training.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z: torch.Tensor):
        """
        Decodes latent z to reconstruct image and text features.
        """
        recon_img = self.img_decoder(z)
        recon_txt = self.txt_decoder(z)
        return recon_img, recon_txt

    def forward(self, x_img: torch.Tensor, x_txt: torch.Tensor):
        """
        Forward pass doing encode -> reparam -> decode.
        """
        mu, logvar = self.encode(x_img, x_txt)
        z = self.reparameterize(mu, logvar)
        recon_img, recon_txt = self.decode(z)
        return recon_img, recon_txt, mu, logvar
