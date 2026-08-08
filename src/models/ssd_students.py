import torch
import torch.nn as nn


class ViTImageEncoder(nn.Module):
    """
    Vision Transformer (ViT) encoder for image features.
    Projects flat image features to token sequence, adds positional embeddings,
    prepends CLS token, runs through Transformer Encoder layers, and maps the CLS
    token to output (mu, logvar) of the latent distribution.
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
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        # Project flat feature to token sequence
        self.proj = nn.Linear(input_dim, seq_len * d_model)

        # [CLS] token parameter
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # Positional embedding for [CLS] + seq_len tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output projection MLP for mu and logvar
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout)
        )
        self.fc_mu = nn.Linear(d_model, d_z)
        self.fc_logvar = nn.Linear(d_model, d_z)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0.0)
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        nn.init.xavier_uniform_(self.fc_mu.weight)
        nn.init.xavier_uniform_(self.fc_logvar.weight)

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)

        # Project to sequence: (batch, seq_len, d_model)
        x_proj = self.proj(x).view(batch_size, self.seq_len, self.d_model)

        # Prepend [CLS]: (batch, seq_len + 1, d_model)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_seq = torch.cat((cls_tokens, x_proj), dim=1)

        # Add positional embedding
        x_seq = x_seq + self.pos_embed

        # Transformer encoder
        out = self.transformer_encoder(x_seq)  # (batch, seq_len + 1, d_model)

        # Extract CLS token
        cls_out = out[:, 0, :]  # (batch, d_model)
        feat = self.mlp(cls_out)
        mu = self.fc_mu(feat)
        logvar = self.fc_logvar(feat)
        return mu, logvar


class ViTImageDecoder(nn.Module):
    """
    Vision Transformer (ViT) decoder to reconstruct image features from latent z.
    Projects latent to a sequence, adds positional embeddings, runs through Transformer
    Encoder layers, and projects output tokens to image feature space.
    """

    def __init__(
        self,
        d_z: int = 128,
        d_model: int = 256,
        out_dim: int = 512,
        seq_len: int = 4,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        # Project latent to token sequence: (batch, seq_len, d_model)
        self.latent_proj = nn.Linear(d_z, seq_len * d_model)
        # Learnable positional embeddings for ViT tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.output_proj = nn.Linear(seq_len * d_model, out_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.latent_proj.weight)
        if self.latent_proj.bias is not None:
            nn.init.constant_(self.latent_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.size(0)
        # Project latent: (batch, seq_len, d_model)
        x_seq = self.latent_proj(z).view(batch_size, self.seq_len, self.d_model)
        # Add positional embedding
        x_seq = x_seq + self.pos_embed
        # Transformer: (batch, seq_len, d_model)
        out = self.transformer_encoder(x_seq)
        # Flatten and project to out_dim
        out = out.reshape(batch_size, -1)
        return self.output_proj(out)


class TransformerTextEncoder(nn.Module):
    """
    Transformer-based encoder for text features.
    Projects flat features to sequence, prepends [CLS], adds positional embedding,
    runs through Transformer Encoder, and outputs (mu, logvar) from [CLS].
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
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        # Project flat feature to sequence
        self.proj = nn.Linear(input_dim, seq_len * d_model)

        # [CLS] token parameter
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout)
        )
        self.fc_mu = nn.Linear(d_model, d_z)
        self.fc_logvar = nn.Linear(d_model, d_z)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0.0)
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        nn.init.xavier_uniform_(self.fc_mu.weight)
        nn.init.xavier_uniform_(self.fc_logvar.weight)

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)

        # Project to sequence: (batch, seq_len, d_model)
        x_proj = self.proj(x).view(batch_size, self.seq_len, self.d_model)

        # Prepend [CLS]: (batch, seq_len + 1, d_model)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_seq = torch.cat((cls_tokens, x_proj), dim=1)

        # Add positional embedding
        x_seq = x_seq + self.pos_embed

        # Transformer encoder
        out = self.transformer_encoder(x_seq)

        # Extract CLS token
        cls_out = out[:, 0, :]
        feat = self.mlp(cls_out)
        mu = self.fc_mu(feat)
        logvar = self.fc_logvar(feat)
        return mu, logvar


class TransformerTextDecoder(nn.Module):
    """
    Transformer-based decoder to reconstruct text features from latent z.
    Projects latent z to serve as memory, uses a learnable query sequence,
    decodes using cross-attention, and projects to text feature space.
    """

    def __init__(
        self,
        d_z: int = 128,
        d_model: int = 256,
        out_dim: int = 384,
        seq_len: int = 4,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.latent_proj = nn.Linear(d_z, d_model)
        self.query_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )
        self.output_proj = nn.Linear(seq_len * d_model, out_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.query_embed, std=0.02)
        nn.init.xavier_uniform_(self.latent_proj.weight)
        if self.latent_proj.bias is not None:
            nn.init.constant_(self.latent_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.size(0)
        # memory: (batch, 1, d_model)
        memory = self.latent_proj(z).unsqueeze(1)
        # tgt: (batch, seq_len, d_model)
        tgt = self.query_embed.expand(batch_size, -1, -1)
        # Decode: (batch, seq_len, d_model)
        out = self.transformer_decoder(tgt, memory)
        # Flatten and project to out_dim
        out = out.reshape(batch_size, -1)
        return self.output_proj(out)


class ImageStudent(nn.Module):
    """
    Unimodal student encoder wrapper for images.
    Uses ViTImageEncoder internally to output (mu, logvar).
    """

    def __init__(
        self,
        input_dim: int = 512,
        d_model: int = 256,
        d_z: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = ViTImageEncoder(
            input_dim=input_dim,
            d_model=d_model,
            d_z=d_z,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x: torch.Tensor):
        return self.encoder(x)


class TextStudent(nn.Module):
    """
    Unimodal student encoder wrapper for text.
    Uses TransformerTextEncoder internally to output (mu, logvar).
    """

    def __init__(
        self,
        input_dim: int = 384,
        d_model: int = 256,
        d_z: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = TransformerTextEncoder(
            input_dim=input_dim,
            d_model=d_model,
            d_z=d_z,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x: torch.Tensor):
        return self.encoder(x)
