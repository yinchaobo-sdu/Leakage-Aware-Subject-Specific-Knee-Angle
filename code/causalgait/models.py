from __future__ import annotations

from typing import Iterable, List, Optional

import torch
import torch.nn as nn


class ScaleTransformerEncoder(nn.Module):
    def __init__(
        self,
        seq_len: int,
        feature_dim: int,
        scale: int,
        patch_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if scale <= 0:
            raise ValueError("Scale must be positive.")
        scaled_len = (seq_len + scale - 1) // scale
        n_patches = scaled_len // patch_size
        if n_patches < 1:
            raise ValueError(
                f"Scale {scale} with seq_len={seq_len} and patch_size={patch_size} has no complete patches."
            )

        self.scale = scale
        self.patch_size = patch_size
        self.feature_dim = feature_dim
        self.n_patches = n_patches
        self.patch_embedding = nn.Linear(feature_dim * patch_size, d_model)
        self.prediction_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position_embedding = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.prediction_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, time, features]
        x = x[:, :: self.scale, :]
        usable_len = self.n_patches * self.patch_size
        x = x[:, :usable_len, :]
        batch_size = x.shape[0]
        x = x.reshape(batch_size, self.n_patches, self.patch_size * self.feature_dim)
        x = self.patch_embedding(x)
        token = self.prediction_token.expand(batch_size, -1, -1)
        x = torch.cat([token, x], dim=1)
        x = x + self.position_embedding[:, : x.shape[1], :]
        encoded = self.encoder(x)
        return self.norm(encoded[:, 0, :])


class MultiScaleTransformerRegressor(nn.Module):
    def __init__(
        self,
        seq_len: int = 128,
        feature_dim: int = 16,
        scales: Iterable[int] = (1, 2, 4),
        patch_size: int = 8,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 1,
        ff_dim: int = 256,
        dropout: float = 0.1,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")
        self.scales = list(scales)
        self.scale_encoders = nn.ModuleList(
            [
                ScaleTransformerEncoder(
                    seq_len=seq_len,
                    feature_dim=feature_dim,
                    scale=scale,
                    patch_size=patch_size,
                    d_model=d_model,
                    nhead=nhead,
                    num_layers=num_layers,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for scale in self.scales
            ]
        )
        self.scale_logits = nn.Parameter(torch.zeros(len(self.scales)))
        self.output_dim = int(output_dim)
        self.output = nn.Linear(d_model, self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = torch.stack([encoder(x) for encoder in self.scale_encoders], dim=1)
        weights = torch.softmax(self.scale_logits, dim=0).view(1, -1, 1)
        aggregated = (tokens * weights).sum(dim=1)
        output = self.output(aggregated)
        return output.squeeze(-1) if self.output_dim == 1 else output


class TemporalConvBlock(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.net(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(residual + y)


class CausalGaitNet(nn.Module):
    def __init__(
        self,
        seq_len: int = 128,
        feature_dim: int = 56,
        scales: Iterable[int] = (1, 2, 4, 8),
        patch_size: int = 8,
        d_model: int = 192,
        nhead: int = 6,
        num_layers: int = 2,
        ff_dim: int = 384,
        dropout: float = 0.05,
        output_dim: int = 1,
        tcn_layers: int = 3,
        tcn_kernel_size: int = 5,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")
        self.output_dim = int(output_dim)
        self.input_projection = nn.Linear(feature_dim, d_model)
        self.tcn = nn.Sequential(
            *[
                TemporalConvBlock(
                    d_model=d_model,
                    kernel_size=tcn_kernel_size,
                    dilation=2**layer,
                    dropout=dropout,
                )
                for layer in range(tcn_layers)
            ]
        )
        self.local_pool = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU())
        self.scale_encoders = nn.ModuleList(
            [
                ScaleTransformerEncoder(
                    seq_len=seq_len,
                    feature_dim=feature_dim,
                    scale=scale,
                    patch_size=patch_size,
                    d_model=d_model,
                    nhead=nhead,
                    num_layers=num_layers,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for scale in scales
            ]
        )
        self.scale_logits = nn.Parameter(torch.zeros(len(self.scale_encoders)))
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Linear(d_model, d_model), nn.Sigmoid())
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, self.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.input_projection(x)
        local = self.tcn(local)
        local_token = self.local_pool(local.mean(dim=1))
        scale_tokens = torch.stack([encoder(x) for encoder in self.scale_encoders], dim=1)
        scale_weights = torch.softmax(self.scale_logits, dim=0).view(1, -1, 1)
        scale_token = (scale_tokens * scale_weights).sum(dim=1)
        gate = self.gate(torch.cat([local_token, scale_token], dim=-1))
        fused = gate * local_token + (1.0 - gate) * scale_token
        output = self.head(fused)
        return output.squeeze(-1) if self.output_dim == 1 else output


class MLPRegressor(nn.Module):
    def __init__(
        self,
        seq_len: int = 128,
        feature_dim: int = 16,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [256, 128, 64]
        self.output_dim = int(output_dim)
        layers: List[nn.Module] = []
        in_dim = seq_len * feature_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, self.output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.network(x.flatten(start_dim=1))
        return output.squeeze(-1) if self.output_dim == 1 else output


def build_model(model_name: str, **kwargs) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "transformer":
        return MultiScaleTransformerRegressor(**kwargs)
    if model_name == "causalgaitnet":
        return CausalGaitNet(**kwargs)
    if model_name == "mlp":
        return MLPRegressor(
            seq_len=kwargs["seq_len"],
            feature_dim=kwargs.get("feature_dim", 16),
            output_dim=kwargs.get("output_dim", 1),
        )
    raise ValueError(f"Unsupported model: {model_name}")
