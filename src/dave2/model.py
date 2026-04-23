"""Phase 3: ViT-based DAVE-2 model family for steering-angle regression."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class DAVE2ViTConfig:
    """Configuration for a ViT-based DAVE-2 steering model."""

    variant: str
    image_size: tuple[int, int] = (66, 200)
    patch_size: tuple[int, int] = (6, 20)
    in_channels: int = 3
    embed_dim: int = 128
    depth: int = 4
    num_heads: int = 4
    mlp_ratio: float = 2.0
    dropout: float = 0.1
    attention_dropout: float = 0.0

    @property
    def num_patches(self) -> int:
        image_height, image_width = self.image_size
        patch_height, patch_width = self.patch_size
        return (image_height // patch_height) * (image_width // patch_width)


DAVE2_VIT_VARIANTS: dict[str, DAVE2ViTConfig] = {
    "mini": DAVE2ViTConfig(
        variant="mini",
        patch_size=(6, 20),
        embed_dim=128,
        depth=4,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.10,
        attention_dropout=0.00,
    ),
    "medium": DAVE2ViTConfig(
        variant="medium",
        patch_size=(6, 10),
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.10,
        attention_dropout=0.05,
    ),
    "large": DAVE2ViTConfig(
        variant="large",
        patch_size=(6, 10),
        embed_dim=384,
        depth=10,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.15,
        attention_dropout=0.10,
    ),
}


def get_dave2_vit_config(variant: str) -> DAVE2ViTConfig:
    """Return a validated model preset."""

    normalized_variant = variant.strip().lower()
    if normalized_variant not in DAVE2_VIT_VARIANTS:
        supported = ", ".join(sorted(DAVE2_VIT_VARIANTS))
        raise ValueError(f"Unknown model variant '{variant}'. Supported: {supported}.")
    return DAVE2_VIT_VARIANTS[normalized_variant]


class PatchEmbedding(nn.Module):
    """Project an image into a sequence of non-overlapping patch tokens."""

    def __init__(self, config: DAVE2ViTConfig) -> None:
        super().__init__()
        image_height, image_width = config.image_size
        patch_height, patch_width = config.patch_size

        if image_height % patch_height != 0 or image_width % patch_width != 0:
            raise ValueError(
                "image_size must be exactly divisible by patch_size for patch embedding."
            )

        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.num_patches = config.num_patches
        self.projection = nn.Conv2d(
            in_channels=config.in_channels,
            out_channels=config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patches = self.projection(images)
        return patches.flatten(start_dim=2).transpose(1, 2)


class MLPBlock(nn.Module):
    """Feed-forward sublayer used inside each transformer block."""

    def __init__(self, embed_dim: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens)


class TransformerBlock(nn.Module):
    """Pre-norm transformer encoder block with explicit attention dropout."""

    def __init__(self, config: DAVE2ViTConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            dropout=config.attention_dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.mlp = MLPBlock(
            embed_dim=config.embed_dim,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(tokens)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        tokens = tokens + self.dropout(attended)
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens


class DAVE2VisionEncoder(nn.Module):
    """Vision Transformer encoder that exposes token-level features."""

    def __init__(self, variant: str = "mini") -> None:
        super().__init__()
        self.config = get_dave2_vit_config(variant)

        self.patch_embed = PatchEmbedding(self.config)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.config.embed_dim))
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, self.config.embed_dim)
        )
        self.embedding_dropout = nn.Dropout(self.config.dropout)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.config) for _ in range(self.config.depth)]
        )
        self.final_norm = nn.LayerNorm(self.config.embed_dim)

        self._reset_parameters()

    @property
    def num_patches(self) -> int:
        return self.patch_embed.num_patches

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embeddings, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="linear",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_tokens(self, images: torch.Tensor) -> torch.Tensor:
        batch_size = images.shape[0]
        patch_tokens = self.patch_embed(images)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat((cls_tokens, patch_tokens), dim=1)
        tokens = tokens + self.position_embeddings[:, : tokens.shape[1], :]
        tokens = self.embedding_dropout(tokens)

        for block in self.transformer_blocks:
            tokens = block(tokens)

        return self.final_norm(tokens)

    def forward_features(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.forward_tokens(images)
        return encoded[:, 0], encoded[:, 1:]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        cls_token, _ = self.forward_features(images)
        return cls_token


class DAVE2(nn.Module):
    """Vision Transformer DAVE-2 model that preserves the original regression API."""

    def __init__(self, variant: str = "mini") -> None:
        super().__init__()
        self.encoder = DAVE2VisionEncoder(variant=variant)
        self.config = self.encoder.config

        hidden_dim = max(self.config.embed_dim // 2, 64)
        bottleneck_dim = max(hidden_dim // 2, 32)
        self.regression_head = nn.Sequential(
            nn.Linear(self.config.embed_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ELU(),
            nn.Linear(bottleneck_dim, 1),
        )

        self._reset_head()

    @property
    def num_patches(self) -> int:
        return self.encoder.num_patches

    def _reset_head(self) -> None:
        for module in self.regression_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        steering_token, _ = self.encoder.forward_features(images)
        return self.regression_head(steering_token)


def build_model(variant: str = "mini") -> DAVE2:
    """Construct a ViT-based DAVE-2 model for the requested size."""

    return DAVE2(variant=variant)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters for reporting and experiment tracking."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def describe_variant(variant: str) -> dict[str, int | float | str | tuple[int, int]]:
    """Return the core preset fields for logging or CLI display."""

    config = get_dave2_vit_config(variant)
    return {
        "variant": config.variant,
        "image_size": config.image_size,
        "patch_size": config.patch_size,
        "num_patches": config.num_patches,
        "embed_dim": config.embed_dim,
        "depth": config.depth,
        "num_heads": config.num_heads,
        "mlp_ratio": config.mlp_ratio,
        "dropout": config.dropout,
        "attention_dropout": config.attention_dropout,
    }
