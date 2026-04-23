"""Perception-to-policy fusion modules for hybrid end-to-end driving."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import FusionConfig


class FeedForwardBlock(nn.Module):
    def __init__(self, embed_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.net(inputs)


class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(embed_dim)
        self.context_norm = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(embed_dim)
        self.feed_forward = FeedForwardBlock(embed_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: Tensor, context: Tensor) -> Tensor:
        normalized_queries = self.query_norm(queries)
        normalized_context = self.context_norm(context)
        attended, _ = self.attention(
            normalized_queries,
            normalized_context,
            normalized_context,
            need_weights=False,
        )
        queries = queries + self.dropout(attended)
        queries = queries + self.dropout(self.feed_forward(self.output_norm(queries)))
        return queries


class EarlyFusionProjector(nn.Module):
    """Modulates visual tokens with perception context before token pooling."""

    def __init__(self, embed_dim: int, control_dim: int, dropout: float) -> None:
        super().__init__()
        self.scene_projection = nn.Linear(embed_dim, embed_dim)
        self.control_projection = nn.Linear(control_dim, embed_dim * 2)
        self.output_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, vision_tokens: Tensor, scene_tokens: Tensor, control_features: Tensor) -> Tensor:
        scene_context = self.scene_projection(scene_tokens.mean(dim=1)).unsqueeze(1)
        gamma, beta = self.control_projection(control_features).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        modulated = vision_tokens * (1.0 + torch.tanh(gamma)) + beta + scene_context
        return self.output_norm(self.dropout(modulated).mean(dim=1))


class LateFusionProjector(nn.Module):
    def __init__(self, embed_dim: int, control_dim: int) -> None:
        super().__init__()
        self.control_projection = nn.Sequential(
            nn.LayerNorm(control_dim),
            nn.Linear(control_dim, embed_dim),
            nn.GELU(),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(embed_dim * 3),
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
        )

    def forward(self, vision_tokens: Tensor, scene_tokens: Tensor, control_features: Tensor) -> Tensor:
        vision_summary = vision_tokens.mean(dim=1)
        scene_summary = scene_tokens.mean(dim=1)
        control_summary = self.control_projection(control_features)
        return self.output(torch.cat((vision_summary, scene_summary, control_summary), dim=-1))


class CrossAttentionFusion(nn.Module):
    def __init__(self, config: FusionConfig, embed_dim: int, control_dim: int) -> None:
        super().__init__()
        self.control_projection = nn.Linear(control_dim, embed_dim)
        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(embed_dim=embed_dim, num_heads=config.num_heads, dropout=config.dropout)
                for _ in range(config.depth)
            ]
        )
        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(self, vision_tokens: Tensor, scene_tokens: Tensor, control_features: Tensor) -> Tensor:
        queries = vision_tokens[:, :1] + self.control_projection(control_features).unsqueeze(1)
        for block in self.blocks:
            queries = block(queries, scene_tokens)
        return self.output_norm(queries.squeeze(1))


class TokenLevelFusion(nn.Module):
    def __init__(self, config: FusionConfig, embed_dim: int, control_dim: int) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=config.num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.depth)
        self.control_projection = nn.Linear(control_dim, embed_dim)
        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(self, vision_tokens: Tensor, scene_tokens: Tensor, control_features: Tensor) -> Tensor:
        control_token = self.control_projection(control_features).unsqueeze(1)
        fused_tokens = torch.cat((vision_tokens, scene_tokens, control_token), dim=1)
        encoded = self.encoder(fused_tokens)
        return self.output_norm(encoded[:, 0])


class MultiBranchFusion(nn.Module):
    """Recommended fusion path: cross-attended policy token plus structured control branch."""

    def __init__(self, config: FusionConfig, embed_dim: int, control_dim: int) -> None:
        super().__init__()
        self.cross_attention = CrossAttentionFusion(config, embed_dim=embed_dim, control_dim=control_dim)
        self.scene_projection = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )
        self.control_projection = nn.Sequential(
            nn.LayerNorm(control_dim),
            nn.Linear(control_dim, embed_dim),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(embed_dim * 3),
            nn.Linear(embed_dim * 3, embed_dim),
            nn.Sigmoid(),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(embed_dim * 3),
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, vision_tokens: Tensor, scene_tokens: Tensor, control_features: Tensor) -> Tensor:
        attended_policy = self.cross_attention(vision_tokens, scene_tokens, control_features)
        scene_summary = self.scene_projection(scene_tokens.mean(dim=1))
        control_summary = self.control_projection(control_features)
        stacked = torch.cat((attended_policy, scene_summary, control_summary), dim=-1)
        gate = self.gate(stacked)
        fused = self.output(stacked)
        return gate * attended_policy + (1.0 - gate) * fused


class HybridFusionModule(nn.Module):
    """Ablation-ready fusion module that supports several integration strategies."""

    def __init__(
        self,
        config: FusionConfig | None,
        vision_dim: int,
        scene_dim: int,
        control_dim: int,
    ) -> None:
        super().__init__()
        self.config = config or FusionConfig()
        self.vision_projection = nn.Linear(vision_dim, self.config.fusion_dim)
        self.scene_projection = nn.Linear(scene_dim, self.config.fusion_dim)
        self.control_dim = control_dim

        if self.config.strategy == "early_fusion":
            self.strategy_module = EarlyFusionProjector(
                embed_dim=self.config.fusion_dim,
                control_dim=control_dim,
                dropout=self.config.dropout,
            )
        elif self.config.strategy == "late_fusion":
            self.strategy_module = LateFusionProjector(
                embed_dim=self.config.fusion_dim,
                control_dim=control_dim,
            )
        elif self.config.strategy == "cross_attention":
            self.strategy_module = CrossAttentionFusion(
                self.config,
                embed_dim=self.config.fusion_dim,
                control_dim=control_dim,
            )
        elif self.config.strategy == "token_level":
            self.strategy_module = TokenLevelFusion(
                self.config,
                embed_dim=self.config.fusion_dim,
                control_dim=control_dim,
            )
        elif self.config.strategy == "multi_branch":
            self.strategy_module = MultiBranchFusion(
                self.config,
                embed_dim=self.config.fusion_dim,
                control_dim=control_dim,
            )
        else:
            raise ValueError(
                "Unsupported fusion strategy "
                f"'{self.config.strategy}'. Expected early_fusion, late_fusion, "
                "cross_attention, multi_branch, or token_level."
            )

    def forward(self, vision_tokens: Tensor, scene_tokens: Tensor, control_features: Tensor) -> Tensor:
        projected_vision = self.vision_projection(vision_tokens)
        projected_scene = self.scene_projection(scene_tokens)
        return self.strategy_module(projected_vision, projected_scene, control_features)
