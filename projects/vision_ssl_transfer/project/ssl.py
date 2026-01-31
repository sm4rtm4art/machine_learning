"""Self-Supervised Learning algorithms.

Implements:
- SimCLR: Contrastive learning with augmentation invariance
- MAE: Masked Autoencoder for vision
- (Planned: MoCo v3, DINO)
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from projects.vision_ssl_transfer.project.model import SSLBackbone
from torch import nn


@dataclass
class SimCLRConfig:
    """SimCLR configuration."""

    temperature: float = 0.5
    projection_dim: int = 128
    projection_hidden_dim: int = 2048


@dataclass
class MAEConfig:
    """MAE configuration."""

    mask_ratio: float = 0.75
    decoder_dim: int = 512
    decoder_depth: int = 8


class ProjectionHead(nn.Module):  # type: ignore[misc]
    """MLP projection head for contrastive learning."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 2048,
        output_dim: int = 128,
    ) -> None:
        """Initialize projection head.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
            output_dim: Output embedding dimension.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project features to embedding space.

        Args:
            x: Input features.

        Returns:
            Projected embeddings.
        """
        return self.net(x)


class SimCLR(nn.Module):  # type: ignore[misc]
    """SimCLR: A Simple Framework for Contrastive Learning.

    Reference: https://arxiv.org/abs/2002.05709
    """

    def __init__(
        self,
        backbone: SSLBackbone,
        config: SimCLRConfig,
    ) -> None:
        """Initialize SimCLR.

        Args:
            backbone: Encoder backbone.
            config: SimCLR configuration.
        """
        super().__init__()
        self.backbone = backbone
        self.config = config

        self.projector = ProjectionHead(
            input_dim=backbone.embed_dim,
            hidden_dim=config.projection_hidden_dim,
            output_dim=config.projection_dim,
        )

    def forward(self, view1: torch.Tensor, view2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass for both views.

        Args:
            view1: First augmented view (B, C, H, W).
            view2: Second augmented view (B, C, H, W).

        Returns:
            Dictionary with embeddings and loss.
        """
        # Encode both views
        h1 = self.backbone(view1)
        h2 = self.backbone(view2)

        # Project to embedding space
        z1 = self.projector(h1)
        z2 = self.projector(h2)

        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Compute NT-Xent loss
        loss = self.nt_xent_loss(z1, z2)

        return {
            "loss": loss,
            "z1": z1,
            "z2": z2,
            "h1": h1,
            "h2": h2,
        }

    def nt_xent_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.

        Args:
            z1: Embeddings from view 1 (B, D).
            z2: Embeddings from view 2 (B, D).

        Returns:
            Scalar loss.
        """
        batch_size = z1.shape[0]
        device = z1.device

        # Concatenate embeddings
        z = torch.cat([z1, z2], dim=0)  # (2B, D)

        # Compute similarity matrix
        sim = torch.mm(z, z.t()) / self.config.temperature  # (2B, 2B)

        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, device=device, dtype=torch.bool)
        sim.masked_fill_(mask, float("-inf"))

        # Create labels: positive pairs are (i, i+B) and (i+B, i)
        labels = torch.cat(
            [
                torch.arange(batch_size, 2 * batch_size, device=device),
                torch.arange(batch_size, device=device),
            ]
        )

        # Cross-entropy loss
        loss = F.cross_entropy(sim, labels)

        return loss

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to feature space (for downstream tasks).

        Args:
            x: Input images.

        Returns:
            Feature representations.
        """
        return self.backbone(x)


class MAE(nn.Module):  # type: ignore[misc]
    """Masked Autoencoder for Vision.

    Reference: https://arxiv.org/abs/2111.06377

    Note: This is a simplified implementation. For production use,
    consider using the official implementation or libraries like `lightly`.
    """

    def __init__(
        self,
        backbone: SSLBackbone,
        config: MAEConfig,
    ) -> None:
        """Initialize MAE.

        Args:
            backbone: Encoder backbone (should be ViT).
            config: MAE configuration.
        """
        super().__init__()
        self.backbone = backbone
        self.config = config

        # Decoder (simplified)
        self.decoder = nn.Sequential(
            nn.Linear(backbone.embed_dim, config.decoder_dim),
            *[
                nn.Sequential(
                    nn.LayerNorm(config.decoder_dim),
                    nn.Linear(config.decoder_dim, config.decoder_dim),
                    nn.GELU(),
                )
                for _ in range(config.decoder_depth)
            ],
        )

        # Prediction head: decoder_dim -> patch_size^2 * 3
        # Assuming patch_size=16 for standard ViT
        self.pred_head = nn.Linear(config.decoder_dim, 16 * 16 * 3)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass with masking.

        Args:
            x: Input images (B, C, H, W).

        Returns:
            Dictionary with loss and reconstructions.
        """
        # For a proper MAE implementation, we need to:
        # 1. Patchify the image
        # 2. Randomly mask patches
        # 3. Encode visible patches
        # 4. Decode all patches
        # 5. Compute reconstruction loss on masked patches

        # Simplified: just compute features and return placeholder loss
        features = self.backbone(x)
        decoded = self.decoder(features)

        # Placeholder loss
        loss = torch.tensor(0.0, device=x.device, requires_grad=True)

        return {
            "loss": loss,
            "features": features,
            "decoded": decoded,
        }

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to feature space (for downstream tasks).

        Args:
            x: Input images.

        Returns:
            Feature representations.
        """
        return self.backbone(x)


def create_ssl_model(
    backbone: SSLBackbone,
    algorithm: str,
    config: dict[str, Any],
) -> nn.Module:
    """Create SSL model from configuration.

    Args:
        backbone: Encoder backbone.
        algorithm: SSL algorithm name ('simclr', 'mae', etc.).
        config: Algorithm-specific configuration.

    Returns:
        Initialized SSL model.
    """
    if algorithm == "simclr":
        simclr_config = SimCLRConfig(**config.get("simclr", {}))
        return SimCLR(backbone, simclr_config)
    elif algorithm == "mae":
        mae_config = MAEConfig(**config.get("mae", {}))
        return MAE(backbone, mae_config)
    else:
        raise ValueError(f"Unknown SSL algorithm: {algorithm}")
