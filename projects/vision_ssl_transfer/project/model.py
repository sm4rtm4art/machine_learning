"""Model definitions for Vision SSL Transfer.

Provides backbone wrappers for SSL pretraining and downstream tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from torch import nn


@dataclass
class BackboneConfig:
    """Backbone configuration."""

    name: str = "vit_small_patch16_224"
    pretrained: bool = False  # False for SSL from scratch
    num_classes: int | None = None  # None for feature extraction
    drop_rate: float = 0.0
    drop_path_rate: float = 0.1


class SSLBackbone(nn.Module):  # type: ignore[misc]
    """Backbone wrapper for SSL pretraining.

    Wraps timm models and provides feature extraction interface.
    """

    def __init__(self, config: BackboneConfig) -> None:
        """Initialize backbone.

        Args:
            config: Backbone configuration.
        """
        super().__init__()
        self.config = config
        self._model: nn.Module | None = None
        self._embed_dim: int | None = None

    def _build_model(self) -> tuple[nn.Module, int]:
        """Build the backbone model lazily."""
        import timm

        model = timm.create_model(
            self.config.name,
            pretrained=self.config.pretrained,
            num_classes=0,  # Remove classification head for feature extraction
            drop_rate=self.config.drop_rate,
            drop_path_rate=self.config.drop_path_rate,
        )

        # Get embedding dimension
        if hasattr(model, "num_features"):
            embed_dim = model.num_features
        elif hasattr(model, "embed_dim"):
            embed_dim = model.embed_dim
        else:
            # Fallback: run a dummy forward pass
            with torch.no_grad():
                dummy = torch.randn(1, 3, 224, 224)
                out = model(dummy)
                embed_dim = out.shape[-1]

        return model, embed_dim

    @property
    def model(self) -> nn.Module:
        """Get or build the model."""
        if self._model is None:
            self._model, self._embed_dim = self._build_model()
        return self._model

    @property
    def embed_dim(self) -> int:
        """Get embedding dimension."""
        if self._embed_dim is None:
            self._model, self._embed_dim = self._build_model()
        return self._embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from images.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Feature tensor of shape (B, embed_dim).
        """
        return self.model(x)

    def get_intermediate_features(
        self, x: torch.Tensor, layers: list[int] | None = None
    ) -> list[torch.Tensor]:
        """Get features from intermediate layers (for attention analysis).

        Args:
            x: Input tensor.
            layers: List of layer indices to extract. None for all.

        Returns:
            List of intermediate feature tensors.
        """
        # This is model-specific; placeholder for ViT
        features = []

        # For ViT models with blocks attribute
        if hasattr(self.model, "blocks"):
            x = self.model.patch_embed(x)
            if hasattr(self.model, "cls_token"):
                cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
                x = torch.cat([cls_token, x], dim=1)
            x = self.model.pos_drop(x + self.model.pos_embed)

            for i, block in enumerate(self.model.blocks):
                x = block(x)
                if layers is None or i in layers:
                    features.append(x.clone())

        return features

    @classmethod
    def from_config(cls, config: BackboneConfig) -> SSLBackbone:
        """Create backbone from configuration.

        Args:
            config: Backbone configuration.

        Returns:
            Initialized backbone.
        """
        return cls(config)

    @classmethod
    def from_pretrained(cls, path: Path) -> SSLBackbone:
        """Load backbone from checkpoint.

        Args:
            path: Path to checkpoint.

        Returns:
            Loaded backbone.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = BackboneConfig(**checkpoint["config"])
        backbone = cls(config)
        backbone.model.load_state_dict(checkpoint["state_dict"])
        return backbone

    def save(self, path: Path) -> None:
        """Save backbone checkpoint.

        Args:
            path: Output path.
        """
        checkpoint = {
            "config": {
                "name": self.config.name,
                "pretrained": self.config.pretrained,
                "num_classes": self.config.num_classes,
                "drop_rate": self.config.drop_rate,
                "drop_path_rate": self.config.drop_path_rate,
            },
            "state_dict": self.model.state_dict(),
        }
        torch.save(checkpoint, path)


class LinearProbeClassifier(nn.Module):  # type: ignore[misc]
    """Linear classifier for probing SSL representations.

    Freezes backbone and trains only the linear head.
    """

    def __init__(
        self,
        backbone: SSLBackbone,
        num_classes: int,
        freeze_backbone: bool = True,
    ) -> None:
        """Initialize linear probe.

        Args:
            backbone: Pretrained SSL backbone.
            num_classes: Number of output classes.
            freeze_backbone: Whether to freeze backbone weights.
        """
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.classifier = nn.Linear(backbone.embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input images.

        Returns:
            Class logits.
        """
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.backbone(x)
        else:
            features = self.backbone(x)

        return self.classifier(features)


class FineTuneClassifier(nn.Module):  # type: ignore[misc]
    """Full fine-tuning classifier.

    Optionally freezes backbone for initial epochs.
    """

    def __init__(
        self,
        backbone: SSLBackbone,
        num_classes: int,
    ) -> None:
        """Initialize fine-tune classifier.

        Args:
            backbone: Pretrained SSL backbone.
            num_classes: Number of output classes.
        """
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input images.

        Returns:
            Class logits.
        """
        features = self.backbone(x)
        return self.classifier(features)

    def freeze_backbone(self) -> None:
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True


def create_backbone(config: BackboneConfig, device: str = "cpu") -> SSLBackbone:
    """Create and initialize backbone.

    Args:
        config: Backbone configuration.
        device: Device to place model on.

    Returns:
        Initialized backbone on specified device.
    """
    backbone = SSLBackbone.from_config(config)
    # Trigger model building
    _ = backbone.model
    backbone = cast(SSLBackbone, backbone.to(device))
    return backbone
