"""Model definition template."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from torch import nn


@dataclass
class ModelConfig:
    """Model configuration."""

    name: str
    pretrained: bool = True
    num_classes: int | None = None


class TemplateModel(nn.Module):  # type: ignore[misc]
    """Template model class.

    Customize this for your specific model architecture.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Initialize model.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config

        # Define your model architecture here
        # Example:
        # self.backbone = load_pretrained(config.name)
        # self.head = nn.Linear(hidden_size, config.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        # Implement forward pass
        raise NotImplementedError("Implement forward pass")

    @classmethod
    def from_config(cls, config: ModelConfig) -> TemplateModel:
        """Create model from configuration.

        Args:
            config: Model configuration.

        Returns:
            Initialized model.
        """
        return cls(config)

    @classmethod
    def from_pretrained(cls, path: Path) -> TemplateModel:
        """Load model from checkpoint.

        Args:
            path: Path to checkpoint.

        Returns:
            Loaded model.
        """
        checkpoint = torch.load(path, map_location="cpu")
        config = ModelConfig(**checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["state_dict"])
        return model

    def save(self, path: Path) -> None:
        """Save model checkpoint.

        Args:
            path: Output path.
        """
        checkpoint = {
            "config": {
                "name": self.config.name,
                "pretrained": self.config.pretrained,
                "num_classes": self.config.num_classes,
            },
            "state_dict": self.state_dict(),
        }
        torch.save(checkpoint, path)


def create_model(config: ModelConfig, device: str = "cpu") -> TemplateModel:
    """Create and initialize model.

    Args:
        config: Model configuration.
        device: Device to place model on.

    Returns:
        Initialized model on specified device.
    """
    model = TemplateModel.from_config(config)
    model = cast(TemplateModel, model.to(device))
    return model
