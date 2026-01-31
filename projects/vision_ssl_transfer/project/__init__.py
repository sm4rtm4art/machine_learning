"""Vision SSL Transfer - Self-Supervised Learning with Explainability.

This module provides tools for:
- SSL pretraining (SimCLR, MAE, DINO)
- Transfer learning evaluation (linear probe, k-NN, fine-tuning)
- Explainability analysis (SHAP, attention visualization)
"""

from projects.vision_ssl_transfer.project.data import (
    OxfordPetsDataset,
    SSLAugmentation,
    create_ssl_dataloaders,
)
from projects.vision_ssl_transfer.project.model import (
    SSLBackbone,
    create_backbone,
)

__all__ = [
    "OxfordPetsDataset",
    "SSLAugmentation",
    "create_ssl_dataloaders",
    "SSLBackbone",
    "create_backbone",
]
