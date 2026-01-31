"""Explainability utilities for SSL models.

Provides:
- SHAP analysis for understanding classifier decisions
- Attention visualization for ViT models
- Latent space visualization with UMAP
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass
class SHAPConfig:
    """SHAP configuration."""

    method: str = "deep"  # deep, gradient, kernel
    n_samples: int = 100
    max_evals: int = 500


@dataclass
class AttentionConfig:
    """Attention visualization configuration."""

    head_fusion: str = "mean"  # mean, max, min
    discard_ratio: float = 0.9


@dataclass
class UMAPConfig:
    """UMAP configuration."""

    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "cosine"


class SHAPExplainer:
    """SHAP-based explanation for image classifiers.

    Note: SHAP explains *classifier* decisions, not SSL representations directly.
    Use this to understand what the downstream classifier learned from SSL features.
    """

    def __init__(
        self,
        model: nn.Module,
        config: SHAPConfig,
    ) -> None:
        """Initialize SHAP explainer.

        Args:
            model: Classifier model to explain.
            config: SHAP configuration.
        """
        self.model = model
        self.config = config
        self._explainer: Any = None

    def _build_explainer(self, background: torch.Tensor) -> Any:
        """Build SHAP explainer lazily.

        Args:
            background: Background samples for SHAP.

        Returns:
            SHAP explainer instance.
        """
        import shap

        if self.config.method == "deep":
            return shap.DeepExplainer(self.model, background)
        elif self.config.method == "gradient":
            return shap.GradientExplainer(self.model, background)
        else:
            raise ValueError(f"Unknown SHAP method: {self.config.method}")

    def explain(
        self,
        images: torch.Tensor,
        background: torch.Tensor | None = None,
    ) -> np.ndarray[Any, Any]:
        """Compute SHAP values for images.

        Args:
            images: Images to explain (B, C, H, W).
            background: Background samples. Required for first call.

        Returns:
            SHAP values array of shape (B, C, H, W).
        """
        if self._explainer is None:
            if background is None:
                raise ValueError("Background samples required for first call")
            self._explainer = self._build_explainer(background)

        shap_values = self._explainer.shap_values(images)

        # Handle multi-class output
        if isinstance(shap_values, list):
            # Return values for predicted class
            with torch.no_grad():
                preds = self.model(images).argmax(dim=1)
            shap_values = np.stack([shap_values[pred][i] for i, pred in enumerate(preds)])

        return shap_values

    def visualize(
        self,
        image: torch.Tensor,
        shap_values: np.ndarray[Any, Any],
        output_path: Path | None = None,
    ) -> Any:
        """Visualize SHAP values overlaid on image.

        Args:
            image: Original image tensor.
            shap_values: SHAP values for the image.
            output_path: Optional path to save visualization.

        Returns:
            Matplotlib figure.
        """
        import matplotlib.pyplot as plt

        # Convert to numpy and denormalize
        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)

        # Sum SHAP values across channels
        shap_sum = np.abs(shap_values).sum(axis=0)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(img_np)
        axes[0].set_title("Original")
        axes[0].axis("off")

        # SHAP heatmap
        axes[1].imshow(shap_sum, cmap="hot")
        axes[1].set_title("SHAP Importance")
        axes[1].axis("off")

        # Overlay
        axes[2].imshow(img_np)
        axes[2].imshow(shap_sum, cmap="hot", alpha=0.5)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")

        return fig


class AttentionVisualizer:
    """Visualize attention patterns in Vision Transformers.

    Caveat: Attention is not a causal explanation! It shows where the model
    attends, but not necessarily what drives decisions.
    """

    def __init__(self, config: AttentionConfig) -> None:
        """Initialize attention visualizer.

        Args:
            config: Attention configuration.
        """
        self.config = config

    def extract_attention(
        self,
        model: nn.Module,
        images: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Extract attention maps from ViT model.

        Args:
            model: ViT model with attention.
            images: Input images (B, C, H, W).

        Returns:
            List of attention tensors, one per layer.
        """
        attention_maps: list[torch.Tensor] = []

        # Register hooks to capture attention
        hooks = []

        def get_attention_hook(_module: nn.Module, _input: Any, output: Any) -> None:
            # For standard ViT attention, output is (attn_output, attn_weights)
            if isinstance(output, tuple) and len(output) > 1:
                attention_maps.append(output[1].detach())

        # Find attention modules
        for name, module in model.named_modules():
            if "attn" in name.lower() and hasattr(module, "forward"):
                hooks.append(module.register_forward_hook(get_attention_hook))

        # Forward pass
        with torch.no_grad():
            _ = model(images)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return attention_maps

    def aggregate_attention(
        self,
        attention_maps: list[torch.Tensor],
    ) -> torch.Tensor:
        """Aggregate attention across layers and heads.

        Args:
            attention_maps: List of attention tensors from each layer.

        Returns:
            Aggregated attention map (B, num_patches, num_patches).
        """
        if not attention_maps:
            raise ValueError("No attention maps extracted")

        # Stack layers: (num_layers, B, num_heads, seq_len, seq_len)
        stacked = torch.stack(attention_maps)

        # Fuse heads
        if self.config.head_fusion == "mean":
            fused = stacked.mean(dim=2)
        elif self.config.head_fusion == "max":
            fused = stacked.max(dim=2)[0]
        elif self.config.head_fusion == "min":
            fused = stacked.min(dim=2)[0]
        else:
            raise ValueError(f"Unknown head fusion: {self.config.head_fusion}")

        # Average across layers
        aggregated = fused.mean(dim=0)

        return aggregated

    def attention_rollout(
        self,
        attention_maps: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute attention rollout across layers.

        Reference: https://arxiv.org/abs/2005.00928

        Args:
            attention_maps: List of attention tensors from each layer.

        Returns:
            Rolled-out attention map (B, num_patches).
        """
        if not attention_maps:
            raise ValueError("No attention maps extracted")

        # Average across heads for each layer
        attentions = [attn.mean(dim=1) for attn in attention_maps]

        # Add identity (residual connections)
        result = torch.eye(attentions[0].shape[-1], device=attentions[0].device)
        result = result.unsqueeze(0).expand(attentions[0].shape[0], -1, -1)

        for attention in attentions:
            attention = attention + torch.eye(
                attention.shape[-1], device=attention.device
            ).unsqueeze(0)
            attention = attention / attention.sum(dim=-1, keepdim=True)
            result = torch.bmm(attention, result)

        # Get attention from CLS token to patches
        cls_attention = result[:, 0, 1:]  # Exclude CLS token itself

        return cls_attention

    def visualize(
        self,
        image: torch.Tensor,
        attention: torch.Tensor,
        output_path: Path | None = None,
        _patch_size: int = 16,
    ) -> Any:
        """Visualize attention as heatmap over image.

        Args:
            image: Original image tensor (C, H, W).
            attention: Attention weights for patches.
            output_path: Optional path to save visualization.
            _patch_size: Size of image patches (unused, for API consistency).

        Returns:
            Matplotlib figure.
        """
        import matplotlib.pyplot as plt
        from scipy.ndimage import zoom

        # Convert image to numpy
        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)

        h, w = img_np.shape[:2]

        # Reshape attention to grid
        num_patches = attention.shape[0]
        grid_size = int(np.sqrt(num_patches))
        attn_grid = attention.cpu().numpy().reshape(grid_size, grid_size)

        # Upsample to image size
        scale = h / grid_size
        attn_upsampled = zoom(attn_grid, scale, order=1)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img_np)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(attn_upsampled, cmap="viridis")
        axes[1].set_title("Attention")
        axes[1].axis("off")

        axes[2].imshow(img_np)
        axes[2].imshow(attn_upsampled, cmap="viridis", alpha=0.5)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")

        return fig


class LatentSpaceVisualizer:
    """Visualize latent space using UMAP/t-SNE."""

    def __init__(self, config: UMAPConfig) -> None:
        """Initialize latent space visualizer.

        Args:
            config: UMAP configuration.
        """
        self.config = config
        self._reducer: Any = None

    def _build_reducer(self) -> Any:
        """Build UMAP reducer lazily."""
        import umap

        return umap.UMAP(
            n_neighbors=self.config.n_neighbors,
            min_dist=self.config.min_dist,
            metric=self.config.metric,
        )

    @property
    def reducer(self) -> Any:
        """Get or build the reducer."""
        if self._reducer is None:
            self._reducer = self._build_reducer()
        return self._reducer

    def fit_transform(
        self,
        embeddings: torch.Tensor | np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Fit UMAP and transform embeddings.

        Args:
            embeddings: Feature embeddings (N, D).

        Returns:
            2D coordinates (N, 2).
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        return self.reducer.fit_transform(embeddings)

    def visualize(
        self,
        coords: np.ndarray[Any, Any],
        labels: np.ndarray[Any, Any] | list[str] | None = None,
        output_path: Path | None = None,
    ) -> Any:
        """Visualize 2D embedding space.

        Args:
            coords: 2D coordinates from UMAP.
            labels: Optional labels for coloring points.
            output_path: Optional path to save visualization.

        Returns:
            Matplotlib figure.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))

        if labels is not None:
            unique_labels = np.unique(labels)
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

            for label, color in zip(unique_labels, colors):
                mask = labels == label
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    c=[color],
                    label=str(label),
                    alpha=0.6,
                    s=10,
                )
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
            ax.scatter(coords[:, 0], coords[:, 1], alpha=0.6, s=10)

        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title("Latent Space Visualization")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")

        return fig


def compute_shap_summary(
    model: nn.Module,
    dataloader: Any,
    n_background: int = 50,
    n_explain: int = 100,
    device: str = "cpu",
) -> dict[str, Any]:
    """Compute SHAP summary statistics.

    Args:
        model: Model to explain.
        dataloader: Data loader.
        n_background: Number of background samples.
        n_explain: Number of samples to explain.
        device: Device to use.

    Returns:
        Dictionary with SHAP summary statistics.
    """
    model.eval()
    model = model.to(device)

    # Collect background and explanation samples
    background_samples: list[torch.Tensor] = []
    explain_samples: list[torch.Tensor] = []

    for batch in dataloader:
        images = batch["image"].to(device)

        if len(background_samples) < n_background:
            background_samples.append(images)
        elif len(explain_samples) < n_explain:
            explain_samples.append(images)
        else:
            break

    background = torch.cat(background_samples)[:n_background]
    explain = torch.cat(explain_samples)[:n_explain]

    # Compute SHAP values
    config = SHAPConfig()
    explainer = SHAPExplainer(model, config)
    shap_values = explainer.explain(explain, background)

    # Compute summary statistics
    mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2, 3))  # Per channel

    return {
        "mean_abs_shap_per_channel": mean_abs_shap.tolist(),
        "overall_importance": float(np.abs(shap_values).mean()),
        "n_samples": len(explain),
    }
