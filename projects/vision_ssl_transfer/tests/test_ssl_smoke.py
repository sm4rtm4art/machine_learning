"""Smoke tests for Vision SSL Transfer project."""

import pytest
import torch


def test_imports() -> None:
    """Test that project modules can be imported."""
    from projects.vision_ssl_transfer.project import (  # noqa: F401  # noqa: F401
        data,
        eval,
        model,
        ssl,
        train,
    )


def test_config_loads() -> None:
    """Test that default config loads without errors."""
    from omegaconf import OmegaConf

    from ml_portfolio.common.paths import get_project_paths

    paths = get_project_paths("vision_ssl_transfer")
    config_path = paths.default_config()

    if config_path.exists():
        config = OmegaConf.load(config_path)
        assert config is not None
        assert "project" in config
        assert "ssl" in config
        assert config.ssl.algorithm in ["simclr", "moco_v3", "mae", "dino"]


def test_backbone_config() -> None:
    """Test backbone configuration creation."""
    from projects.vision_ssl_transfer.project.model import BackboneConfig

    config = BackboneConfig(name="vit_small_patch16_224", pretrained=False)
    assert config.name == "vit_small_patch16_224"
    assert config.pretrained is False


def test_ssl_augmentation() -> None:
    """Test SSL augmentation pipeline."""
    from PIL import Image
    from projects.vision_ssl_transfer.project.data import SSLAugmentation

    aug = SSLAugmentation(image_size=224)

    # Create dummy image
    dummy_image = Image.new("RGB", (256, 256), color="red")

    # Apply augmentation
    view1, view2 = aug(dummy_image)

    assert view1.shape == (3, 224, 224)
    assert view2.shape == (3, 224, 224)
    # Views should be different (almost certainly due to random augmentation)
    assert not torch.allclose(view1, view2)


def test_eval_augmentation() -> None:
    """Test evaluation augmentation pipeline."""
    from PIL import Image
    from projects.vision_ssl_transfer.project.data import EvalAugmentation

    aug = EvalAugmentation(image_size=224)

    dummy_image = Image.new("RGB", (256, 256), color="blue")
    output = aug(dummy_image)

    assert output.shape == (3, 224, 224)


def test_simclr_config() -> None:
    """Test SimCLR configuration."""
    from projects.vision_ssl_transfer.project.ssl import SimCLRConfig

    config = SimCLRConfig(temperature=0.5, projection_dim=128)
    assert config.temperature == 0.5
    assert config.projection_dim == 128


def test_projection_head() -> None:
    """Test projection head module."""
    from projects.vision_ssl_transfer.project.ssl import ProjectionHead

    head = ProjectionHead(input_dim=384, hidden_dim=2048, output_dim=128)

    dummy_input = torch.randn(4, 384)
    output = head(dummy_input)

    assert output.shape == (4, 128)


@pytest.mark.skip(reason="Requires timm model download")
def test_backbone_creation() -> None:
    """Test backbone model creation."""
    from projects.vision_ssl_transfer.project.model import BackboneConfig, create_backbone

    config = BackboneConfig(name="vit_tiny_patch16_224", pretrained=False)
    backbone = create_backbone(config, device="cpu")

    dummy_input = torch.randn(2, 3, 224, 224)
    output = backbone(dummy_input)

    assert output.shape[0] == 2
    assert len(output.shape) == 2  # (batch, embed_dim)


@pytest.mark.skip(reason="Requires timm model download")
def test_simclr_forward() -> None:
    """Test SimCLR forward pass."""
    from projects.vision_ssl_transfer.project.model import BackboneConfig, create_backbone
    from projects.vision_ssl_transfer.project.ssl import SimCLR, SimCLRConfig

    backbone_config = BackboneConfig(name="vit_tiny_patch16_224", pretrained=False)
    backbone = create_backbone(backbone_config, device="cpu")

    ssl_config = SimCLRConfig(temperature=0.5)
    model = SimCLR(backbone, ssl_config)

    view1 = torch.randn(4, 3, 224, 224)
    view2 = torch.randn(4, 3, 224, 224)

    outputs = model(view1, view2)

    assert "loss" in outputs
    assert "z1" in outputs
    assert "z2" in outputs
    assert outputs["loss"].requires_grad


def test_train_config() -> None:
    """Test training configuration."""
    from projects.vision_ssl_transfer.project.train import SSLTrainConfig, TransferTrainConfig

    ssl_config = SSLTrainConfig(epochs=10, learning_rate=1e-4)
    assert ssl_config.epochs == 10
    assert ssl_config.learning_rate == 1e-4

    transfer_config = TransferTrainConfig(epochs=50, freeze_backbone_epochs=5)
    assert transfer_config.epochs == 50
    assert transfer_config.freeze_backbone_epochs == 5


def test_eval_config() -> None:
    """Test evaluation configuration."""
    from projects.vision_ssl_transfer.project.eval import EvalConfig

    config = EvalConfig(knn_k=20, knn_temperature=0.07)
    assert config.knn_k == 20
    assert config.knn_temperature == 0.07


def test_explainability_configs() -> None:
    """Test explainability configurations."""
    from projects.vision_ssl_transfer.project.explainability import (
        AttentionConfig,
        SHAPConfig,
        UMAPConfig,
    )

    shap_config = SHAPConfig(method="deep", n_samples=100)
    assert shap_config.method == "deep"

    attention_config = AttentionConfig(head_fusion="mean")
    assert attention_config.head_fusion == "mean"

    umap_config = UMAPConfig(n_neighbors=15, metric="cosine")
    assert umap_config.metric == "cosine"


def test_corruption_functions() -> None:
    """Test image corruption functions for robustness evaluation."""
    from projects.vision_ssl_transfer.project.eval import apply_corruption

    images = torch.rand(2, 3, 32, 32)

    # Test gaussian noise
    noisy = apply_corruption(images, "gaussian_noise", severity=1)
    assert noisy.shape == images.shape
    assert not torch.allclose(noisy, images)

    # Test contrast
    low_contrast = apply_corruption(images, "contrast", severity=2)
    assert low_contrast.shape == images.shape

    # Test occlusion
    occluded = apply_corruption(images, "occlusion", severity=0.2)
    assert occluded.shape == images.shape
