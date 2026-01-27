"""Smoke tests for template project."""

import pytest


def test_imports() -> None:
    """Test that project modules can be imported."""
    from projects._template.project import data, eval, model, train  # noqa: F401


def test_config_loads() -> None:
    """Test that default config loads without errors."""
    from omegaconf import OmegaConf

    from ml_portfolio.common.paths import get_project_paths

    paths = get_project_paths("_template")
    config_path = paths.default_config()

    if config_path.exists():
        config = OmegaConf.load(config_path)
        assert config is not None
        assert "project" in config


@pytest.mark.skip(reason="Model not implemented")
def test_model_creation() -> None:
    """Test that model can be created."""
    from projects._template.project.model import ModelConfig, TemplateModel

    config = ModelConfig(name="test")
    model = TemplateModel(config)
    assert model is not None


@pytest.mark.skip(reason="Model not implemented")
def test_model_forward() -> None:
    """Test that model forward pass works."""
    import torch
    from projects._template.project.model import ModelConfig, TemplateModel

    config = ModelConfig(name="test")
    model = TemplateModel(config)

    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)

    # Forward pass
    output = model(dummy_input)
    assert output is not None
