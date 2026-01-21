"""Smoke tests for TrOCR OCR project."""

import pytest


def test_imports() -> None:
    """Test that project modules can be imported."""
    from projects.llm_ocr_trocr.project import data, model  # noqa: F401
    from projects.llm_ocr_trocr.project import preprocess, postprocess  # noqa: F401
    from projects.llm_ocr_trocr.project import train, eval  # noqa: F401


def test_config_loads() -> None:
    """Test that default config loads without errors."""
    from omegaconf import OmegaConf
    from ml_portfolio.common.paths import get_project_paths

    paths = get_project_paths("llm_ocr_trocr")
    config_path = paths.default_config()

    config = OmegaConf.load(config_path)
    assert config is not None
    assert "model" in config
    assert "training" in config
    assert "evaluation" in config


def test_preprocessing() -> None:
    """Test image preprocessing functions."""
    from PIL import Image
    from projects.llm_ocr_trocr.project.preprocess import (
        ImagePreprocessor,
        PreprocessConfig,
        apply_gaussian_blur,
        apply_rotation,
    )

    # Create dummy image
    image = Image.new("RGB", (100, 50), color="white")

    # Test preprocessor
    config = PreprocessConfig(width=384, height=384)
    preprocessor = ImagePreprocessor(config)
    processed = preprocessor(image)

    assert processed.size == (384, 384)
    assert processed.mode == "RGB"

    # Test perturbations
    blurred = apply_gaussian_blur(image, sigma=1.0)
    assert blurred.size == image.size

    rotated = apply_rotation(image, degrees=5)
    assert rotated.size == image.size


def test_postprocessing() -> None:
    """Test text postprocessing functions."""
    from projects.llm_ocr_trocr.project.postprocess import (
        TextPostprocessor,
        SROIEFieldExtractor,
    )

    # Test postprocessor
    processor = TextPostprocessor()
    assert processor("  hello  world  ") == "hello world"
    assert processor("test\n\ntext") == "test text"

    # Test field extraction
    assert SROIEFieldExtractor.extract_date("Date: 15-01-2024") == "15-01-2024"
    assert SROIEFieldExtractor.extract_total("Total: $12.50") == "12.50"


def test_ocr_metrics() -> None:
    """Test OCR metrics computation."""
    from ml_portfolio.metrics.ocr import compute_cer, compute_wer, compute_ocr_metrics

    # Perfect match
    assert compute_cer("hello", "hello") == 0.0
    assert compute_wer("hello world", "hello world") == 0.0

    # Single character error
    cer = compute_cer("hello", "hallo")
    assert 0 < cer < 1

    # Word error
    wer = compute_wer("hello world", "hello there")
    assert wer == 0.5  # 1 of 2 words wrong

    # Batch metrics
    metrics = compute_ocr_metrics(
        predictions=["hello", "world"],
        references=["hello", "world"],
    )
    assert metrics.cer == 0.0
    assert metrics.exact_match == 1.0


@pytest.mark.skip(reason="Requires model download")
def test_model_creation() -> None:
    """Test that TrOCR model can be created."""
    from projects.llm_ocr_trocr.project.model import TrOCRWrapper, ModelConfig

    config = ModelConfig(name="microsoft/trocr-base-printed")
    wrapper = TrOCRWrapper(config, device="cpu")
    assert wrapper is not None


@pytest.mark.skip(reason="Requires model download")
def test_model_inference() -> None:
    """Test model inference on dummy image."""
    from PIL import Image
    from projects.llm_ocr_trocr.project.model import TrOCRWrapper, ModelConfig

    config = ModelConfig(name="microsoft/trocr-base-printed")
    wrapper = TrOCRWrapper(config, device="cpu")

    # Create dummy image with text-like content
    image = Image.new("RGB", (200, 50), color="white")

    result = wrapper.predict(image)
    assert result is not None
    assert isinstance(result.text, str)
    assert 0 <= result.confidence <= 1
