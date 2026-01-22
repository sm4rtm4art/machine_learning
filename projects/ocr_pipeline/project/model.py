"""TrOCR model wrapper."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from ml_portfolio.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelConfig:
    """TrOCR model configuration."""

    name: str = "microsoft/trocr-base-printed"
    max_length: int = 128
    num_beams: int = 4
    use_cache: bool = True


@dataclass
class PredictionResult:
    """Result of a single prediction."""

    text: str
    confidence: float
    tokens: list[int] | None = None


class TrOCRWrapper:
    """Wrapper for TrOCR model with prediction and export utilities."""

    def __init__(
        self,
        config: ModelConfig,
        device: str = "cpu",
    ) -> None:
        """Initialize TrOCR wrapper.

        Args:
            config: Model configuration.
            device: Device to run on ('cpu', 'cuda', 'mps').
        """
        self.config = config
        self.device = device

        logger.info(f"Loading TrOCR model: {config.name}")
        self.processor = TrOCRProcessor.from_pretrained(config.name)
        self.model = VisionEncoderDecoderModel.from_pretrained(config.name)
        self.model = self.model.to(device)
        self.model.eval()

    @classmethod
    def from_config(cls, config: ModelConfig, device: str = "cpu") -> "TrOCRWrapper":
        """Create wrapper from configuration.

        Args:
            config: Model configuration.
            device: Device to use.

        Returns:
            Initialized TrOCRWrapper.
        """
        return cls(config, device)

    @classmethod
    def from_pretrained(cls, path: Path, device: str = "cpu") -> "TrOCRWrapper":
        """Load wrapper from saved checkpoint.

        Args:
            path: Path to saved model directory.
            device: Device to use.

        Returns:
            Loaded TrOCRWrapper.
        """
        config = ModelConfig(name=str(path))
        wrapper = cls(config, device)
        wrapper.model = VisionEncoderDecoderModel.from_pretrained(path)
        wrapper.model = wrapper.model.to(device)
        return wrapper

    def predict(
        self,
        image: Image.Image | torch.Tensor,
        return_confidence: bool = True,
    ) -> PredictionResult:
        """Predict text from a single image.

        Args:
            image: PIL Image or preprocessed tensor.
            return_confidence: Whether to compute confidence score.

        Returns:
            PredictionResult with text and confidence.
        """
        self.model.eval()

        with torch.no_grad():
            # Process image
            if isinstance(image, Image.Image):
                pixel_values = self.processor(
                    images=image, return_tensors="pt"
                ).pixel_values
            else:
                pixel_values = image.unsqueeze(0) if image.dim() == 3 else image

            pixel_values = pixel_values.to(self.device)

            # Generate with scores for confidence
            outputs = self.model.generate(
                pixel_values,
                max_length=self.config.max_length,
                num_beams=self.config.num_beams,
                return_dict_in_generate=True,
                output_scores=return_confidence,
            )

            # Decode text
            generated_ids = outputs.sequences
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # Compute confidence (average token probability)
            confidence = 1.0
            if return_confidence and hasattr(outputs, "scores") and outputs.scores:
                scores = torch.stack(outputs.scores, dim=1)
                probs = torch.softmax(scores, dim=-1)
                token_probs = probs.gather(2, generated_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
                confidence = float(token_probs.mean())

            return PredictionResult(
                text=text,
                confidence=confidence,
                tokens=generated_ids[0].tolist(),
            )

    def predict_batch(
        self,
        images: list[Image.Image] | torch.Tensor,
    ) -> list[PredictionResult]:
        """Predict text from a batch of images.

        Args:
            images: List of PIL Images or batch tensor.

        Returns:
            List of PredictionResults.
        """
        self.model.eval()

        with torch.no_grad():
            # Process images
            if isinstance(images, list):
                pixel_values = self.processor(
                    images=images, return_tensors="pt"
                ).pixel_values
            else:
                pixel_values = images

            pixel_values = pixel_values.to(self.device)

            # Generate
            generated_ids = self.model.generate(
                pixel_values,
                max_length=self.config.max_length,
                num_beams=self.config.num_beams,
            )

            # Decode
            texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

            return [
                PredictionResult(text=text, confidence=1.0)
                for text in texts
            ]

    def save(self, path: Path) -> None:
        """Save model to directory.

        Args:
            path: Output directory.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)
        logger.info(f"Saved model to {path}")

    def export_onnx(
        self,
        output_path: Path,
        opset_version: int = 14,
    ) -> Path:
        """Export model to ONNX format.

        Args:
            output_path: Output ONNX file path.
            opset_version: ONNX opset version.

        Returns:
            Path to exported ONNX file.
        """
        import onnx

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create dummy input
        dummy_image = torch.randn(1, 3, 384, 384).to(self.device)

        # Export encoder
        logger.info(f"Exporting to ONNX: {output_path}")

        # Note: Full TrOCR ONNX export is complex due to decoder
        # This exports just the encoder for now
        torch.onnx.export(
            self.model.encoder,
            dummy_image,
            str(output_path),
            opset_version=opset_version,
            input_names=["pixel_values"],
            output_names=["encoder_output"],
            dynamic_axes={
                "pixel_values": {0: "batch_size"},
                "encoder_output": {0: "batch_size"},
            },
        )

        # Validate
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)

        logger.info(f"ONNX export complete: {output_path}")
        return output_path


class ONNXRunner:
    """ONNX Runtime inference runner for TrOCR."""

    def __init__(self, model_path: Path, processor_path: Path) -> None:
        """Initialize ONNX runner.

        Args:
            model_path: Path to ONNX model.
            processor_path: Path to saved processor.
        """
        import onnxruntime as ort

        self.processor = TrOCRProcessor.from_pretrained(processor_path)
        self.session = ort.InferenceSession(str(model_path))

        logger.info(f"Loaded ONNX model from {model_path}")

    def predict(self, image: Image.Image) -> str:
        """Run inference on a single image.

        Args:
            image: PIL Image.

        Returns:
            Predicted text.
        """
        # Process image
        pixel_values = self.processor(images=image, return_tensors="np").pixel_values

        # Run inference
        outputs = self.session.run(None, {"pixel_values": pixel_values})

        # Note: This only runs the encoder
        # Full inference requires decoder implementation
        return "ONNX inference not fully implemented"
