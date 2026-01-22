"""Image preprocessing for OCR pipeline."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageFilter, ImageOps

from ml_portfolio.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PreprocessConfig:
    """Preprocessing configuration."""

    width: int = 384
    height: int = 384
    normalize: bool = True
    deskew: bool = False
    padding_color: tuple[int, int, int] = (255, 255, 255)


class ImagePreprocessor:
    """Image preprocessor for OCR."""

    def __init__(self, config: PreprocessConfig) -> None:
        """Initialize preprocessor.

        Args:
            config: Preprocessing configuration.
        """
        self.config = config

    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply preprocessing pipeline.

        Args:
            image: Input PIL Image.

        Returns:
            Preprocessed PIL Image.
        """
        return self.preprocess(image)

    def preprocess(self, image: Image.Image) -> Image.Image:
        """Apply full preprocessing pipeline.

        Args:
            image: Input PIL Image.

        Returns:
            Preprocessed PIL Image.
        """
        # Convert to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Deskew if enabled
        if self.config.deskew:
            image = self.deskew(image)

        # Resize with padding to maintain aspect ratio
        image = self.resize_with_padding(image)

        return image

    def resize_with_padding(self, image: Image.Image) -> Image.Image:
        """Resize image with padding to target size.

        Maintains aspect ratio by adding padding.

        Args:
            image: Input image.

        Returns:
            Resized and padded image.
        """
        target_w, target_h = self.config.width, self.config.height
        orig_w, orig_h = image.size

        # Calculate scale to fit within target
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        # Resize
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Create padded image
        padded = Image.new("RGB", (target_w, target_h), self.config.padding_color)

        # Paste resized image centered
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        padded.paste(image, (paste_x, paste_y))

        return padded

    def deskew(self, image: Image.Image) -> Image.Image:
        """Deskew image using simple heuristics.

        Args:
            image: Input image.

        Returns:
            Deskewed image.
        """
        # Convert to grayscale for angle detection
        gray = image.convert("L")

        # Simple approach: try rotations and find minimum bounding box
        # More sophisticated methods would use Hough transform
        best_angle = 0
        min_width = image.size[0]

        for angle in range(-10, 11, 1):
            rotated = gray.rotate(angle, expand=True, fillcolor=255)
            bbox = rotated.getbbox()
            if bbox:
                width = bbox[2] - bbox[0]
                if width < min_width:
                    min_width = width
                    best_angle = angle

        if best_angle != 0:
            image = image.rotate(best_angle, expand=True, fillcolor=self.config.padding_color)

        return image


# Robustness perturbations for testing
def apply_gaussian_blur(image: Image.Image, sigma: float) -> Image.Image:
    """Apply Gaussian blur.

    Args:
        image: Input image.
        sigma: Blur sigma (standard deviation).

    Returns:
        Blurred image.
    """
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))


def apply_rotation(image: Image.Image, degrees: float) -> Image.Image:
    """Apply rotation.

    Args:
        image: Input image.
        degrees: Rotation angle in degrees.

    Returns:
        Rotated image.
    """
    return image.rotate(degrees, expand=False, fillcolor=(255, 255, 255))


def apply_jpeg_compression(image: Image.Image, quality: int) -> Image.Image:
    """Apply JPEG compression artifacts.

    Args:
        image: Input image.
        quality: JPEG quality (1-100).

    Returns:
        Compressed image.
    """
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def apply_brightness(image: Image.Image, factor: float) -> Image.Image:
    """Apply brightness adjustment.

    Args:
        image: Input image.
        factor: Brightness factor (1.0 = original, <1 darker, >1 brighter).

    Returns:
        Adjusted image.
    """
    from PIL import ImageEnhance

    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def apply_contrast(image: Image.Image, factor: float) -> Image.Image:
    """Apply contrast adjustment.

    Args:
        image: Input image.
        factor: Contrast factor (1.0 = original).

    Returns:
        Adjusted image.
    """
    from PIL import ImageEnhance

    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def apply_noise(image: Image.Image, intensity: float = 0.1) -> Image.Image:
    """Apply random noise.

    Args:
        image: Input image.
        intensity: Noise intensity (0-1).

    Returns:
        Noisy image.
    """
    arr = np.array(image).astype(np.float32)
    noise = np.random.normal(0, intensity * 255, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# Perturbation registry for robustness evaluation
PERTURBATIONS = {
    "gaussian_blur": apply_gaussian_blur,
    "rotation": apply_rotation,
    "jpeg_compression": apply_jpeg_compression,
    "brightness": apply_brightness,
    "contrast": apply_contrast,
    "noise": apply_noise,
}


def apply_perturbation(
    image: Image.Image,
    perturbation_name: str,
    intensity: float,
) -> Image.Image:
    """Apply a named perturbation.

    Args:
        image: Input image.
        perturbation_name: Name of perturbation to apply.
        intensity: Intensity parameter for the perturbation.

    Returns:
        Perturbed image.
    """
    if perturbation_name not in PERTURBATIONS:
        raise ValueError(f"Unknown perturbation: {perturbation_name}")

    return PERTURBATIONS[perturbation_name](image, intensity)
