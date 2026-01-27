"""SROIE dataset handling."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

from ml_portfolio.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SROIESample:
    """A single SROIE dataset sample."""

    image_path: Path
    text: str
    field_type: str | None = None  # company, date, address, total
    metadata: dict[str, Any] | None = None


@dataclass
class DataConfig:
    """Data configuration."""

    data_dir: Path
    train_split: float = 0.85
    val_split: float = 0.15
    max_samples: int | None = None
    seed: int = 42


class SROIEDataset(Dataset[dict[str, Any]]):  # type: ignore[misc]
    """SROIE receipts dataset for OCR.

    The SROIE dataset contains receipt images with structured fields:
    - company: Company/store name
    - date: Transaction date
    - address: Store address
    - total: Total amount

    Each receipt can have multiple text lines extracted.
    """

    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        processor: Any | None = None,
        transform: Any | None = None,
        max_samples: int | None = None,
    ) -> None:
        """Initialize SROIE dataset.

        Args:
            data_dir: Path to SROIE data directory.
            split: Data split ('train', 'val', 'test').
            processor: TrOCR processor for tokenization.
            transform: Optional image transform.
            max_samples: Maximum samples to load (for debugging).
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.processor = processor
        self.transform = transform
        self.max_samples = max_samples

        self.samples: list[SROIESample] = []
        self._load_data()

    def _load_data(self) -> None:
        """Load data from SROIE directory structure."""
        # SROIE structure:
        # data_dir/
        #   train/
        #     img/
        #       X51005365187.jpg
        #     box/
        #       X51005365187.txt  (bounding boxes)
        #     entities/
        #       X51005365187.txt  (structured fields)
        #   test/
        #     ...

        split_dir = self.data_dir / ("train" if self.split in ["train", "val"] else "test")
        img_dir = split_dir / "img"
        box_dir = split_dir / "box"

        if not img_dir.exists():
            logger.warning(f"Image directory not found: {img_dir}")
            return

        # Load all image-text pairs
        all_samples = []
        for img_path in sorted(img_dir.glob("*.jpg")):
            box_path = box_dir / f"{img_path.stem}.txt"

            if not box_path.exists():
                continue

            # Parse box file (format: x1,y1,x2,y2,x3,y3,x4,y4,text)
            try:
                with open(box_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        # Split by comma, last part is text
                        parts = line.split(",")
                        if len(parts) >= 9:
                            text = ",".join(parts[8:])  # Text might contain commas
                            all_samples.append(
                                SROIESample(
                                    image_path=img_path,
                                    text=text,
                                    metadata={"box_coords": parts[:8]},
                                )
                            )
            except Exception as e:
                logger.warning(f"Error loading {box_path}: {e}")

        # Split train/val
        if self.split in ["train", "val"]:
            import random

            random.seed(42)
            random.shuffle(all_samples)

            split_idx = int(len(all_samples) * 0.85)
            if self.split == "train":
                all_samples = all_samples[:split_idx]
            else:
                all_samples = all_samples[split_idx:]

        # Apply max_samples limit
        if self.max_samples:
            all_samples = all_samples[: self.max_samples]

        self.samples = all_samples
        logger.info(f"Loaded {len(self.samples)} samples for {self.split} split")

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with:
                - pixel_values: Processed image tensor
                - labels: Tokenized text (if processor provided)
                - text: Original text string
                - image_path: Path to image file
        """
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample.image_path).convert("RGB")

        # Apply custom transform
        if self.transform is not None:
            image = self.transform(image)

        result: dict[str, Any] = {
            "text": sample.text,
            "image_path": str(sample.image_path),
        }

        # Process with TrOCR processor
        if self.processor is not None:
            # Process image
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
            result["pixel_values"] = pixel_values.squeeze(0)

            # Tokenize text
            labels = self.processor.tokenizer(
                sample.text,
                padding="max_length",
                max_length=128,
                truncation=True,
                return_tensors="pt",
            ).input_ids
            result["labels"] = labels.squeeze(0)

        if sample.field_type:
            result["field_type"] = sample.field_type

        if sample.metadata:
            result["metadata"] = sample.metadata

        return result

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over samples."""
        for i in range(len(self)):
            yield self[i]


def create_dataloaders(
    data_dir: Path,
    processor: Any,
    config: DataConfig,
    batch_size: int = 8,
) -> tuple[Any, Any, Any]:
    """Create train, validation, and test dataloaders.

    Args:
        data_dir: Path to SROIE data.
        processor: TrOCR processor.
        config: Data configuration.
        batch_size: Batch size for dataloaders.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    from torch.utils.data import DataLoader

    train_dataset = SROIEDataset(
        data_dir,
        split="train",
        processor=processor,
        max_samples=config.max_samples,
    )

    val_dataset = SROIEDataset(
        data_dir,
        split="val",
        processor=processor,
        max_samples=config.max_samples,
    )

    test_dataset = SROIEDataset(
        data_dir,
        split="test",
        processor=processor,
        max_samples=config.max_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
