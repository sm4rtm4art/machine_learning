"""Dataset handling for Vision SSL Transfer.

Supports:
- Oxford-IIIT Pet dataset (benchmark)
- Custom tortoise dataset (real-world validation)
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


@dataclass
class DataConfig:
    """Data configuration."""

    data_dir: Path
    dataset: str = "oxford_pets"
    image_size: int = 224
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    seed: int = 42
    augmentation: dict[str, Any] = field(default_factory=dict)


class SSLAugmentation:
    """SSL-style augmentations for contrastive learning.

    Returns two augmented views of the same image.
    """

    def __init__(
        self,
        image_size: int = 224,
        color_jitter: float = 0.8,
        grayscale_prob: float = 0.2,
        gaussian_blur_prob: float = 0.5,
        horizontal_flip_prob: float = 0.5,
    ) -> None:
        """Initialize augmentation pipeline.

        Args:
            image_size: Output image size.
            color_jitter: Strength of color jitter.
            grayscale_prob: Probability of grayscale conversion.
            gaussian_blur_prob: Probability of Gaussian blur.
            horizontal_flip_prob: Probability of horizontal flip.
        """
        self.image_size = image_size
        self.color_jitter = color_jitter
        self.grayscale_prob = grayscale_prob
        self.gaussian_blur_prob = gaussian_blur_prob
        self.horizontal_flip_prob = horizontal_flip_prob

        # Lazy import to avoid dependency issues in tests
        self._transform: Callable[[Image.Image], torch.Tensor] | None = None

    def _build_transform(self) -> Callable[[Image.Image], torch.Tensor]:
        """Build the augmentation transform lazily."""
        from torchvision import transforms

        transform: Callable[[Image.Image], torch.Tensor] = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.image_size, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(p=self.horizontal_flip_prob),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4 * self.color_jitter,
                            contrast=0.4 * self.color_jitter,
                            saturation=0.4 * self.color_jitter,
                            hue=0.1 * self.color_jitter,
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=self.grayscale_prob),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))],
                    p=self.gaussian_blur_prob,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        return transform

    @property
    def transform(self) -> Callable[[Image.Image], torch.Tensor]:
        """Get or build the transform."""
        if self._transform is None:
            self._transform = self._build_transform()
        return self._transform

    def __call__(self, image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply augmentation twice to get two views.

        Args:
            image: Input PIL image.

        Returns:
            Tuple of two augmented tensor views.
        """
        view1 = self.transform(image)
        view2 = self.transform(image)
        return view1, view2


class EvalAugmentation:
    """Standard evaluation augmentation (center crop, normalize)."""

    def __init__(self, image_size: int = 224) -> None:
        """Initialize evaluation transform.

        Args:
            image_size: Output image size.
        """
        self.image_size = image_size
        self._transform: Callable[[Image.Image], torch.Tensor] | None = None

    def _build_transform(self) -> Callable[[Image.Image], torch.Tensor]:
        """Build the evaluation transform lazily."""
        from torchvision import transforms

        transform: Callable[[Image.Image], torch.Tensor] = transforms.Compose(
            [
                transforms.Resize(int(self.image_size * 1.14)),  # 256 for 224
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        return transform

    @property
    def transform(self) -> Callable[[Image.Image], torch.Tensor]:
        """Get or build the transform."""
        if self._transform is None:
            self._transform = self._build_transform()
        return self._transform

    def __call__(self, image: Image.Image) -> torch.Tensor:
        """Apply evaluation transform.

        Args:
            image: Input PIL image.

        Returns:
            Normalized tensor.
        """
        return self.transform(image)


class OxfordPetsDataset(Dataset[dict[str, Any]]):  # type: ignore[misc]
    """Oxford-IIIT Pet Dataset wrapper.

    Supports both SSL (unlabeled) and supervised (labeled) modes.
    """

    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        transform: Callable[..., Any] | None = None,
        labeled: bool = True,
    ) -> None:
        """Initialize dataset.

        Args:
            data_dir: Path to dataset directory.
            split: Data split ('train', 'val', 'test').
            transform: Transform to apply to images.
            labeled: Whether to return labels (False for SSL pretraining).
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.labeled = labeled

        self.samples: list[dict[str, Any]] = []
        self._load_data()

    def _load_data(self) -> None:
        """Load dataset from disk.

        Expected structure:
        data_dir/
            images/
                Abyssinian_1.jpg
                ...
            annotations/
                trainval.txt
                test.txt
        """
        # Placeholder: actual loading will use torchvision or manual parsing
        # For now, we scan the images directory if it exists
        images_dir = self.data_dir / "images"
        if images_dir.exists():
            for img_path in images_dir.glob("*.jpg"):
                # Parse class from filename (e.g., "Abyssinian_123.jpg")
                name = img_path.stem
                parts = name.rsplit("_", 1)
                class_name = parts[0] if len(parts) > 1 else name

                self.samples.append(
                    {
                        "image_path": img_path,
                        "class_name": class_name,
                        "is_cat": class_name[0].isupper(),  # Cats start uppercase
                    }
                )

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with image (and optionally label).
        """
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")

        if self.transform is not None:
            transformed = self.transform(image)
            # SSL transforms return tuple, supervised return tensor
            if isinstance(transformed, tuple):
                return {
                    "view1": transformed[0],
                    "view2": transformed[1],
                    "index": idx,
                }
            image_tensor = transformed
        else:
            # Default: just convert to tensor
            from torchvision import transforms

            image_tensor = transforms.ToTensor()(image)

        result: dict[str, Any] = {"image": image_tensor, "index": idx}

        if self.labeled:
            result["class_name"] = sample["class_name"]
            result["is_cat"] = sample["is_cat"]

        return result

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over samples."""
        for i in range(len(self)):
            yield self[i]


class TortoiseDataset(Dataset[dict[str, Any]]):  # type: ignore[misc]
    """Custom tortoise dataset for real-world validation.

    Handles temporal splitting to avoid data leakage.
    """

    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        transform: Callable[..., Any] | None = None,
        labeled: bool = False,
    ) -> None:
        """Initialize tortoise dataset.

        Args:
            data_dir: Path to tortoise images.
            split: Data split ('train', 'val', 'test').
            transform: Transform to apply.
            labeled: Whether labels are available.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.labeled = labeled

        self.samples: list[dict[str, Any]] = []
        self._load_data()

    def _load_data(self) -> None:
        """Load tortoise images.

        Note: Real implementation should sort by timestamp
        and split by date ranges to avoid leakage.
        """
        # Placeholder: scan for images
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            for img_path in self.data_dir.glob(ext):
                self.samples.append(
                    {
                        "image_path": img_path,
                        "label": None,  # Unlabeled by default
                    }
                )

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single sample."""
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")

        if self.transform is not None:
            transformed = self.transform(image)
            if isinstance(transformed, tuple):
                return {"view1": transformed[0], "view2": transformed[1], "index": idx}
            image_tensor = transformed
        else:
            from torchvision import transforms

            image_tensor = transforms.ToTensor()(image)

        result: dict[str, Any] = {"image": image_tensor, "index": idx}

        if self.labeled and sample["label"] is not None:
            result["label"] = sample["label"]

        return result

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over samples."""
        for i in range(len(self)):
            yield self[i]


def create_ssl_dataloaders(
    data_dir: Path,
    config: DataConfig,
    batch_size: int = 256,
    num_workers: int = 4,
) -> tuple[DataLoader[dict[str, Any]], DataLoader[dict[str, Any]]]:
    """Create dataloaders for SSL pretraining.

    Args:
        data_dir: Path to data directory.
        config: Data configuration.
        batch_size: Batch size.
        num_workers: Number of data loading workers.

    Returns:
        Tuple of (train_loader, val_loader) for SSL.
    """
    ssl_transform = SSLAugmentation(
        image_size=config.image_size,
        **config.augmentation,
    )

    if config.dataset == "oxford_pets":
        train_dataset = OxfordPetsDataset(
            data_dir,
            split="train",
            transform=ssl_transform,
            labeled=False,
        )
        val_dataset = OxfordPetsDataset(
            data_dir,
            split="val",
            transform=EvalAugmentation(config.image_size),
            labeled=False,
        )
    elif config.dataset == "tortoise_custom":
        train_dataset = TortoiseDataset(
            data_dir,
            split="train",
            transform=ssl_transform,
            labeled=False,
        )
        val_dataset = TortoiseDataset(
            data_dir,
            split="val",
            transform=EvalAugmentation(config.image_size),
            labeled=False,
        )
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    train_loader: DataLoader[dict[str, Any]] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader: DataLoader[dict[str, Any]] = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def create_supervised_dataloaders(
    data_dir: Path,
    config: DataConfig,
    batch_size: int = 64,
    num_workers: int = 4,
) -> tuple[DataLoader[dict[str, Any]], DataLoader[dict[str, Any]], DataLoader[dict[str, Any]]]:
    """Create dataloaders for supervised evaluation.

    Args:
        data_dir: Path to data directory.
        config: Data configuration.
        batch_size: Batch size.
        num_workers: Number of data loading workers.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    eval_transform = EvalAugmentation(config.image_size)

    if config.dataset == "oxford_pets":
        train_dataset = OxfordPetsDataset(data_dir, split="train", transform=eval_transform, labeled=True)
        val_dataset = OxfordPetsDataset(data_dir, split="val", transform=eval_transform, labeled=True)
        test_dataset = OxfordPetsDataset(data_dir, split="test", transform=eval_transform, labeled=True)
    else:
        raise ValueError(f"Supervised evaluation not supported for: {config.dataset}")

    train_loader: DataLoader[dict[str, Any]] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader: DataLoader[dict[str, Any]] = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader: DataLoader[dict[str, Any]] = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
