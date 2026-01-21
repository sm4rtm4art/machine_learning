"""Dataset handling template."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from torch.utils.data import Dataset


@dataclass
class DataConfig:
    """Data configuration."""

    data_dir: Path
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    seed: int = 42


class TemplateDataset(Dataset[dict[str, Any]]):
    """Template dataset class.

    Customize this for your specific data format.
    """

    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        transform: Any | None = None,
    ) -> None:
        """Initialize dataset.

        Args:
            data_dir: Path to data directory.
            split: Data split ('train', 'val', 'test').
            transform: Optional transform to apply.
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform

        # Load your data here
        self.samples: list[dict[str, Any]] = []
        self._load_data()

    def _load_data(self) -> None:
        """Load data from disk.

        Override this method for your data format.
        """
        # Example: Load from a manifest file
        # manifest_path = self.data_dir / f"{self.split}_manifest.json"
        # with open(manifest_path) as f:
        #     self.samples = json.load(f)
        pass

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with sample data.
        """
        sample = self.samples[idx]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over samples."""
        for i in range(len(self)):
            yield self[i]


def create_data_splits(
    data_dir: Path,
    config: DataConfig,
) -> tuple[TemplateDataset, TemplateDataset, TemplateDataset]:
    """Create train/val/test splits.

    Args:
        data_dir: Path to data directory.
        config: Data configuration.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    train_dataset = TemplateDataset(data_dir, split="train")
    val_dataset = TemplateDataset(data_dir, split="val")
    test_dataset = TemplateDataset(data_dir, split="test")

    return train_dataset, val_dataset, test_dataset
