#!/usr/bin/env python
"""Download and prepare datasets for Vision SSL Transfer.

Supports:
- Oxford-IIIT Pet dataset (benchmark)
- Custom tortoise dataset setup
"""

from pathlib import Path

import typer
from rich.console import Console

from ml_portfolio.common.logging import get_logger, setup_logging
from ml_portfolio.common.paths import get_project_paths

app = typer.Typer()
console = Console()
logger = get_logger(__name__)

PROJECT_NAME = "vision_ssl_transfer"


def download_oxford_pets(output_dir: Path, force: bool = False) -> None:
    """Download Oxford-IIIT Pet dataset.

    Args:
        output_dir: Output directory.
        force: Force re-download.
    """
    import tarfile
    import urllib.request

    images_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
    annotations_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"

    images_tar = output_dir / "images.tar.gz"
    annotations_tar = output_dir / "annotations.tar.gz"

    # Download images
    if not images_tar.exists() or force:
        console.print("[blue]Downloading images...[/blue]")
        urllib.request.urlretrieve(images_url, images_tar)
        console.print("[green]Images downloaded[/green]")
    else:
        console.print("[yellow]Images already downloaded[/yellow]")

    # Download annotations
    if not annotations_tar.exists() or force:
        console.print("[blue]Downloading annotations...[/blue]")
        urllib.request.urlretrieve(annotations_url, annotations_tar)
        console.print("[green]Annotations downloaded[/green]")
    else:
        console.print("[yellow]Annotations already downloaded[/yellow]")

    # Extract
    console.print("[blue]Extracting...[/blue]")

    with tarfile.open(images_tar, "r:gz") as tar:
        tar.extractall(output_dir)

    with tarfile.open(annotations_tar, "r:gz") as tar:
        tar.extractall(output_dir)

    console.print("[green]Extraction complete[/green]")

    # Create splits
    create_splits(output_dir)


def create_splits(data_dir: Path) -> None:
    """Create train/val/test splits from annotations.

    Args:
        data_dir: Data directory with annotations.
    """
    import random

    annotations_dir = data_dir / "annotations"
    trainval_file = annotations_dir / "trainval.txt"
    test_file = annotations_dir / "test.txt"

    if not trainval_file.exists():
        console.print("[yellow]No annotations found, skipping split creation[/yellow]")
        return

    # Read trainval samples
    with open(trainval_file) as f:
        trainval_lines = [line.strip().split()[0] for line in f if line.strip()]

    # Read test samples
    with open(test_file) as f:
        test_lines = [line.strip().split()[0] for line in f if line.strip()]

    # Split trainval into train and val
    random.seed(42)
    random.shuffle(trainval_lines)

    split_idx = int(len(trainval_lines) * 0.9)
    train_lines = trainval_lines[:split_idx]
    val_lines = trainval_lines[split_idx:]

    # Write split files
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(exist_ok=True)

    with open(splits_dir / "train.txt", "w") as f:
        f.write("\n".join(train_lines))

    with open(splits_dir / "val.txt", "w") as f:
        f.write("\n".join(val_lines))

    with open(splits_dir / "test.txt", "w") as f:
        f.write("\n".join(test_lines))

    console.print(
        f"[green]Created splits: {len(train_lines)} train, {len(val_lines)} val, {len(test_lines)} test[/green]"
    )


@app.command()
def main(
    dataset: str = typer.Option(
        "oxford_pets",
        "--dataset",
        "-d",
        help="Dataset to download: oxford_pets, tortoise_setup",
    ),
    output_dir: Path = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory. Defaults to data/vision_ssl_transfer/",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-download even if data exists",
    ),
) -> None:
    """Download and prepare datasets."""
    setup_logging()

    paths = get_project_paths(PROJECT_NAME)
    output_dir = output_dir or paths.data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {dataset} to {output_dir}")

    if dataset == "oxford_pets":
        # Check if already exists
        if (output_dir / "images").exists() and not force:
            console.print("[yellow]Oxford Pets already downloaded. Use --force to re-download.[/yellow]")
            return

        download_oxford_pets(output_dir, force)

    elif dataset == "tortoise_setup":
        # Just create directory structure for manual data placement
        tortoise_dir = output_dir / "tortoise"
        tortoise_dir.mkdir(exist_ok=True)

        (tortoise_dir / "unlabeled").mkdir(exist_ok=True)
        (tortoise_dir / "labeled" / "present").mkdir(parents=True, exist_ok=True)
        (tortoise_dir / "labeled" / "absent").mkdir(parents=True, exist_ok=True)

        console.print("[green]Created tortoise dataset directory structure:[/green]")
        console.print(f"  {tortoise_dir}/unlabeled/     - Place unlabeled images here")
        console.print(f"  {tortoise_dir}/labeled/present/ - Images with tortoise")
        console.print(f"  {tortoise_dir}/labeled/absent/  - Images without tortoise")
        console.print("")
        console.print("[yellow]Note: Sort images by timestamp before placing![/yellow]")

    else:
        console.print(f"[red]Unknown dataset: {dataset}[/red]")
        raise typer.Exit(1)

    logger.info("Data download complete")


if __name__ == "__main__":
    app()
