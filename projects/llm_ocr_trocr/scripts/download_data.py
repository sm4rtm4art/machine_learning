#!/usr/bin/env python
"""Download and prepare SROIE dataset."""

import shutil
import zipfile
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress

from ml_portfolio.common.logging import get_logger, setup_logging
from ml_portfolio.common.paths import get_project_paths

app = typer.Typer()
console = Console()
logger = get_logger(__name__)

PROJECT_NAME = "llm_ocr_trocr"

# SROIE dataset URLs (from ICDAR 2019 challenge)
# Note: Actual URLs would need to be obtained from the official source
SROIE_INFO = """
SROIE Dataset Download Instructions
====================================

The SROIE (Scanned Receipts OCR and Information Extraction) dataset
is from the ICDAR 2019 Robust Reading Challenge.

Official source: https://rrc.cvc.uab.es/?ch=13

To download:
1. Register at https://rrc.cvc.uab.es/
2. Navigate to Challenge 13 (SROIE)
3. Download the training and test data
4. Extract to: {data_dir}

Expected structure:
{data_dir}/
  train/
    img/       # Receipt images (*.jpg)
    box/       # Text bounding boxes (*.txt)
    entities/  # Structured field labels (*.txt)
  test/
    img/
    box/
"""


@app.command()
def main(
    output_dir: Path = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for data. Defaults to data/llm_ocr_trocr/",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-download even if data exists",
    ),
) -> None:
    """Download and prepare the SROIE dataset."""
    setup_logging()

    paths = get_project_paths(PROJECT_NAME)
    output_dir = output_dir or paths.data_dir

    logger.info(f"Preparing data directory: {output_dir}")

    # Check if data already exists
    train_dir = output_dir / "train" / "img"
    if train_dir.exists() and any(train_dir.iterdir()) and not force:
        console.print("[yellow]Data already exists. Use --force to re-download.[/yellow]")
        console.print(f"\nData location: {output_dir}")

        # Show dataset stats
        train_images = list((output_dir / "train" / "img").glob("*.jpg"))
        console.print(f"Training images: {len(train_images)}")

        if (output_dir / "test" / "img").exists():
            test_images = list((output_dir / "test" / "img").glob("*.jpg"))
            console.print(f"Test images: {len(test_images)}")

        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Show download instructions
    console.print(SROIE_INFO.format(data_dir=output_dir))

    # Create directory structure
    for split in ["train", "test"]:
        for subdir in ["img", "box", "entities"]:
            (output_dir / split / subdir).mkdir(parents=True, exist_ok=True)

    console.print(f"\n[green]Created directory structure at: {output_dir}[/green]")
    console.print("\nPlease download the dataset manually and extract it to this location.")

    # Check if user has placed data
    console.print("\n[yellow]After downloading, run this script again to verify the data.[/yellow]")


@app.command()
def verify(
    data_dir: Path = typer.Option(
        None,
        "--data-dir",
        "-d",
        help="Data directory to verify",
    ),
) -> None:
    """Verify SROIE dataset structure and contents."""
    setup_logging()

    paths = get_project_paths(PROJECT_NAME)
    data_dir = data_dir or paths.data_dir

    console.print(f"Verifying dataset at: {data_dir}\n")

    issues = []

    # Check train directory
    train_img = data_dir / "train" / "img"
    train_box = data_dir / "train" / "box"

    if not train_img.exists():
        issues.append("Missing train/img directory")
    else:
        train_images = list(train_img.glob("*.jpg"))
        console.print(f"Training images: {len(train_images)}")

        if not train_box.exists():
            issues.append("Missing train/box directory")
        else:
            train_boxes = list(train_box.glob("*.txt"))
            console.print(f"Training box files: {len(train_boxes)}")

            # Check alignment
            img_ids = {f.stem for f in train_images}
            box_ids = {f.stem for f in train_boxes}

            missing_boxes = img_ids - box_ids
            if missing_boxes:
                issues.append(f"Missing box files for {len(missing_boxes)} images")

    # Check test directory
    test_img = data_dir / "test" / "img"
    if test_img.exists():
        test_images = list(test_img.glob("*.jpg"))
        console.print(f"Test images: {len(test_images)}")

    # Report
    if issues:
        console.print("\n[red]Issues found:[/red]")
        for issue in issues:
            console.print(f"  - {issue}")
    else:
        console.print("\n[green]Dataset verification passed![/green]")


if __name__ == "__main__":
    app()
