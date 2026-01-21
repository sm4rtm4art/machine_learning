#!/usr/bin/env python
"""Download and prepare dataset."""

from pathlib import Path

import typer
from rich.console import Console

from ml_portfolio.common.logging import get_logger, setup_logging
from ml_portfolio.common.paths import get_project_paths

app = typer.Typer()
console = Console()
logger = get_logger(__name__)

PROJECT_NAME = "_template"


@app.command()
def main(
    output_dir: Path = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for data. Defaults to data/<project>/",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-download even if data exists",
    ),
) -> None:
    """Download and prepare the dataset."""
    setup_logging()

    paths = get_project_paths(PROJECT_NAME)
    output_dir = output_dir or paths.data_dir

    logger.info(f"Downloading data to {output_dir}")

    # Check if data already exists
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        console.print("[yellow]Data already exists. Use --force to re-download.[/yellow]")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Implement your data download logic here
    # Example:
    # download_url("https://example.com/dataset.zip", output_dir / "raw.zip")
    # extract_archive(output_dir / "raw.zip", output_dir)
    # prepare_splits(output_dir)

    console.print("[red]Not implemented: Add data download logic for your dataset[/red]")

    logger.info("Data download complete")


if __name__ == "__main__":
    app()
