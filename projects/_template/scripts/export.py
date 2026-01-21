#!/usr/bin/env python
"""Export model to deployment format."""

from pathlib import Path

import mlflow
import typer
from rich.console import Console

from ml_portfolio.common.logging import get_logger, setup_logging
from ml_portfolio.common.paths import get_project_paths
from ml_portfolio.tracking.mlflow_utils import setup_mlflow

app = typer.Typer()
console = Console()
logger = get_logger(__name__)

PROJECT_NAME = "_template"


@app.command()
def main(
    run_id: str = typer.Option(
        ...,
        "--run-id",
        "-r",
        help="MLflow run ID to export",
    ),
    output_path: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path for exported model",
    ),
    format_: str = typer.Option(
        "onnx",
        "--format",
        "-f",
        help="Export format (onnx, torchscript)",
    ),
    quantize: bool = typer.Option(
        False,
        "--quantize",
        "-q",
        help="Apply INT8 quantization",
    ),
) -> None:
    """Export model to deployment format."""
    setup_logging()

    paths = get_project_paths(PROJECT_NAME)
    output_path = output_path or paths.artifacts_dir / f"model.{format_}"

    # Load model from MLflow
    setup_mlflow(PROJECT_NAME)
    logger.info(f"Loading model from run {run_id}")
    # model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")

    # Export based on format
    if format_ == "onnx":
        logger.info(f"Exporting to ONNX: {output_path}")
        # export_to_onnx(model, output_path)

        if quantize:
            quantized_path = output_path.with_suffix(".quantized.onnx")
            logger.info(f"Quantizing to INT8: {quantized_path}")
            # quantize_onnx(output_path, quantized_path)

    elif format_ == "torchscript":
        logger.info(f"Exporting to TorchScript: {output_path}")
        # export_to_torchscript(model, output_path)

    else:
        console.print(f"[red]Unknown format: {format_}[/red]")
        raise typer.Exit(1)

    # Log exported model to MLflow
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(str(output_path), f"exports/{format_}")
        if quantize and format_ == "onnx":
            mlflow.log_artifact(str(quantized_path), f"exports/{format_}")

    console.print("[red]Not implemented: Add export logic[/red]")

    logger.info("Export complete")


if __name__ == "__main__":
    app()
