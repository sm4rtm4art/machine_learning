#!/usr/bin/env python
"""Export SSL models to deployment format."""

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

PROJECT_NAME = "vision_ssl_transfer"


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
    opset_version: int = typer.Option(
        17,
        "--opset",
        help="ONNX opset version",
    ),
) -> None:
    """Export model to deployment format."""
    setup_logging()

    paths = get_project_paths(PROJECT_NAME)
    output_path = output_path or paths.artifacts_dir / f"model.{format_}"

    setup_mlflow(PROJECT_NAME)

    logger.info(f"Loading model from run {run_id}")

    import torch
    from projects.vision_ssl_transfer.project.model import SSLBackbone

    # Load backbone
    artifact_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="best_ssl_model.pt"
    )
    backbone = SSLBackbone.from_pretrained(Path(artifact_path))
    backbone.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)

    if format_ == "onnx":
        logger.info(f"Exporting to ONNX: {output_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            backbone.model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["image"],
            output_names=["embedding"],
            dynamic_axes={
                "image": {0: "batch_size"},
                "embedding": {0: "batch_size"},
            },
        )

        console.print(f"[green]Exported to {output_path}[/green]")

        # Verify
        import onnx

        model = onnx.load(str(output_path))
        onnx.checker.check_model(model)
        console.print("[green]ONNX model verified[/green]")

        if quantize:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            quantized_path = output_path.with_suffix(".quantized.onnx")
            quantize_dynamic(
                str(output_path),
                str(quantized_path),
                weight_type=QuantType.QInt8,
            )
            console.print(f"[green]Quantized model saved to {quantized_path}[/green]")

    elif format_ == "torchscript":
        logger.info(f"Exporting to TorchScript: {output_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Trace the model
        traced = torch.jit.trace(backbone.model, dummy_input)
        traced.save(str(output_path))

        console.print(f"[green]Exported to {output_path}[/green]")

    else:
        console.print(f"[red]Unknown format: {format_}[/red]")
        raise typer.Exit(1)

    # Log to MLflow
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(str(output_path), f"exports/{format_}")
        if quantize and format_ == "onnx":
            mlflow.log_artifact(str(quantized_path), f"exports/{format_}")

    logger.info("Export complete")


if __name__ == "__main__":
    app()
