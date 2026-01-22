#!/usr/bin/env python
"""Export TrOCR model to ONNX format."""

from pathlib import Path

import mlflow
import typer
from rich.console import Console

from ml_portfolio.common.logging import get_logger, setup_logging
from ml_portfolio.common.paths import get_project_paths
from ml_portfolio.tracking.mlflow_utils import setup_mlflow

from projects.ocr_pipeline.project.model import TrOCRWrapper, ModelConfig

app = typer.Typer()
console = Console()
logger = get_logger(__name__)

PROJECT_NAME = "ocr_pipeline"


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
        help="Output path for ONNX model",
    ),
    quantize: bool = typer.Option(
        False,
        "--quantize",
        "-q",
        help="Apply INT8 quantization",
    ),
    opset_version: int = typer.Option(
        14,
        "--opset",
        help="ONNX opset version",
    ),
) -> None:
    """Export TrOCR model to ONNX format."""
    setup_logging()

    paths = get_project_paths(PROJECT_NAME)
    output_path = output_path or paths.artifacts_dir / "model.onnx"

    # Load model from MLflow
    setup_mlflow(PROJECT_NAME)
    logger.info(f"Loading model from run {run_id}")

    client = mlflow.tracking.MlflowClient()
    artifact_path = client.download_artifacts(run_id, "model")
    model_path = Path(artifact_path) / "best_model"

    # Create wrapper
    config = ModelConfig(name=str(model_path))
    wrapper = TrOCRWrapper.from_pretrained(model_path)

    # Export to ONNX
    logger.info(f"Exporting to ONNX: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        onnx_path = wrapper.export_onnx(output_path, opset_version=opset_version)

        # Quantize if requested
        if quantize:
            quantized_path = output_path.with_suffix(".int8.onnx")
            logger.info(f"Quantizing to INT8: {quantized_path}")

            try:
                from onnxruntime.quantization import quantize_dynamic, QuantType

                quantize_dynamic(
                    str(onnx_path),
                    str(quantized_path),
                    weight_type=QuantType.QInt8,
                )

                # Compare file sizes
                original_size = onnx_path.stat().st_size / (1024 * 1024)
                quantized_size = quantized_path.stat().st_size / (1024 * 1024)
                compression = (1 - quantized_size / original_size) * 100

                console.print(f"\n[green]Quantization complete![/green]")
                console.print(f"Original size: {original_size:.1f} MB")
                console.print(f"Quantized size: {quantized_size:.1f} MB")
                console.print(f"Compression: {compression:.1f}%")

            except ImportError:
                console.print("[yellow]onnxruntime-quantization not available. Skipping quantization.[/yellow]")
            except Exception as e:
                console.print(f"[red]Quantization failed: {e}[/red]")

        # Log to MLflow
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(str(onnx_path), "exports/onnx")
            if quantize and quantized_path.exists():
                mlflow.log_artifact(str(quantized_path), "exports/onnx")

        console.print(f"\n[green]Export complete![/green]")
        console.print(f"ONNX model: {onnx_path}")

    except Exception as e:
        console.print(f"[red]Export failed: {e}[/red]")
        logger.exception("ONNX export failed")
        raise typer.Exit(1)


@app.command()
def benchmark(
    onnx_path: Path = typer.Argument(..., help="Path to ONNX model"),
    num_runs: int = typer.Option(100, "--runs", "-n", help="Number of inference runs"),
) -> None:
    """Benchmark ONNX model inference speed."""
    import time

    import numpy as np
    import onnxruntime as ort

    setup_logging()

    console.print(f"Loading ONNX model: {onnx_path}")

    session = ort.InferenceSession(str(onnx_path))

    # Get input shape
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    console.print(f"Input shape: {input_shape}")

    # Create dummy input (batch_size, channels, height, width)
    dummy_input = np.random.randn(1, 3, 384, 384).astype(np.float32)

    # Warmup
    console.print("Warming up...")
    for _ in range(10):
        session.run(None, {input_name: dummy_input})

    # Benchmark
    console.print(f"Running {num_runs} inference passes...")
    latencies = []

    for _ in range(num_runs):
        start = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms

    # Report
    latencies_arr = np.array(latencies)
    console.print(f"\n[bold]Latency Statistics (ms)[/bold]")
    console.print(f"  Mean:   {np.mean(latencies_arr):.2f}")
    console.print(f"  Std:    {np.std(latencies_arr):.2f}")
    console.print(f"  P50:    {np.percentile(latencies_arr, 50):.2f}")
    console.print(f"  P95:    {np.percentile(latencies_arr, 95):.2f}")
    console.print(f"  P99:    {np.percentile(latencies_arr, 99):.2f}")
    console.print(f"  Min:    {np.min(latencies_arr):.2f}")
    console.print(f"  Max:    {np.max(latencies_arr):.2f}")


if __name__ == "__main__":
    app()
