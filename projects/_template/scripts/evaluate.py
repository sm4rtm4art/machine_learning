#!/usr/bin/env python
"""Evaluate model and generate reports."""

from pathlib import Path

import mlflow
import typer
from rich.console import Console

from ml_portfolio.common.logging import get_logger, setup_logging
from ml_portfolio.common.paths import get_project_paths
from ml_portfolio.tracking.mlflow_utils import setup_mlflow

# from projects._template.project.eval import evaluate, save_results

app = typer.Typer()
console = Console()
logger = get_logger(__name__)

PROJECT_NAME = "_template"


@app.command()
def main(
    run_id: str = typer.Option(
        None,
        "--run-id",
        "-r",
        help="MLflow run ID to evaluate",
    ),
    model_path: Path = typer.Option(
        None,
        "--model-path",
        "-m",
        help="Path to model checkpoint (alternative to run-id)",
    ),
    output_dir: Path = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for reports",
    ),
) -> None:
    """Evaluate model and generate reports."""
    setup_logging()

    if run_id is None and model_path is None:
        console.print("[red]Error: Must provide either --run-id or --model-path[/red]")
        raise typer.Exit(1)

    paths = get_project_paths(PROJECT_NAME)
    output_dir = output_dir or paths.reports_dir

    # Load model
    if run_id:
        setup_mlflow(PROJECT_NAME)
        logger.info(f"Loading model from run {run_id}")
        # model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")
    else:
        logger.info(f"Loading model from {model_path}")
        # model = TemplateModel.from_pretrained(model_path)

    # Load test data
    # test_dataset = TemplateDataset(paths.data_dir, split="test")
    # test_loader = DataLoader(test_dataset, batch_size=32)

    # Evaluate
    # result = evaluate(model, test_loader, config)

    # Save results
    # save_results(result, output_dir, run_id or "local")

    # Log to MLflow if we have a run
    if run_id:
        with mlflow.start_run(run_id=run_id):
            # mlflow.log_metrics(result.metrics)
            # mlflow.log_artifacts(str(output_dir / run_id), "evaluation")
            pass

    console.print("[red]Not implemented: Add evaluation logic[/red]")

    logger.info("Evaluation complete")


if __name__ == "__main__":
    app()
