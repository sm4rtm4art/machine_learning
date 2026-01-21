#!/usr/bin/env python
"""Serve model via FastAPI."""

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
    model_path: Path = typer.Option(
        ...,
        "--model-path",
        "-m",
        help="Path to model file (ONNX or PyTorch)",
    ),
    port: int = typer.Option(
        8000,
        "--port",
        "-p",
        help="Port to serve on",
    ),
    host: str = typer.Option(
        "0.0.0.0",
        "--host",
        "-h",
        help="Host to bind to",
    ),
) -> None:
    """Start model inference server."""
    setup_logging()

    logger.info(f"Loading model from {model_path}")

    # Create FastAPI app
    # from fastapi import FastAPI
    # import uvicorn

    # api = FastAPI(title=f"{PROJECT_NAME} API")

    # @api.post("/predict")
    # async def predict(request: PredictRequest) -> PredictResponse:
    #     result = model.predict(request.input)
    #     return PredictResponse(output=result)

    # @api.get("/health")
    # async def health():
    #     return {"status": "healthy"}

    # logger.info(f"Starting server on {host}:{port}")
    # uvicorn.run(api, host=host, port=port)

    console.print("[red]Not implemented: Add serving logic[/red]")


if __name__ == "__main__":
    app()
