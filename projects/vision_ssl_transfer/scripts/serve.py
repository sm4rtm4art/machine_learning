#!/usr/bin/env python
"""Serve SSL model via FastAPI for inference."""

from pathlib import Path

import typer
from rich.console import Console

from ml_portfolio.common.logging import get_logger, setup_logging

app = typer.Typer()
console = Console()
logger = get_logger(__name__)

PROJECT_NAME = "vision_ssl_transfer"


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

    # Determine model type and load
    if model_path.suffix == ".onnx":
        import onnxruntime as ort

        session = ort.InferenceSession(str(model_path))
        model_type = "onnx"
    else:
        from projects.vision_ssl_transfer.project.model import SSLBackbone

        model = SSLBackbone.from_pretrained(model_path)
        model.eval()
        model_type = "pytorch"

    # Create FastAPI app
    import io
    from contextlib import asynccontextmanager
    from typing import Any

    import numpy as np
    from fastapi import FastAPI, File, UploadFile
    from fastapi.responses import JSONResponse
    from PIL import Image

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Any:
        # Startup
        logger.info(f"Model loaded: {model_type}")
        yield
        # Shutdown
        logger.info("Shutting down")

    api = FastAPI(
        title=f"{PROJECT_NAME} API",
        description="SSL Feature Extraction API",
        version="0.1.0",
        lifespan=lifespan,
    )

    def preprocess_image(image: Image.Image) -> np.ndarray[Any, Any]:
        """Preprocess image for inference."""
        from torchvision import transforms

        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        tensor = transform(image)
        return tensor.unsqueeze(0).numpy()

    @api.get("/health")
    async def health() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy", "model_type": model_type}

    @api.post("/embed")
    async def embed(file: UploadFile = File(...)) -> JSONResponse:
        """Extract embedding from uploaded image."""
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Preprocess
        input_array = preprocess_image(image)

        # Inference
        if model_type == "onnx":
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: input_array})
            embedding = outputs[0][0]
        else:
            import torch

            with torch.no_grad():
                tensor = torch.from_numpy(input_array)
                embedding = model(tensor).numpy()[0]

        return JSONResponse(
            content={
                "embedding": embedding.tolist(),
                "embedding_dim": len(embedding),
            }
        )

    @api.post("/similarity")
    async def similarity(
        file1: UploadFile = File(...),
        file2: UploadFile = File(...),
    ) -> JSONResponse:
        """Compute cosine similarity between two images."""
        # Read images
        contents1 = await file1.read()
        contents2 = await file2.read()

        image1 = Image.open(io.BytesIO(contents1)).convert("RGB")
        image2 = Image.open(io.BytesIO(contents2)).convert("RGB")

        # Preprocess
        input1 = preprocess_image(image1)
        input2 = preprocess_image(image2)

        # Get embeddings
        if model_type == "onnx":
            input_name = session.get_inputs()[0].name
            emb1 = session.run(None, {input_name: input1})[0][0]
            emb2 = session.run(None, {input_name: input2})[0][0]
        else:
            import torch

            with torch.no_grad():
                emb1 = model(torch.from_numpy(input1)).numpy()[0]
                emb2 = model(torch.from_numpy(input2)).numpy()[0]

        # Compute cosine similarity
        similarity = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

        return JSONResponse(content={"similarity": similarity})

    # Start server
    import uvicorn

    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(api, host=host, port=port)


if __name__ == "__main__":
    app()
