#!/usr/bin/env python
"""Serve TrOCR model via FastAPI."""

from pathlib import Path

import typer
from rich.console import Console

from ml_portfolio.common.logging import get_logger, setup_logging
from ml_portfolio.common.paths import get_project_paths

app = typer.Typer()
console = Console()
logger = get_logger(__name__)

PROJECT_NAME = "llm_ocr_trocr"


@app.command()
def main(
    model_path: Path = typer.Option(
        ...,
        "--model-path",
        "-m",
        help="Path to model directory or ONNX file",
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
        help="Host to bind to",
    ),
    use_onnx: bool = typer.Option(
        False,
        "--onnx",
        help="Use ONNX runtime for inference",
    ),
) -> None:
    """Start OCR inference server."""
    setup_logging()

    logger.info(f"Loading model from {model_path}")

    # Import here to avoid slow startup when just checking help
    import base64
    import io
    from typing import Optional

    import torch
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from PIL import Image
    from pydantic import BaseModel
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    # Create FastAPI app
    api = FastAPI(
        title="TrOCR OCR API",
        description="OCR inference API using TrOCR model",
        version="0.1.0",
    )

    # Load model
    if use_onnx:
        console.print("[yellow]ONNX serving not fully implemented[/yellow]")
        raise typer.Exit(1)

    processor = TrOCRProcessor.from_pretrained(model_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # Request/Response models
    class PredictRequest(BaseModel):
        """OCR prediction request."""
        image_base64: str
        return_confidence: bool = True

    class PredictResponse(BaseModel):
        """OCR prediction response."""
        text: str
        confidence: Optional[float] = None

    class HealthResponse(BaseModel):
        """Health check response."""
        status: str
        model_loaded: bool
        device: str

    @api.post("/predict", response_model=PredictResponse)
    async def predict(request: PredictRequest) -> PredictResponse:
        """Run OCR on an image."""
        try:
            # Decode image
            image_bytes = base64.b64decode(request.image_base64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # Process
            pixel_values = processor(images=image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device)

            # Generate
            with torch.no_grad():
                if request.return_confidence:
                    outputs = model.generate(
                        pixel_values,
                        max_length=128,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )

                    # Decode
                    text = processor.batch_decode(
                        outputs.sequences, skip_special_tokens=True
                    )[0]

                    # Compute confidence
                    if outputs.scores:
                        scores = torch.stack(outputs.scores, dim=1)
                        probs = torch.softmax(scores, dim=-1)
                        token_probs = probs.gather(
                            2, outputs.sequences[:, 1:].unsqueeze(-1)
                        ).squeeze(-1)
                        confidence = float(token_probs.mean())
                    else:
                        confidence = None
                else:
                    generated_ids = model.generate(pixel_values, max_length=128)
                    text = processor.batch_decode(
                        generated_ids, skip_special_tokens=True
                    )[0]
                    confidence = None

            return PredictResponse(text=text, confidence=confidence)

        except Exception as e:
            logger.exception("Prediction failed")
            raise HTTPException(status_code=500, detail=str(e))

    @api.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            device=device,
        )

    @api.get("/")
    async def root() -> dict[str, str]:
        """Root endpoint."""
        return {
            "service": "TrOCR OCR API",
            "version": "0.1.0",
            "docs": f"http://{host}:{port}/docs",
        }

    # Run server
    console.print(f"\n[green]Starting server on http://{host}:{port}[/green]")
    console.print(f"API docs: http://{host}:{port}/docs")
    console.print(f"Health check: http://{host}:{port}/health")

    uvicorn.run(api, host=host, port=port)


if __name__ == "__main__":
    app()
