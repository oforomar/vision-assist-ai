"""
Vision Assist captioning service.

A single FastAPI app that the Spring backend calls over HTTP (frame-relay): it captions a
video frame with Qwen2-VL and classifies it as SAFE or DANGEROUS, returning the caption plus
token/latency metadata.
"""

import base64
import binascii
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from io import BytesIO

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from prometheus_client import Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from app.captioning.model import Captioner
from app.classification.classifier import DangerClassifier
from app.config import settings
from app.schemas import (
    CaptionMetadata,
    CaptionRequest,
    CaptionResponse,
    Classification,
    HealthResponse,
)

# Module-level singletons. The model is loaded once at startup (lifespan); the classifier is cheap.
captioner = Captioner()
classifier = DangerClassifier()

# Latency histograms scraped via /metrics. Buckets span a warm GPU (<0.5s) to a cold/CPU box (~10s).
_LATENCY_BUCKETS = (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.5, 4.0, 6.0, 10.0)
INFERENCE_SECONDS = Histogram(
    "ai_inference_seconds", "GPU inference time (model.generate only).", buckets=_LATENCY_BUCKETS
)
REQUEST_SECONDS = Histogram(
    "ai_request_seconds",
    "Total /caption handler time (decode + preprocess + inference + classify).",
    buckets=_LATENCY_BUCKETS,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    captioner.load()
    yield


app = FastAPI(
    title="Vision Assist Captioning Service",
    description="Captions a video frame with Qwen2-VL and classifies it as SAFE or DANGEROUS.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Default HTTP metrics + GET /metrics for Prometheus (internal-only in deployment).
Instrumentator().instrument(app).expose(app, include_in_schema=False)


def _decode_image(image_base64: str) -> Image.Image:
    """Decodes a (possibly data-URL prefixed) base64 string into an RGB image."""
    raw = image_base64.split(",", 1)[1] if "," in image_base64 else image_base64
    try:
        image_bytes = base64.b64decode(raw)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {exc}") from exc
    try:
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Could not decode image.") from exc


@app.post("/caption", response_model=CaptionResponse)
def caption(request: CaptionRequest) -> CaptionResponse:
    if not captioner.is_loaded:
        raise HTTPException(status_code=503, detail="Model is still loading. Try again shortly.")

    # Total server-side handling time; latency_ms below covers only model.generate(), so the
    # difference is the decode/preprocess/classify overhead. The backend times its own call around
    # this request, letting the client attribute network vs server time per caption.
    request_start = time.perf_counter()
    image = _decode_image(request.image_base64)
    result = captioner.generate(image, prompt=request.prompt)
    label, reason = classifier.classify(result["caption"])

    request_seconds = time.perf_counter() - request_start
    INFERENCE_SECONDS.observe(result["latency_ms"] / 1000)
    REQUEST_SECONDS.observe(request_seconds)

    return CaptionResponse(
        caption=result["caption"],
        classification=Classification(label=label, reason=reason),
        metadata=CaptionMetadata(
            model=captioner.model_name,
            device=captioner.device,
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
            total_tokens=result["total_tokens"],
            latency_ms=result["latency_ms"],
            request_ms=round(request_seconds * 1000, 2),
            image_width=image.width,
            image_height=image.height,
            generated_at=datetime.now(timezone.utc).isoformat(),
        ),
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if captioner.is_loaded else "loading",
        model=captioner.model_name,
        device=captioner.device,
        model_loaded=captioner.is_loaded,
    )
