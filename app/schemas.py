"""Pydantic request/response models for the captioning service."""

from typing import Optional

from pydantic import BaseModel, Field


class CaptionRequest(BaseModel):
    image_base64: str = Field(
        ...,
        description=(
            "Base64-encoded image. A data-URL prefix "
            "(e.g. 'data:image/jpeg;base64,') is accepted and stripped."
        ),
    )
    prompt: Optional[str] = Field(
        default=None,
        description=(
            "Optional text prompt for conditional captioning. "
            "Omit for an unconditional caption."
        ),
    )


class Classification(BaseModel):
    label: str = Field(..., description="SAFE or DANGEROUS.")
    reason: str = Field(..., description="Human-readable explanation of the classification.")


class CaptionMetadata(BaseModel):
    model: str
    device: str
    input_tokens: int = Field(
        ...,
        description="Prompt tokens fed to the model, including the expanded image (vision) tokens.",
    )
    output_tokens: int = Field(..., description="Tokens generated for the caption.")
    total_tokens: int
    latency_ms: float = Field(..., description="Wall-clock time spent in model.generate().")
    request_ms: float = Field(
        ...,
        description=(
            "Total server-side handling time for this request "
            "(decode + preprocess + inference + classify). Always >= latency_ms."
        ),
    )
    image_width: int
    image_height: int
    generated_at: str = Field(..., description="UTC ISO-8601 timestamp.")


class CaptionResponse(BaseModel):
    caption: str
    classification: Classification
    metadata: CaptionMetadata


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    model_loaded: bool
