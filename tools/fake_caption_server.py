"""
Fake captioning service for testing the backend frame-relay without loading the real model.

Mirrors the real /caption contract (caption + classification + metadata) but returns canned captions
after a small fixed delay, cycling one DANGEROUS caption so the UI danger badge can be exercised.

Run:  uvicorn tools.fake_caption_server:app --port 9000
"""

import asyncio
import itertools
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Fake VisionAssist Caption Service")

# Simulated processing latency (ms). Bump this to see how the relay's back-pressure behaves.
LATENCY_MS = 300

_CAPTIONS = itertools.cycle(
    [
        ("a man walking down a city street at dusk", "SAFE", "No danger keywords detected."),
        ("a knife resting on a wooden kitchen table", "DANGEROUS", "Detected danger: knife"),
        ("an open laptop and a cup of coffee on a desk", "SAFE", "No danger keywords detected."),
        ("a flight of stairs leading down to a dim basement", "DANGEROUS", "Detected danger: stairs"),
    ]
)


class CaptionRequest(BaseModel):
    image_base64: str
    prompt: Optional[str] = None


@app.post("/caption")
async def caption(_: CaptionRequest):
    await asyncio.sleep(LATENCY_MS / 1000)
    text, label, reason = next(_CAPTIONS)
    words = len(text.split())
    return {
        "caption": text,
        "classification": {"label": label, "reason": reason},
        "metadata": {
            "model": "fake-stub",
            "device": "cpu",
            "input_tokens": 1,
            "output_tokens": words,
            "total_tokens": 1 + words,
            "latency_ms": float(LATENCY_MS),
            # Real service reports total handler time (decode + inference + classify); fake a
            # couple of ms of overhead so latency consumers see the request_ms > latency_ms shape.
            "request_ms": float(LATENCY_MS) + 2.5,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": "fake-stub", "device": "cpu", "model_loaded": True}
