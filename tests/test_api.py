"""
API tests for /caption and /health.

These never load the real (multi-GB) model: a stub captioner is swapped in, and the TestClient
is used *without* the `with` context manager so the FastAPI lifespan (model load) does not run.
"""

import base64
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import main


def _png_b64() -> str:
    buf = BytesIO()
    Image.new("RGB", (8, 8), "red").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class _StubCaptioner:
    """Stands in for the real Captioner so tests run without loading the model."""

    model_name = "stub-model"
    device = "cpu"
    is_loaded = True

    def load(self):  # pragma: no cover - never invoked in tests
        pass

    def generate(self, image, prompt=None):
        # A caption that the DangerClassifier flags as DANGEROUS, to verify wiring.
        return {
            "caption": "a knife on a table",
            "input_tokens": 1,
            "output_tokens": 6,
            "total_tokens": 7,
            "latency_ms": 12.34,
        }


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "captioner", _StubCaptioner())
    return TestClient(main.app)


def test_caption_returns_caption_classification_and_metadata(client):
    resp = client.post("/caption", json={"image_base64": _png_b64()})
    assert resp.status_code == 200
    body = resp.json()

    assert body["caption"] == "a knife on a table"
    assert body["classification"]["label"] == "DANGEROUS"
    assert "knife" in body["classification"]["reason"].lower()

    meta = body["metadata"]
    assert meta["model"] == "stub-model"
    assert meta["device"] == "cpu"
    assert meta["input_tokens"] == 1
    assert meta["output_tokens"] == 6
    assert meta["total_tokens"] == 7
    assert meta["latency_ms"] == 12.34
    # Total handler time: covers decode + generate + classify, so never below inference time.
    # The stub's generate() takes ~0ms, so this is small but present.
    assert meta["request_ms"] >= 0
    assert meta["image_width"] == 8
    assert meta["image_height"] == 8
    assert meta["generated_at"]


def test_caption_accepts_data_url_prefix(client):
    resp = client.post(
        "/caption", json={"image_base64": "data:image/png;base64," + _png_b64()}
    )
    assert resp.status_code == 200


def test_caption_rejects_invalid_image(client):
    resp = client.post("/caption", json={"image_base64": "not-valid-base64-image"})
    assert resp.status_code == 400


def test_caption_503_when_model_not_loaded(monkeypatch):
    stub = _StubCaptioner()
    stub.is_loaded = False
    monkeypatch.setattr(main, "captioner", stub)
    resp = TestClient(main.app).post("/caption", json={"image_base64": _png_b64()})
    assert resp.status_code == 503


def test_metrics_exposes_latency_histograms(client):
    # One caption call populates the custom histograms...
    assert client.post("/caption", json={"image_base64": _png_b64()}).status_code == 200
    # ...which the Prometheus endpoint then reports.
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "ai_inference_seconds" in resp.text
    assert "ai_request_seconds_count" in resp.text


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_loaded"] is True
    assert body["model"] == "stub-model"
    assert body["status"] == "ok"
