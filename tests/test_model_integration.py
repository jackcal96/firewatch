"""Integration tests that exercise the real Ultralytics backend.

These are marked ``model`` and skipped when weights are absent, so the default
test run stays fast and dependency-light:

    pytest -m "not model"     # skip everything here
    python scripts/download_model.py && pytest -m model
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from firewatch.config import AppConfig, ModelConfig
from firewatch.detector import FireSmokeDetector

pytestmark = pytest.mark.model

MODEL_PATH = Path("models/best.pt")


@pytest.fixture(scope="module")
def detector() -> FireSmokeDetector:
    if not MODEL_PATH.is_file():
        pytest.skip(
            f"{MODEL_PATH} not found; run: python scripts/download_model.py"
        )
    config = AppConfig.load("configs/default.yaml").model
    return FireSmokeDetector.from_config(config)


def test_backend_loads_and_reports_fire_and_smoke_classes(detector):
    # The default D-Fire fine-tune labels smoke as 0 and fire as 1.
    names = detector._backend._model.names  # noqa: SLF001 - integration introspection
    labels = {str(v).lower() for v in names.values()}
    assert {"fire", "smoke"} <= labels


def test_blank_frame_produces_no_detections(detector):
    """A featureless frame must not hallucinate a fire."""
    detections = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
    assert detections == []


def test_real_inference_returns_well_formed_boxes(detector):
    """A frame with fire-like colours yields plausibly-shaped detections.

    This asserts on structure (bounds, ranges, geometry) rather than on the
    model firing, because a synthetic blob is not a real fire and a miss is
    legitimate model behaviour.
    """
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[200:400, 200:440] = (0, 80, 255)  # high-saturation orange region

    for detection in detector.detect(frame):
        x1, y1, x2, y2 = detection.bbox
        assert 0.0 <= x1 < x2 <= 640
        assert 0.0 <= y1 < y2 <= 480
        assert 0.0 <= detection.confidence <= 1.0
        assert detection.label.value in {"fire", "smoke", "other"}


def test_confidence_threshold_is_respected_by_real_model():
    if not MODEL_PATH.is_file():
        pytest.skip("weights missing")

    frame = np.zeros((320, 320, 3), dtype=np.uint8)
    frame[100:220, 100:220] = (0, 60, 220)

    permissive = FireSmokeDetector.from_config(
        ModelConfig(model_path=str(MODEL_PATH), confidence_threshold=0.05)
    )
    strict = FireSmokeDetector.from_config(
        ModelConfig(model_path=str(MODEL_PATH), confidence_threshold=0.95)
    )

    permissive_count = len(permissive.detect(frame))
    strict_count = len(strict.detect(frame))

    assert strict_count <= permissive_count
    assert all(d.confidence >= 0.95 for d in strict.detect(frame))
