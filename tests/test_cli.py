"""Tests for the CLI: argument parsing, overrides and end-to-end analyze."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from firewatch.cli import _split_overrides, main


def test_split_overrides_parses_pairs():
    assert _split_overrides(["model__device=cuda:0", "alert__min_confidence=0.6"]) == {
        "model__device": "cuda:0",
        "alert__min_confidence": "0.6",
    }


def test_split_overrides_rejects_malformed():
    with pytest.raises(ValueError, match="section__field=value"):
        _split_overrides(["nonsense"])


def test_check_with_missing_model_returns_2(tmp_path):
    # Model weights do not exist, so `check` should exit 2 with guidance.
    code = main(["check", "--config", str(_write_cfg(tmp_path, model_path="missing.pt"))])
    assert code == 2


def _write_cfg(tmp_path: Path, model_path: str = "missing.pt", **extra) -> Path:
    cfg = {
        "model": {"model_path": model_path},
        "logging": {"log_dir": str(tmp_path / "logs")},
        "alert": {"consecutive_frames": 1, "cooldown_seconds": 0.0},
    }
    cfg.update(extra)
    path = tmp_path / "cfg.yaml"
    import yaml

    path.write_text(yaml.safe_dump(cfg))
    return path


def test_analyze_missing_input_returns_2(tmp_path):
    code = main(
        ["analyze", "--input", str(tmp_path / "nope.jpg"), "--config", str(_write_cfg(tmp_path))]
    )
    assert code == 2


def test_analyze_nonexistent_config_returns_2(tmp_path):
    code = main(["analyze", "--input", "x.jpg", "--config", str(tmp_path / "no.yaml")])
    assert code == 2


def test_analyze_image_with_stub_model(tmp_path, monkeypatch):
    """Full CLI path over a real image file using a stubbed model.

    We replace ``UltralyticsBackend`` so the test needs no weights, but the
    real detector, pipeline, alert policy and JSONL logger all run.
    """
    image_path = tmp_path / "frame.jpg"
    cv2.imwrite(str(image_path), np.zeros((64, 64, 3), dtype=np.uint8))

    from firewatch import detector as detector_module
    from firewatch.detector import RawDetections

    class StubBackend:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def predict(self, frame):
            return RawDetections(
                xyxy=np.array([[5, 5, 50, 50]], dtype=np.float32),
                conf=np.array([0.88], dtype=np.float32),
                cls=np.array([1], dtype=np.float32),
                names={0: "smoke", 1: "fire"},
            )

    monkeypatch.setattr(detector_module, "UltralyticsBackend", StubBackend)

    log_dir = tmp_path / "logs"
    cfg = _write_cfg(tmp_path, model_path="unused.pt")
    code = main(
        [
            "analyze",
            "--input",
            str(image_path),
            "--config",
            str(cfg),
            "--set",
            "logging__write_frames_with_detections=true",
        ]
    )

    assert code == 0
    detect_log = log_dir / "detections.jsonl"
    assert detect_log.is_file()
    record = json.loads(detect_log.read_text().strip())
    assert record["detections"][0]["label"] == "fire"

    alert_log = log_dir / "alerts.jsonl"
    assert alert_log.is_file()
    assert json.loads(alert_log.read_text().strip())["label"] == "fire"


def test_cli_check_succeeds_with_stub_backend(tmp_path, monkeypatch):
    from firewatch import detector as detector_module

    class StubBackend:
        def __init__(self, *args, **kwargs) -> None:
            pass

    monkeypatch.setattr(detector_module, "UltralyticsBackend", StubBackend)
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"not a real model")

    code = main(["check", "--config", str(_write_cfg(tmp_path, model_path=str(weights)))])
    assert code == 0
