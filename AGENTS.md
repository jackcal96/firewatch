# AGENTS.md

Repository guidance for AI agents working on **firewatch**.

## What this is

Fire/smoke detection from RTSP camera feeds. YOLOv8 (Ultralytics) object
detection + OpenCV capture + JSONL logging + webhook alerts. Pure Python.

## Environment

* Python 3.13 venv at `.venv/`. Always use `.venv/bin/python` and
  `.venv/bin/pip`; the system interpreter lacks the dependencies.
* CPU-only torch is installed (`torch==2.14.0+cpu`). There is no GPU in the
  dev container, so keep `model.device` at `cpu` in tests.
* `models/best.pt` is git-ignored. Fetch it with
  `python scripts/download_model.py` (`rabahdev/fire-smoke-yolov8n`,
  classes `{0: smoke, 1: fire}`). Model integration tests skip if it is absent.

## Commands

```bash
.venv/bin/python -m pytest -m "not model"   # fast suite; no camera/network/weights
.venv/bin/python -m pytest                  # includes real-model integration tests
.venv/bin/python -m firewatch check --config configs/default.yaml
.venv/bin/python -m firewatch analyze --input some.jpg
```

## Architecture notes

* `DetectionBackend` (detector.py), `Notifier` (alerts.py) and the
  `open_capture` callable (stream.py) are deliberate seams. Tests inject
  deterministic fakes there — do not replace them with mocks of the code under
  test, and keep those signatures stable.
* Config precedence is CLI `--set` > env (`FIREWATCH_*`) > YAML > defaults.
  Config models use `extra="forbid"`; unknown keys must keep failing loudly.
* `SourceConfig.file_path` is validated with a `model_validator`, not a
  `field_validator` — Pydantic skips field validators for absent keys.
* Stream `reconnect_attempts` = failed opens in the current outage (resets on
  recovery); `reconnects_succeeded` = recoveries. A `reconnecting` flag tracks
  an ongoing outage; without it a reconnect that succeeds first try is uncounted.
* Confidences pass through `float32`, so tests comparing exact values must use
  `pytest.approx`.

## Testing conventions

* Tests must not require a camera, network, or model weights. Anything needing
  weights goes in `tests/test_model_integration.py` with `@pytest.mark.model`.
* `tests/conftest.py` defines `FakeBackend` (replays scripted results, then
  repeats the last one), `make_raw`, `RecordingNotifier`, `ExplodingNotifier`.

## Runtime facts verified locally

* Real RTSP was tested end-to-end with a local `mediamtx` server plus an ffmpeg
  publisher; reconnect/backoff and webhook POST delivery were both confirmed.
  `imageio-ffmpeg` provides a bundled ffmpeg binary when the system has none.
* Measured inference: ~100–1000 ms/frame on CPU at `imgsz=640`.
* The model scores smoke ~0.6–0.8 and fire ~0.3–0.5 on real photos of fire, so
  thresholds around 0.25–0.3 for detection are reasonable; alert defaults stay
  higher to limit false pages.
