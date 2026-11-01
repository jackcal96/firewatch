# AGENTS.md

Repository guidance for AI agents working on **firewatch**.

## What this is

Fire/smoke detection from RTSP camera feeds, with optional turret aiming and
gated water suppression. YOLOv8 (Ultralytics) object detection + OpenCV capture
+ JSONL logging + webhook alerts. Pure Python, plus an ESP32 Arduino sketch for
the turret. The agent is strictly software: it never specifies or validates the
physical plumbing.

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
.venv/bin/python -m firewatch turret --config configs/watergun.yaml
```

`pyserial` is **not** installed in the dev venv, so `SerialTurret` is imported
lazily and only the dry-run turret is exercised here. `tests/test_turret.py`
installs a fake into `sys.modules["serial"]` to test the wire protocol.

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
* `AimGate` (supervisor.py) is the protocol that lets the supervisor gate a
  spray on aim without importing any turret code. `AimController` implements
  it. Keep that direction of dependency: `turret.py` and `aiming.py` must not
  import `supervisor.py` beyond the shared `AimOutcome`/`DenyReason` types.

## Testing conventions

* Tests must not require a camera, network, or model weights. Anything needing
  weights goes in `tests/test_model_integration.py` with `@pytest.mark.model`.
* `tests/conftest.py` defines `FakeBackend` (replays scripted results, then
  repeats the last one), `make_raw`, `RecordingNotifier`, `ExplodingNotifier`.

## Actuation safety invariants (do not weaken)

`supervisor.py` is the only module that can command hardware. These rules are
load-bearing; changing them changes the safety argument:

* Actuation defaults to **off**, and `dry_run` defaults to **true**. A config
  must never reach a live actuator without an explicit human decision.
* `ActuationConfig._check_safety_invariants` refuses unsafe combinations at
  construction: `min_confidence` < 0.5, `consecutive_frames` < 2, smoke-only
  `trigger_classes`, `require_zone` with no zones, and `dry_run: false` with no
  `close_command`. Add gates here rather than removing them.
* `LOCKED_OUT` is terminal until `reset()` is called explicitly. Never
  auto-reset: that is how one false positive spends the whole shot budget.
* `max_shots` defaults to 1. Spraying is not idempotent and cannot be undone.
* The watchdog must close the valve when the detector stops heartbeating. A
  dead detector holding a valve open is the worst failure in the system.
* `tick()` exists so timers fire without new frames. If you change the spray
  timer path, keep `tick()` working — a stalled stream must still close a valve.
* Decisions carry a `reason` from `DenyReason`. Every refusal should stay
  explainable from the audit log alone.

## Aiming safety invariants (do not weaken)

A turret breaks an assumption that made a fixed nozzle safe: with a fixed
nozzle, fire being in the right place meant the water was too. Once the mount
can move, that is no longer true, so the achieved bearing must be *verified*
rather than assumed.

* **Aiming and spraying are separate.** `pipeline._track()` aims at the
  strongest fire every frame; the supervisor independently re-checks the
  achieved bearing as its final gate. Tracking must never imply permission to
  spray. Do not move the aim check earlier, and do not let aiming command the
  actuator.
* The aim gate runs **last** in `_advance`, immediately before
  `_start_spraying`. Everything above it proves the fire is real; the gate
  proves the nozzle is on it.
* `AimController` treats the turret's **reported** position as the only
  evidence. A mount that did not arrive, reports nothing, or raises must fail
  closed. Never trust "the command was sent" as proof of aim.
* `solve_bearing` returns `None` outside the envelope rather than clamping.
  Clamping would report success while pointing somewhere unasked.
* The envelope is enforced in Python **and** in the firmware. Do not remove
  either; a serial glitch must not be able to drive the mount past its limits.
* `TurretConfig` refuses a live turret with no `serial_port`, and a home datum
  outside the envelope. Add gates here rather than removing them.
* `AimController` re-checks `calibration.is_degenerate` even though
  `build_aim_controller` already does. A zero scale maps every target to one
  bearing, so a "successful" aim would be meaningless.

## Live preview gotcha

`cv2.imshow` on a machine with no display **aborts the process** — it does not
raise a catchable `cv2.error`. So `LiveView` checks `display_available()`
*before* calling it; catching the exception is not sufficient and the
pre-flight check is what actually prevents the crash. Any new display code
needs the same pre-flight. Note this dev container has both
`opencv-python` and `opencv-python-headless` installed, with the GUI build
winning, so a missing `DISPLAY` is the real blocker here.

## Runtime facts verified locally

* Real RTSP was tested end-to-end with a local `mediamtx` server plus an ffmpeg
  publisher; reconnect/backoff and webhook POST delivery were both confirmed.
  `imageio-ffmpeg` provides a bundled ffmpeg binary when the system has none.
* Measured inference: ~100–1000 ms/frame on CPU at `imgsz=640`.
* The model scores smoke ~0.6–0.8 and fire ~0.3–0.5 on real photos of fire, so
  thresholds around 0.25–0.3 for detection are reasonable; alert defaults stay
  higher to limit false pages.
