# firewatch

Fire and smoke detection from an RTSP camera feed, using a YOLOv8 object
detection model. It reports per-detection confidence scores, raises alerts when
fire is detected, and writes structured logs for every frame that contained a
detection.

## How it works

```
RTSP camera ──► frame source ──► YOLOv8 detector ──► alert policy ──► notifiers
  (reconnect)      (OpenCV)      (conf filtering)   (streak+cooldown)   (log/webhook)
                                          │
                                          └──► JSONL detection log
```

* **Model** – `rabahdev/fire-smoke-yolov8n`, a YOLOv8n fine-tune on the D-Fire
  dataset with two classes: `smoke` (0) and `fire` (1). Any Ultralytics-compatible
  `.pt` or `.onnx` file can be swapped in.
* **Source** – OpenCV over the FFMPEG backend, using RTSP-over-TCP, with automatic
  reconnection and exponential backoff.
* **Alerts** – require sustained evidence (`consecutive_frames`) and are
  rate-limited per class (`cooldown_seconds`) so a flickering detection does not
  page anyone. Delivered to the log and, optionally, an HTTP webhook.
* **Logs** – newline-delimited JSON, one record per line, ready for `jq`, Vector,
  Fluent Bit or any log shipper.

## Requirements

* Python 3.10+
* An RTSP camera (or a video/image file for testing)
* No GPU required. CPU inference runs a few frames per second on a modern core;
  see [Performance](#performance) for tuning.

## Setup

```bash
git clone <this-repo> && cd project

python3 -m venv .venv
source .venv/bin/activate

# Install CPU-only PyTorch first (smaller and faster than the default CUDA build).
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Then the rest.
pip install -r requirements.txt

# Download the fire/smoke weights into models/best.pt
python scripts/download_model.py
```

For a CUDA GPU, install the matching CUDA build of torch instead of the CPU
index above, then set `model.device: cuda:0` in your config.

Verify the install without touching a camera:

```bash
python -m firewatch check --config configs/default.yaml
```

## Running

### Against a camera

Set your stream URL, then run. Credentials are best supplied via the
environment so they stay out of the config file:

```bash
export FIREWATCH_RTSP_URL="rtsp://user:password@192.168.1.10:554/stream1"
python -m firewatch run --config configs/default.yaml
```

Stop with `Ctrl-C`; the process releases the camera and flushes its logs.

### Against a file (no camera needed)

```bash
# A single image
python -m firewatch analyze --input /path/to/frame.jpg

# A video
python -m firewatch analyze --input /path/to/clip.mp4
```

`analyze` runs the identical detection, alerting and logging path, which makes it
the quickest way to validate a model or threshold change.

## Configuration

Settings live in `configs/default.yaml` and are resolved in this order,
highest priority first:

1. CLI overrides — `--set section__field=value`
2. Environment variables — `FIREWATCH_RTSP_URL`, `FIREWATCH_MODEL_PATH`,
   `FIREWATCH_LOG_DIR`, `FIREWATCH_ALERT_LOG`, `FIREWATCH_WEBHOOK_URL`
3. The YAML file passed with `--config`
4. Built-in defaults

```bash
python -m firewatch run --config configs/default.yaml \
  --set model__device=cuda:0 \
  --set model__confidence_threshold=0.45 \
  --set alert__cooldown_seconds=120
```

Unknown keys and malformed values are rejected at startup rather than ignored, so
a typo fails loudly instead of silently leaving a default in place.

### Key settings

| Setting | Default | Purpose |
| --- | --- | --- |
| `source.rtsp_url` | placeholder | The camera stream. |
| `source.transport` | `tcp` | TCP is more reliable than UDP on lossy networks. |
| `source.max_reconnect_attempts` | `0` | `0` retries forever. |
| `model.confidence_threshold` | `0.35` | Detections below this are dropped. |
| `model.device` | `cpu` | Set to `cuda:0` or `mps` if available. |
| `model.unknown_labels_policy` | `ignore` | How to treat a class the code does not know. |
| `alert.alert_on_fire` | `true` | Raise alerts for fire. |
| `alert.alert_on_smoke` | `false` | Smoke is often false-positive-prone; off by default. |
| `alert.min_confidence` | `0.5` | Confidence required to alert (separate from detection). |
| `alert.consecutive_frames` | `2` | Sustained frames needed before alerting. |
| `alert.cooldown_seconds` | `60` | Minimum gap between repeated alerts per class. |
| `alert.min_bbox_area_px` | `0` | Drop tiny boxes, e.g. `400` = 20x20 px. |
| `alert.webhook_url` | unset | Optional HTTP POST target. |
| `runtime.process_every_n_frames` | `1` | Raise to cut CPU load. |
| `runtime.max_fps` | unset | Cap the processing rate. |

### Tuning guidance

Confidence scores from this model are informative but not absolute: smoke
detections typically land around 0.6–0.8 while fire flames often score 0.3–0.5
on real imagery. Before deploying, run `analyze` over clips from your own camera
and pick thresholds from the observed distribution. If fire recall is too low,
use `model.confidence_threshold` around `0.25` to surface candidate boxes, and
keep `alert.min_confidence` higher so only confident hits page anyone.

## Alerts

Every alert is written to `logs/alerts.jsonl` and logged at `WARNING` level. To
also POST it as JSON to a webhook, set `alert.webhook_url`:

```bash
export FIREWATCH_WEBHOOK_URL="https://hooks.example.com/firewatch"
python -m firewatch run --config configs/default.yaml
```

Payload shape:

```json
{
  "event": "alert",
  "label": "fire",
  "confidence": 0.3073,
  "frame_index": 42,
  "timestamp": "2026-01-01T00:00:00+00:00",
  "consecutive_frames": 3,
  "detections": [
    {"label": "fire", "raw_label": "fire", "confidence": 0.3073,
     "bbox": [651.47, 234.96, 671.89, 293.06]}
  ]
}
```

A failing webhook is logged and swallowed; it never stops detection.

## Logs

`logs/detections.jsonl` holds one record per frame that contained a detection:

```json
{"frame_index": 0, "timestamp": "2026-01-01T00:00:00+00:00", "inference_ms": 87.4,
 "max_fire_confidence": 0.62, "max_smoke_confidence": 0.71,
 "detections": [{"label": "smoke", "raw_label": "smoke", "confidence": 0.7119,
                 "bbox": [21.83, 7.74, 940.63, 496.05]}]}
```

Frames without detections are not logged, which keeps the file proportional to
events rather than uptime. Set `logging.write_frames_with_detections: true` to
also save annotated JPEGs for review.

To additionally record *every* frame, use `pipeline.process_frame` directly from
your own loop, or run the detection log through `jq`:

```bash
jq -c 'select(.max_fire_confidence > 0.5)' logs/detections.jsonl
```

## Actuation (driving a suppression device)

> **Read this before enabling anything.** Actuation is the part of this project
> that can cause physical harm. The default configuration cannot actuate: it
> detects, alerts and logs, and nothing else.

Detection is not a decision. The model answers "is there fire in this frame?",
which is a different question from "should I open a valve?". `firewatch`
separates the two: the model produces observations, and
`firewatch/supervisor.py` decides whether those observations justify acting.

### The safety model

Actuation is off by default and requires several deliberate steps to enable.
When enabled, the valve opens only if **all** of these hold:

| Gate | Default | Purpose |
| --- | --- | --- |
| `enable_actuation` | `false` | Master switch. Off means nothing can move. |
| `dry_run` | `true` | Simulates commands; no hardware is touched. |
| `trigger_classes` | `[fire]` | Smoke alone can never open a valve. |
| `min_confidence` | `0.6` | Weak evidence does not actuate. Floor is 0.5. |
| `consecutive_frames` | `5` | Sustained evidence, not a single frame. Minimum 2. |
| `min_bbox_area_px` | `400` | Ignores small boxes, which are usually noise. |
| `arm_delay_seconds` | `2.0` | Evidence must hold this long in wall-clock time. |
| `require_zone` | `true` | Fire must be inside a configured protected zone. |
| `max_spray_seconds` | `10.0` | Hard ceiling on any single spray. |
| `max_shots` | `1` | Budget, after which the device locks out. |
| `watchdog_timeout_seconds` | `3.0` | A dead detector forces the valve closed. |

Every one of these is validated at startup. An unsafe combination — a
confidence below 0.5, a single-frame trigger, smoke-only triggering, a live
mode with no close command — is refused rather than silently accepted.

The state machine is `IDLE → CONFIRMING → SPRAYING → COOLDOWN → LOCKED_OUT`.
`LOCKED_OUT` is terminal until a human calls `reset()`: an automatic reset
would let the device spend its whole shot budget on one persistent false
positive. A closed valve is the resting state, so power loss or a crash fails
closed.

### Trying it safely

Run the whole decision path against a video, with no hardware anywhere:

```bash
firewatch analyze --input your_clip.mp4 \
  --set actuation__enable_actuation=true \
  --set actuation__require_zone=false \
  --set actuation__consecutive_frames=2 \
  --set actuation__arm_delay_seconds=0.0
```

This is a genuine rehearsal: the supervisor runs its real logic and writes to
`logs/actuation.jsonl`, but `DryRunActuator` performs no I/O. You will see
lines like:

```
DRY RUN: valve OPEN commanded (no hardware attached)
DRY RUN: valve CLOSE commanded (no hardware attached)
```

and an audit record for each decision explaining *why*:

```json
{"event": "actuation_decision", "action": "open", "allowed": true,
 "reason": "fire_confirmed", "state": "spraying", "confidence": 0.5092,
 "shots_used": 1, "detail": "spraying; shot 1/1 (DRY RUN - no water)"}
```

Confirm what a config would do before running it:

```bash
firewatch check --config your_config.yaml
```

### Zones

A zone is the area a nozzle is allowed to wet. Detections outside every enabled
zone are still logged and alerted on, but never actuate. Coordinates are
normalised (0.0-1.0), so a zone definition survives a resolution change.

See `configs/zones.example.yaml` for a worked example. To derive coordinates,
save an annotated frame and read the pixel positions off it:

```bash
firewatch analyze --input frame.jpg \
  --set logging__write_frames_with_detections=true
# annotated image lands in logs/annotated/
```

### Connecting real hardware

`open_command` and `close_command` are argv lists, run without a shell, so this
project does not care how your device is driven — GPIO, a relay board, MQTT, a
serial port. Provide a small script and point the config at it:

```yaml
actuation:
  enable_actuation: true
  dry_run: false
  require_zone: true
  zones: [...]                      # must not be empty
  open_command: ["python", "scripts/valve.py", "open"]
  close_command: ["python", "scripts/valve.py", "close"]
```

Turning off `dry_run` is the last step, not the first. Run in dry-run for weeks
and read `logs/actuation.jsonl` first: if there is a single entry there that you
would not have wanted to become real water, the answer is not to tune the
thresholds, it is to keep it in dry-run.

### What this does not do

* It does not choose, size, aim or plumb the hardware.
* It does not detect fire *spread*, only where fire was seen in a frame.
* It has no independent sensor. One model's opinion is the only evidence, which
  is why the constraints are conservative and `max_shots` defaults to 1.
* It is not a substitute for a certified suppression system. A fire alarm and
  an extinguisher remain the primary controls.

## Tests

```bash
pytest -m "not model"   # fast unit tests: no camera, network or weights needed
pytest                  # also runs the real-model integration tests
pytest --cov=firewatch  # with coverage
```

The unit tests drive the real detection, alerting and logging code through
injected fakes at the seams (`DetectionBackend`, `Notifier` and the capture
opener), so no test needs a camera, the network, or model weights. The four
`model` tests load the actual weights and are skipped automatically when
`models/best.pt` is missing.

## Project layout

```
firewatch/
  config.py          # pydantic config models and precedence resolution
  types.py           # Detection, FrameResult, DetectionClass
  detector.py        # Ultralytics backend + threshold/label filtering
  stream.py          # RTSP/file capture with reconnect and backoff
  alerts.py          # alert policy (streak, cooldown) and notifiers
  geometry.py        # protected zones: point-in-polygon and bbox coverage
  actuator.py        # hardware boundary: dry-run and subprocess actuators
  supervisor.py      # actuation state machine and safety interlocks
  logging_utils.py   # JSONL detection/alert/actuation writers
  visualize.py       # annotated frame rendering
  pipeline.py        # orchestration; run_forever() entry point
  cli.py             # argparse CLI (run / analyze / check)
tests/               # pytest suite; fakes live in conftest.py
configs/default.yaml       # default configuration
configs/zones.example.yaml # worked example of protected zones
scripts/download_model.py
```

The detection path and the actuation path are deliberately separate modules.
`detector.py` never knows a valve exists, and `supervisor.py` never knows how a
frame was produced. That boundary is what lets the safety logic be tested with
no camera and no hardware, and it keeps the code that can move water small
enough to review in full.

## Performance

CPU inference at `imgsz: 640` costs roughly 100–300 ms per frame depending on the
machine, which is ample for fire monitoring but not for 30 fps video. If you need
to reduce load:

* `runtime.process_every_n_frames: 3` – analyse every third frame.
* `runtime.max_fps: 5` – cap the processing rate explicitly.
* `model.imgsz: 416` – smaller input, faster inference, smaller fire recall.
* Use a CUDA GPU: install the CUDA torch build and set `model.device: cuda:0`.

## Limitations

* Detections are bounding boxes, not segmentation, so a box indicates where fire
  was seen, not how far it has spread.
* The model was trained on the D-Fire dataset; unusual scenes, heavy steam or
  bright orange industrial lighting can produce false positives. Validate on your
  own footage before relying on it.
* Smoke detection is weaker than fire detection. It is disabled for alerting by
  default for that reason, and it can never trigger actuation.
* This is a monitoring aid, not a certified fire alarm. It does not replace
  hard-wired smoke detectors or a monitored alarm system.
* Actuation relies on a single model as its only evidence. There is no
  independent thermal or gas sensor, so the system cannot cross-check what it
  thinks it sees. Keep `max_shots` at 1 and stay in dry-run until the audit log
  shows a long, clean record.

## Licence

MIT for this code. The bundled model weights are AGPL-3.0, inherited from
Ultralytics YOLOv8 — check that licence before commercial use.
