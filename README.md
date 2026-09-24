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
  logging_utils.py   # JSONL detection/alert/event writers
  visualize.py       # annotated frame rendering
  pipeline.py        # orchestration; run_forever() entry point
  cli.py             # argparse CLI (run / analyze / check)
tests/               # pytest suite; fakes live in conftest.py
configs/default.yaml # default configuration
scripts/download_model.py
```

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
  default for that reason.
* This is a monitoring aid, not a certified fire alarm. It does not replace
  hard-wired smoke detectors or a monitored alarm system.

## Licence

MIT for this code. The bundled model weights are AGPL-3.0, inherited from
Ultralytics YOLOv8 — check that licence before commercial use.
