"""Configuration loading.

Values are resolved in priority order (highest first):

1. Explicit CLI arguments
2. Environment variables (optionally via a ``.env`` file)
3. A YAML config file passed with ``--config``
4. Field defaults defined here

This keeps a single source of truth for every tunable while still letting an
operator override the RTSP URL from the environment without editing files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

EnvPrefix = "FIREWATCH_"


class StrictModel(BaseModel):
    """Base model that rejects unknown keys.

    Without ``extra='forbid'`` a typo such as ``confidence_thresold`` is
    silently ignored, leaving the operator to wonder why the setting had no
    effect. Failing loudly is the safer default for a safety system.
    """

    model_config = ConfigDict(extra="forbid")


# Environment variables are prefixed to avoid colliding with unrelated vars.
_ENV_OVERRIDES: dict[str, str] = {
    "rtsp_url": "RTSP_URL",
    "model_path": "MODEL_PATH",
    "log_dir": "LOG_DIR",
    "alert_log": "ALERT_LOG",
    "webhook_url": "WEBHOOK_URL",
}


class SourceConfig(StrictModel):
    """Where frames come from and how resilient capture should be."""

    rtsp_url: str = Field(
        default="rtsp://user:password@192.168.1.10:554/stream1",
        description="RTSP URL. Ignored when 'file' source is used.",
    )
    source_type: Literal["rtsp", "file"] = "rtsp"
    file_path: str | None = Field(
        default=None, description="Video/image path used when source_type='file'."
    )
    transport: Literal["tcp", "udp"] = "tcp"
    read_timeout_seconds: float = Field(default=5.0, gt=0)
    reconnect_initial_backoff: float = Field(default=1.0, gt=0)
    reconnect_max_backoff: float = Field(default=30.0, gt=0)
    max_reconnect_attempts: int = Field(default=0, ge=0, description="0 = retry forever")
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _require_file_path(self) -> "SourceConfig":
        # A field_validator on file_path would not run when the key is absent
        # from the config, which is exactly the case we need to catch.
        if self.source_type == "file" and not self.file_path:
            raise ValueError("file_path is required when source_type='file'")
        return self


class ModelConfig(StrictModel):
    """Object-detection model and inference parameters."""

    model_path: str = Field(
        default="models/best.pt",
        description="Ultralytics-compatible weights (.pt or .onnx).",
    )
    device: str = Field(default="cpu", description="'cpu', 'cuda:0', 'mps', ...")
    imgsz: int = Field(default=640, gt=0)
    confidence_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    max_detections: int = Field(default=50, gt=0)
    half_precision: bool = Field(default=False, description="FP16; CUDA only.")
    unknown_labels_policy: Literal["ignore", "as_other"] = "ignore"


class AlertConfig(StrictModel):
    """When to alert, and how to rate-limit repeated alerts."""

    enable_alerts: bool = True
    alert_on_fire: bool = True
    alert_on_smoke: bool = False
    min_confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Confidence floor to raise an alert."
    )
    consecutive_frames: int = Field(
        default=2, ge=1, description="Frames a class must appear in before alerting."
    )
    cooldown_seconds: float = Field(
        default=60.0, ge=0, description="Suppress repeat alerts for this long."
    )
    min_bbox_area_px: float = Field(
        default=0.0, ge=0, description="Ignore detections smaller than this area."
    )
    webhook_url: str | None = Field(
        default=None, description="Optional HTTP POST target for alerts."
    )
    webhook_timeout_seconds: float = Field(default=5.0, gt=0)


class ZoneConfig(StrictModel):
    """A protected area that a detection must fall inside to count.

    ``polygon`` is a list of ``[x, y]`` points in normalised coordinates
    (0.0-1.0), so the same zone survives a resolution change on the camera.
    """

    name: str = Field(min_length=1)
    polygon: list[tuple[float, float]] = Field(
        min_length=3, description="At least three [x, y] points, normalised 0-1."
    )
    enabled: bool = True

    @model_validator(mode="after")
    def _points_in_range(self) -> "ZoneConfig":
        for x, y in self.polygon:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(
                    f"zone '{self.name}' has a point outside 0-1: ({x}, {y}). "
                    "Coordinates are normalised, not pixels."
                )
        return self


class ActuationConfig(StrictModel):
    """When, and whether, to physically actuate a suppression device.

    This is the highest-risk configuration in the project, so every default is
    the conservative one: actuation is off, and even when switched on it runs
    in dry-run until an operator explicitly disables that.

    The valve opens only when *all* of these hold:
        * ``enable_actuation`` is true and the device is not locked out
        * a qualifying detection has persisted for ``consecutive_frames``
        * confidence is at or above ``min_confidence``
        * the box is large enough (``min_bbox_area_px``) and, when
          ``require_zone`` is set, sits inside a configured zone
        * the evidence has held for ``arm_delay_seconds`` with the heartbeat
          fresh, so a stale detector cannot command a spray
        * the shot budget (``max_shots``) is not exhausted
    """

    enable_actuation: bool = Field(
        default=False, description="Master switch. Off means never actuate."
    )
    dry_run: bool = Field(
        default=True,
        description="Simulate everything, touch no hardware. Disable only after "
        "weeks of clean dry-run logs.",
    )
    trigger_classes: list[Literal["fire", "smoke"]] = Field(
        default=["fire"],
        description="Classes allowed to actuate. Smoke alone should never open a "
        "valve; it is too weak a signal.",
    )
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    consecutive_frames: int = Field(default=5, ge=1)
    min_bbox_area_px: float = Field(default=400.0, ge=0.0)
    arm_delay_seconds: float = Field(
        default=2.0, ge=0.0, description="Evidence must hold this long before spraying."
    )
    min_spray_seconds: float = Field(
        default=1.0, ge=0.0, description="Avoids rapid valve cycling."
    )
    max_spray_seconds: float = Field(
        default=10.0, gt=0, description="Hard ceiling on a single spray."
    )
    cooldown_seconds: float = Field(default=300.0, ge=0.0)
    max_shots: int = Field(
        default=1, ge=1, description="Shots before the device locks out for a human."
    )
    watchdog_timeout_seconds: float = Field(
        default=3.0, gt=0, description="Max heartbeat age tolerated before a spray."
    )
    stop_on_evidence_loss: bool = Field(
        default=True, description="Close the valve once fire is no longer visible."
    )
    require_zone: bool = Field(
        default=True, description="Require detections to fall inside a protected zone."
    )
    zones: list[ZoneConfig] = Field(default_factory=list)
    open_command: list[str] | None = Field(
        default=None, description="argv of the command that opens the valve."
    )
    close_command: list[str] | None = Field(
        default=None, description="argv of the command that closes the valve."
    )
    command_timeout_seconds: float = Field(default=5.0, gt=0)

    @model_validator(mode="after")
    def _check_safety_invariants(self) -> "ActuationConfig":
        if not self.enable_actuation:
            return self

        if self.require_zone and not any(z.enabled for z in self.zones):
            raise ValueError(
                "require_zone is set but no enabled zones are configured. Add a "
                "zone, or set require_zone: false to accept the whole frame."
            )
        if self.min_confidence < 0.5:
            raise ValueError(
                f"min_confidence={self.min_confidence} is too low to actuate on. "
                "Require at least 0.5."
            )
        if self.consecutive_frames < 2:
            raise ValueError(
                "consecutive_frames must be at least 2 for actuation; a single "
                "frame is not evidence."
            )
        if self.max_spray_seconds < self.min_spray_seconds:
            raise ValueError("max_spray_seconds must be >= min_spray_seconds")
        if "smoke" in self.trigger_classes and "fire" not in self.trigger_classes:
            raise ValueError(
                "trigger_classes must include 'fire'. Actuating on smoke alone is "
                "not a supported configuration."
            )
        if not self.dry_run:
            if not self.open_command or not self.close_command:
                raise ValueError(
                    "dry_run is false but open_command/close_command are unset. "
                    "Refusing to start a live actuator with no way to close it."
                )
        return self


class LoggingConfig(StrictModel):
    """Where detections, alerts and actuation events are persisted."""

    log_dir: str = "logs"
    detection_log: str = "detections.jsonl"
    alert_log: str = "alerts.jsonl"
    actuation_log: str = "actuation.jsonl"
    write_frames_with_detections: bool = False
    annotated_frame_dir: str = "logs/annotated"


class RuntimeConfig(StrictModel):
    """Frame pacing and display options."""

    process_every_n_frames: int = Field(default=1, ge=1)
    max_fps: float | None = Field(
        default=None, gt=0, description="Cap processing rate; None = unlimited."
    )
    max_frames: int = Field(default=0, ge=0, description="0 = run until stopped.")


class AppConfig(StrictModel):
    """Top-level configuration object."""

    source: SourceConfig = Field(default_factory=SourceConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    alert: AlertConfig = Field(default_factory=AlertConfig)
    actuation: ActuationConfig = Field(default_factory=ActuationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @classmethod
    def load(cls, path: str | Path | None = None, **overrides: Any) -> "AppConfig":
        """Build config from defaults, a YAML file, env vars, and overrides."""
        data: dict[str, Any] = {}
        if path is not None:
            config_path = Path(path)
            if not config_path.is_file():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            loaded = yaml.safe_load(config_path.read_text()) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"Config file must contain a mapping: {config_path}")
            data = loaded

        _apply_env_overrides(data)
        _apply_overrides(data, overrides)
        return cls.model_validate(data)


def _apply_env_overrides(data: dict[str, Any]) -> None:
    for section, env_name in _ENV_OVERRIDES.items():
        value = os.environ.get(f"{EnvPrefix}{env_name}")
        if value is None:
            continue
        if section in {"rtsp_url", "source_type", "file_path"}:
            target = data.setdefault("source", {})
            target[section] = value
        elif section == "model_path":
            data.setdefault("model", {})["model_path"] = value
        elif section == "log_dir":
            data.setdefault("logging", {})["log_dir"] = value
        elif section == "alert_log":
            data.setdefault("logging", {})["alert_log"] = value
        elif section == "webhook_url":
            data.setdefault("alert", {})["webhook_url"] = value


def _apply_overrides(data: dict[str, Any], overrides: dict[str, Any]) -> None:
    """Merge ``section__field=value`` style overrides into nested config dicts."""
    for key, value in overrides.items():
        if value is None or "__" not in key:
            continue
        section, _, field_name = key.partition("__")
        data.setdefault(section, {})[field_name] = value
