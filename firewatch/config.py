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


class LoggingConfig(StrictModel):
    """Where detections and alerts are persisted."""

    log_dir: str = "logs"
    detection_log: str = "detections.jsonl"
    alert_log: str = "alerts.jsonl"
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
