"""JSONL persistence for detections, alerts and pipeline events.

One JSON object per line keeps the files append-friendly and trivially
ingestible by log shippers. Writers are opened lazily and flushed per record
so a crash mid-stream does not lose the last detection.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from .config import LoggingConfig

logger = logging.getLogger(__name__)


class JsonlWriter:
    """Append-only JSONL writer that is safe to share across threads."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._handle = self.path.open("a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, default=str)
        with self._lock:
            self._handle.write(line + "\n")
            self._handle.flush()

    def close(self) -> None:
        with self._lock:
            if not self._handle.closed:
                self._handle.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class DetectionLogger:
    """Records every processed frame that contained a detection."""

    def __init__(self, config: LoggingConfig) -> None:
        self._config = config
        log_dir = Path(config.log_dir)
        self._detections = JsonlWriter(log_dir / config.detection_log)
        self._alerts = JsonlWriter(log_dir / config.alert_log)
        self._actuation = JsonlWriter(log_dir / config.actuation_log)

    def log_frame(self, result: Any) -> None:
        """Persist a frame result, skipping frames with no detections."""
        if not result.detections:
            return
        self._detections.write(result.to_dict())

    def log_alert(self, alert: Any) -> None:
        self._alerts.write(alert.to_dict())

    def log_actuation(self, record: Any) -> None:
        """Persist an actuation decision or actuator event.

        Every command that can open a valve is recorded here, whether or not it
        succeeded. This file is the audit trail for "why did it spray?".
        """
        self._actuation.write(record.to_dict() if hasattr(record, "to_dict") else record)

    def log_event(self, event: str, **fields: Any) -> None:
        record = {
            "event": event,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **fields,
        }
        self._detections.write(record)

    def close(self) -> None:
        self._detections.close()
        self._alerts.close()
        self._actuation.close()

    def __enter__(self) -> "DetectionLogger":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
