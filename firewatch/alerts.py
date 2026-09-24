"""Alert decision logic and notification delivery.

The policy is deliberately stateful: a single noisy frame should not page
anyone. ``consecutive_frames`` requires sustained evidence, and
``cooldown_seconds`` stops a smouldering fire from emitting an alert every
frame. All time comparisons are done against a caller-supplied clock so the
logic is deterministic under test.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Protocol

from .config import AlertConfig
from .types import FrameResult, DetectionClass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Alert:
    """A raised alert, ready for logging and delivery."""

    label: DetectionClass
    confidence: float
    frame_index: int
    timestamp: datetime
    consecutive_frames: int
    detections: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": "alert",
            "label": self.label.value,
            "confidence": round(self.confidence, 4),
            "frame_index": self.frame_index,
            "timestamp": self.timestamp.astimezone(timezone.utc).isoformat(),
            "consecutive_frames": self.consecutive_frames,
            "detections": self.detections,
        }

    def format_message(self) -> str:
        return (
            f"[{self.label.value.upper()}] detected with confidence "
            f"{self.confidence:.2f} at frame {self.frame_index} "
            f"({self.timestamp.astimezone(timezone.utc).isoformat()})"
        )


class Notifier(Protocol):
    """Delivery channel for a raised alert."""

    def send(self, alert: Alert) -> bool: ...


class LoggingNotifier:
    """Always-available fallback that writes alerts to the logger."""

    def send(self, alert: Alert) -> bool:
        logger.warning("ALERT %s", alert.format_message())
        return True


class WebhookNotifier:
    """POSTs the alert as JSON to a configured URL."""

    def __init__(self, url: str, timeout: float = 5.0) -> None:
        self._url = url
        self._timeout = timeout

    def send(self, alert: Alert) -> bool:
        payload = json.dumps(alert.to_dict()).encode("utf-8")
        request = urllib.request.Request(
            self._url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                if 200 <= response.status < 300:
                    return True
                logger.error("Webhook returned HTTP %s", response.status)
                return False
        except (urllib.error.URLError, OSError) as exc:
            logger.error("Webhook delivery failed: %s", exc)
            return False


class AlertPolicy:
    """Decides whether a frame's detections warrant an alert.

    State tracked per class:
        * ``_streak``      running count of consecutive frames with a hit
        * ``_last_alert``  monotonic timestamp of the last alert for the class
    """

    def __init__(
        self,
        config: AlertConfig,
        clock: Callable[[], float] = time.monotonic,
        notifiers: list[Notifier] | None = None,
    ) -> None:
        self._config = config
        self._clock = clock
        self._notifiers: list[Notifier] = (
            notifiers if notifiers is not None else [LoggingNotifier()]
        )
        self._streak: dict[DetectionClass, int] = {}
        self._last_alert: dict[DetectionClass, float] = {}

    def evaluate(self, result: FrameResult) -> list[Alert]:
        """Update streaks and return any alerts raised for this frame."""
        if not self._config.enable_alerts:
            return []

        watched = self._watched_classes()
        alerts: list[Alert] = []

        for label in DetectionClass:
            if label not in watched:
                # Still track the streak so a class becoming watched mid-run
                # does not inherit a stale count.
                self._streak[label] = 0
                continue

            detections = [d for d in result.detections if d.label is label]
            qualified = [
                d
                for d in detections
                if d.confidence >= self._config.min_confidence
                and d.area >= self._config.min_bbox_area_px
            ]

            if not qualified:
                self._streak[label] = 0
                continue

            self._streak[label] = self._streak.get(label, 0) + 1
            if self._streak[label] < self._config.consecutive_frames:
                continue

            now = self._clock()
            last = self._last_alert.get(label)
            if last is not None and (now - last) < self._config.cooldown_seconds:
                continue

            self._last_alert[label] = now
            alert = Alert(
                label=label,
                confidence=max(d.confidence for d in qualified),
                frame_index=result.frame_index,
                timestamp=result.timestamp,
                consecutive_frames=self._streak[label],
                detections=[d.to_dict() for d in qualified],
            )
            alerts.append(alert)
            self._deliver(alert)

        return alerts

    def _watched_classes(self) -> set[DetectionClass]:
        watched: set[DetectionClass] = set()
        if self._config.alert_on_fire:
            watched.add(DetectionClass.FIRE)
        if self._config.alert_on_smoke:
            watched.add(DetectionClass.SMOKE)
        return watched

    def _deliver(self, alert: Alert) -> None:
        for notifier in self._notifiers:
            try:
                notifier.send(alert)
            except Exception:  # noqa: BLE001 - one bad notifier must not stop the pipeline
                logger.exception("Notifier %r raised while sending alert", notifier)


def build_notifiers(config: AlertConfig) -> list[Notifier]:
    """Assemble notifiers from config, keeping the logger as a guaranteed sink."""
    notifiers: list[Notifier] = [LoggingNotifier()]
    if config.webhook_url:
        notifiers.append(WebhookNotifier(config.webhook_url, config.webhook_timeout_seconds))
    return notifiers
