"""Pipeline orchestration: capture -> detect -> alert -> log.

``FireSmokePipeline`` is transport-agnostic; it consumes any iterable of
frames, which is what makes the whole system testable without a camera. The
``run_forever`` helper wires that iterable to a real source for production use.
"""

from __future__ import annotations

import logging
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Callable, Iterable

import numpy as np
import cv2

from .alerts import AlertPolicy, build_notifiers
from .config import AppConfig
from .detector import FireSmokeDetector
from .logging_utils import DetectionLogger
from .stream import RtspFrameSource
from .types import FrameResult
from .visualize import draw_detections

logger = logging.getLogger(__name__)


class FireSmokePipeline:
    """Turns frames into detections, alerts and log records."""

    def __init__(
        self,
        config: AppConfig,
        detector: FireSmokeDetector,
        alert_policy: AlertPolicy | None = None,
        detection_logger: DetectionLogger | None = None,
    ) -> None:
        self._config = config
        self._detector = detector
        self._logger = detection_logger or DetectionLogger(config.logging)
        self._alert_policy = alert_policy or AlertPolicy(
            config.alert, notifiers=build_notifiers(config.alert)
        )
        self._owns_logger = detection_logger is None
        self.frames_processed = 0
        self.alerts_raised = 0

    def process_frame(self, frame: np.ndarray, frame_index: int) -> FrameResult:
        """Run detection on one frame, alert on hits, and log the outcome."""
        started = time.perf_counter()
        detections = self._detector.detect(frame)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        result = FrameResult(
            frame_index=frame_index,
            timestamp=datetime.now(timezone.utc),
            detections=detections,
            inference_ms=elapsed_ms,
        )

        self._logger.log_frame(result)
        for alert in self._alert_policy.evaluate(result):
            self.alerts_raised += 1
            self._logger.log_alert(alert)

        self.frames_processed += 1

        if detections and self._config.logging.write_frames_with_detections:
            self._save_annotated(frame, frame_index, detections)

        return result

    def run(self, frames: Iterable[np.ndarray], stop: Callable[[], bool] = lambda: False) -> int:
        """Process an iterable of frames until exhausted or stopped."""
        min_interval = (
            1.0 / self._config.runtime.max_fps if self._config.runtime.max_fps else 0.0
        )
        every_n = self._config.runtime.process_every_n_frames
        max_frames = self._config.runtime.max_frames
        processed = 0

        try:
            for index, frame in enumerate(frames):
                if stop():
                    logger.info("Stop requested; exiting pipeline")
                    break
                if index % every_n != 0:
                    continue

                tick = time.monotonic()
                self.process_frame(frame, index)
                processed += 1

                if max_frames and processed >= max_frames:
                    logger.info("Reached max_frames=%s; exiting", max_frames)
                    break

                if min_interval:
                    remaining = min_interval - (time.monotonic() - tick)
                    if remaining > 0:
                        time.sleep(remaining)
        finally:
            self.close()

        return processed

    def _save_annotated(
        self, frame: np.ndarray, frame_index: int, detections: list
    ) -> None:
        annotated = draw_detections(frame, detections)
        out_dir = Path(self._config.logging.annotated_frame_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        cv2.imwrite(str(out_dir / f"frame_{frame_index:07d}_{stamp}.jpg"), annotated)

    def close(self) -> None:
        if self._owns_logger:
            self._logger.close()


def run_forever(config: AppConfig) -> int:
    """Production entry point: capture from the configured source until Ctrl-C."""
    source = RtspFrameSource(config.source)
    detector = FireSmokeDetector.from_config(config.model)
    pipeline = FireSmokePipeline(config, detector)

    stop_event = Event()

    def _handle_signal(signum: int, _frame: object) -> None:
        logger.info("Received signal %s; shutting down", signum)
        stop_event.set()

    # SIGINT/SIGTERM only exist on POSIX; guard for portability.
    for name in ("SIGINT", "SIGTERM"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), _handle_signal)

    logger.info(
        "Starting fire/smoke detection on %s (model=%s, device=%s, conf=%.2f)",
        source.url,
        config.model.model_path,
        config.model.device,
        config.model.confidence_threshold,
    )

    try:
        processed = pipeline.run(source.frames(stop=stop_event.is_set), stop=stop_event.is_set)
    finally:
        source.release()

    logger.info(
        "Stopped. frames_processed=%s alerts=%s read_failures=%s reconnects=%s",
        processed,
        pipeline.alerts_raised,
        source.stats.read_failures,
        source.stats.reconnects_succeeded,
    )
    return processed
