"""Frame capture from an RTSP camera (or a local file) with auto-reconnect.

RTSP cameras drop connections. Rather than letting the pipeline die, the
source reconnects with exponential backoff. ``_open_capture`` is a module
level function so tests can substitute a fake capture object without a real
camera or OpenCV build.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from .config import SourceConfig

logger = logging.getLogger(__name__)


def _open_capture(url: str, transport: str) -> cv2.VideoCapture:
    """Open an RTSP/file capture, preferring the FFMPEG backend.

    Returns a capture object whether or not it successfully opened; callers
    check ``isOpened()``. Returning rather than raising keeps the failure path
    inside :meth:`RtspFrameSource.frames` so reconnection is uniform.
    """
    if transport == "tcp":
        # Must be set before opening; the FFMPEG backend reads this env var at
        # open time. UDP often drops packets on congested networks.
        import os

        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

    return cv2.VideoCapture(url, cv2.CAP_FFMPEG)


@dataclass
class SourceStats:
    """Observability counters for a source instance."""

    frames_read: int = 0
    read_failures: int = 0
    reconnect_attempts: int = 0
    reconnects_succeeded: int = 0


class RtspFrameSource:
    """Yields BGR frames, reconnecting transparently when the stream drops."""

    def __init__(
        self,
        config: SourceConfig,
        open_capture: Callable[[str, str], cv2.VideoCapture] = _open_capture,
    ) -> None:
        self._config = config
        self._open_capture = open_capture
        self._capture: cv2.VideoCapture | None = None
        self.stats = SourceStats()

    @property
    def url(self) -> str:
        if self._config.source_type == "file":
            return str(self._config.file_path)
        return self._config.rtsp_url

    def open(self) -> bool:
        """Open the underlying capture and apply any requested resizing."""
        capture = self._open_capture(self.url, self._config.transport)
        if not capture.isOpened():
            return False

        if self._config.width:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
        if self._config.height:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)

        self._capture = capture
        return True

    def read(self) -> np.ndarray | None:
        """Read the next frame, or ``None`` if the stream ended/failed."""
        if self._capture is None:
            return None
        ok, frame = self._capture.read()
        if not ok or frame is None:
            self.stats.read_failures += 1
            return None
        self.stats.frames_read += 1
        return frame

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def frames(self, stop: Callable[[], bool] = lambda: False):
        """Iterate frames indefinitely, reconnecting on failure.

        A file source terminates when the video ends (reconnecting to a file
        would just replay it forever, which is never what an operator wants).
        An RTSP source retries with exponential backoff until ``stop()``
        returns True or the configured attempt budget is exhausted.

        ``reconnect_attempts`` is the number of failed opens in the current
        outage (reset once the stream recovers) and drives the attempt budget.
        ``reconnects_succeeded`` counts each recovery, whether the re-open
        worked first try or after several failures.
        """
        backoff = self._config.reconnect_initial_backoff
        reconnecting = False

        while not stop():
            if self._capture is None:
                if not self.open():
                    if self._config.source_type == "file":
                        logger.error("Could not open file source %s", self.url)
                        return
                    self.stats.reconnect_attempts += 1
                    reconnecting = True
                    if self._exhausted(self.stats.reconnect_attempts):
                        logger.error(
                            "Giving up after %s reconnect attempts",
                            self.stats.reconnect_attempts,
                        )
                        return
                    logger.warning(
                        "Connect failed (%s/%s); retrying in %.1fs",
                        self.stats.reconnect_attempts,
                        self._config.max_reconnect_attempts or "inf",
                        backoff,
                    )
                    if _sleep_interruptible(backoff, stop):
                        return
                    backoff = min(backoff * 2, self._config.reconnect_max_backoff)
                    continue
                if reconnecting:
                    self.stats.reconnects_succeeded += 1
                    logger.info(
                        "Reconnected to %s after %s failed attempt(s)",
                        self.url,
                        self.stats.reconnect_attempts,
                    )
                    reconnecting = False
                backoff = self._config.reconnect_initial_backoff
                self.stats.reconnect_attempts = 0

            frame = self.read()
            if frame is None:
                if self._config.source_type == "file":
                    logger.info(
                        "File source exhausted after %s frames", self.stats.frames_read
                    )
                    return
                logger.warning("Stream read failed; reconnecting")
                self.release()
                reconnecting = True
                continue

            yield frame

    def _exhausted(self, attempts: int) -> bool:
        limit = self._config.max_reconnect_attempts
        return limit > 0 and attempts >= limit


def _sleep_interruptible(seconds: float, stop: Callable[[], bool]) -> bool:
    """Sleep in small slices so a stop request is honoured promptly."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if stop():
            return True
        time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))
    return stop()
