"""Tests for RTSP capture including reconnect and backoff behaviour."""

from __future__ import annotations

import numpy as np

from firewatch.config import SourceConfig
from firewatch.stream import RtspFrameSource, _sleep_interruptible


class FakeCapture:
    """Minimal stand-in for ``cv2.VideoCapture``.

    ``script`` is a list of frame-or-None values returned by successive
    ``read()`` calls; ``open_ok`` controls ``isOpened()``.
    """

    def __init__(self, script=None, open_ok: bool = True) -> None:
        self._script = list(script or [])
        self._open_ok = open_ok
        self.released = False
        self.props: list[tuple[int, float]] = []

    def isOpened(self) -> bool:  # noqa: N802 - mirrors OpenCV API
        return self._open_ok

    def read(self):
        if self._script:
            item = self._script.pop(0)
            if item is None:
                return False, None
            return True, item
        return False, None

    def set(self, prop, value):  # noqa: A003 - mirrors OpenCV API
        self.props.append((prop, value))
        return True

    def release(self):
        self.released = True


class ScriptedOpener:
    """Returns queued captures; records how many times it was called."""

    def __init__(self, captures: list[FakeCapture]) -> None:
        self._captures = list(captures)
        self.calls = 0
        self.urls: list[str] = []

    def __call__(self, url: str, transport: str) -> FakeCapture:
        self.calls += 1
        self.urls.append(url)
        if self._captures:
            return self._captures.pop(0)
        return FakeCapture(open_ok=False)


def _rtsp_config(**overrides) -> SourceConfig:
    data = {
        "source_type": "rtsp",
        "rtsp_url": "rtsp://cam/stream",
        "max_reconnect_attempts": 3,
        "reconnect_initial_backoff": 0.001,
        "reconnect_max_backoff": 0.002,
    }
    data.update(overrides)
    return SourceConfig(**data)


def test_reads_frames_from_open_stream():
    frames = [np.zeros((4, 4, 3), np.uint8), np.ones((4, 4, 3), np.uint8)]
    capture = FakeCapture(script=frames)
    source = RtspFrameSource(_rtsp_config(), open_capture=ScriptedOpener([capture]))

    collected = []
    for frame in source.frames(stop=lambda: len(collected) >= 2):
        collected.append(frame)

    assert len(collected) == 2
    assert source.stats.frames_read == 2


def test_reconnects_after_read_failure():
    first = FakeCapture(script=[np.zeros((2, 2, 3), np.uint8), None])
    second = FakeCapture(script=[np.ones((2, 2, 3), np.uint8), None])

    class StopAfterReconnect(ScriptedOpener):
        """Stops the loop once the second capture has delivered its frame."""

        def __call__(self, url, transport):
            if self.calls >= 2:
                raise AssertionError("reconnected more than once")
            return super().__call__(url, transport)

    opener = StopAfterReconnect([first, second])
    source = RtspFrameSource(_rtsp_config(), open_capture=opener)

    frames = []
    for frame in source.frames(stop=lambda: len(frames) >= 2):
        frames.append(frame)

    assert len(frames) == 2
    assert opener.calls == 2
    assert source.stats.reconnects_succeeded == 1
    assert first.released is True


def test_reconnect_attempts_counter_resets_after_success():
    first = FakeCapture(open_ok=False)
    second = FakeCapture(script=[np.zeros((2, 2, 3), np.uint8), None])
    opener = ScriptedOpener([first, second])
    source = RtspFrameSource(
        _rtsp_config(max_reconnect_attempts=5), open_capture=opener
    )

    frames = []
    for frame in source.frames(stop=lambda: len(frames) >= 1):
        frames.append(frame)

    # One failed open, then a success: attempts was reset after recovery.
    assert len(frames) == 1
    assert source.stats.reconnects_succeeded == 1
    assert source.stats.reconnect_attempts == 0


def test_gives_up_after_max_reconnect_attempts():
    opener = ScriptedOpener([FakeCapture(open_ok=False)])
    source = RtspFrameSource(
        _rtsp_config(max_reconnect_attempts=2), open_capture=opener
    )
    collected = list(source.frames())
    assert collected == []
    assert source.stats.reconnect_attempts == 2


def test_resize_requested_props_are_applied():
    capture = FakeCapture(script=[np.zeros((2, 2, 3), np.uint8)])
    source = RtspFrameSource(
        _rtsp_config(width=1280, height=720, max_reconnect_attempts=1),
        open_capture=ScriptedOpener([capture]),
    )
    for _ in source.frames(stop=lambda: source.stats.frames_read >= 1):
        pass
    import cv2

    assert (cv2.CAP_PROP_FRAME_WIDTH, 1280) in capture.props
    assert (cv2.CAP_PROP_FRAME_HEIGHT, 720) in capture.props


def test_file_source_stops_at_end_of_stream():
    capture = FakeCapture(script=[np.zeros((2, 2, 3), np.uint8), None])
    opener = ScriptedOpener([capture])
    source = RtspFrameSource(
        SourceConfig(source_type="file", file_path="clip.mp4"), open_capture=opener
    )
    collected = list(source.frames())
    assert len(collected) == 1
    assert opener.calls == 1, "file sources must not reconnect"


def test_file_source_missing_file_returns_no_frames():
    opener = ScriptedOpener([FakeCapture(open_ok=False)])
    source = RtspFrameSource(
        SourceConfig(source_type="file", file_path="missing.mp4"), open_capture=opener
    )
    assert list(source.frames()) == []


def test_stop_flag_halts_iteration():
    frames = [np.zeros((2, 2, 3), np.uint8)] * 5
    capture = FakeCapture(script=frames)
    source = RtspFrameSource(_rtsp_config(), open_capture=ScriptedOpener([capture]))

    seen = 0
    for _ in source.frames(stop=lambda: seen >= 2):
        seen += 1
    assert seen == 2


def test_sleep_interruptible_can_be_cancelled():
    assert _sleep_interruptible(5.0, lambda: True) is True
    assert _sleep_interruptible(0.01, lambda: False) is False


def test_release_is_idempotent():
    source = RtspFrameSource(_rtsp_config(), open_capture=ScriptedOpener([]))
    source.release()
    source.release()
