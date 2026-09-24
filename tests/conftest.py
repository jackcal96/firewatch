"""Shared test fixtures.

The pipeline is deliberately built around small protocols (``DetectionBackend``,
``Notifier``, capture opening) so tests can drive real pipeline logic with
deterministic fakes instead of mocks of the code under test.
"""

from __future__ import annotations

import numpy as np
import pytest

from firewatch.config import (
    AlertConfig,
    AppConfig,
    LoggingConfig,
    ModelConfig,
    RuntimeConfig,
    SourceConfig,
)
from firewatch.detector import RawDetections


class FakeBackend:
    """Detection backend that replays a scripted list of results.

    When the script is exhausted the backend keeps returning the last scripted
    result, because a real detector facing similar frames keeps producing
    similar output. Tests that need a clean frame should script an empty
    result explicitly.
    """

    def __init__(self, results: list[RawDetections] | None = None) -> None:
        self._results = list(results or [])
        self.calls = 0
        self._empty = RawDetections(
            xyxy=np.empty((0, 4), dtype=np.float32),
            conf=np.empty((0,), dtype=np.float32),
            cls=np.empty((0,), dtype=np.float32),
            names={0: "smoke", 1: "fire"},
        )
        self._default = self._results[-1] if self._results else self._empty

    def predict(self, frame: np.ndarray) -> RawDetections:
        self.calls += 1
        if self._results:
            return self._results.pop(0)
        return self._default


def make_raw(
    boxes: list[tuple[float, float, float, float]],
    confidences: list[float],
    class_ids: list[int],
    names: dict[int, str] | None = None,
) -> RawDetections:
    """Build a :class:`RawDetections` from plain Python lists."""
    return RawDetections(
        xyxy=np.array(boxes, dtype=np.float32).reshape(-1, 4),
        conf=np.array(confidences, dtype=np.float32),
        cls=np.array(class_ids, dtype=np.float32),
        names=names if names is not None else {0: "smoke", 1: "fire"},
    )


class RecordingNotifier:
    """Notifier that remembers everything sent to it."""

    def __init__(self, succeed: bool = True) -> None:
        self.sent = []
        self._succeed = succeed

    def send(self, alert) -> bool:
        self.sent.append(alert)
        return self._succeed


class ExplodingNotifier:
    """Notifier that always raises, to prove delivery is fault-isolated."""

    def send(self, alert) -> bool:
        raise RuntimeError("delivery backend is down")


@pytest.fixture
def model_config(tmp_path) -> ModelConfig:
    return ModelConfig(model_path=str(tmp_path / "fake.pt"), confidence_threshold=0.4)


@pytest.fixture
def alert_config() -> AlertConfig:
    return AlertConfig(
        enable_alerts=True,
        alert_on_fire=True,
        alert_on_smoke=False,
        min_confidence=0.5,
        consecutive_frames=1,
        cooldown_seconds=0.0,
    )


@pytest.fixture
def app_config(tmp_path) -> AppConfig:
    """Fully isolated app config writing logs under tmp_path."""
    return AppConfig(
        source=SourceConfig(source_type="file", file_path="unused.mp4"),
        model=ModelConfig(model_path=str(tmp_path / "fake.pt")),
        alert=AlertConfig(
            min_confidence=0.5, consecutive_frames=1, cooldown_seconds=0.0
        ),
        logging=LoggingConfig(log_dir=str(tmp_path / "logs")),
        runtime=RuntimeConfig(),
    )


@pytest.fixture
def frame() -> np.ndarray:
    return np.zeros((64, 64, 3), dtype=np.uint8)
