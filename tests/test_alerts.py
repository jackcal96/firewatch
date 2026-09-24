"""Tests for alert policy: hysteresis, thresholds, cooldown, delivery."""

from __future__ import annotations

from datetime import datetime, timezone

from firewatch.alerts import Alert, AlertPolicy, LoggingNotifier, build_notifiers
from firewatch.config import AlertConfig
from firewatch.types import Detection, DetectionClass, FrameResult
from firewatch.visualize import draw_detections

from .conftest import ExplodingNotifier, RecordingNotifier


class FakeClock:
    """Controllable monotonic clock for deterministic cooldown tests."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _frame_result(class_ids: list[int], confidences: list[float], index: int = 0):
    detections = [
        Detection(
            label=DetectionClass.from_label("fire" if c == 1 else "smoke"),
            raw_label="fire" if c == 1 else "smoke",
            confidence=conf,
            bbox=(0.0, 0.0, 20.0, 20.0),
        )
        for c, conf in zip(class_ids, confidences)
    ]
    return FrameResult(
        frame_index=index,
        timestamp=datetime.now(timezone.utc),
        detections=detections,
    )


def test_fire_above_threshold_alerts(alert_config):
    notifier = RecordingNotifier()
    policy = AlertPolicy(alert_config, clock=FakeClock(), notifiers=[notifier])
    alerts = policy.evaluate(_frame_result([1], [0.9]))
    assert len(alerts) == 1
    assert alerts[0].label is DetectionClass.FIRE
    assert alerts[0].confidence == 0.9
    assert len(notifier.sent) == 1


def test_fire_below_threshold_does_not_alert(alert_config):
    policy = AlertPolicy(alert_config, clock=FakeClock(), notifiers=[RecordingNotifier()])
    assert policy.evaluate(_frame_result([1], [0.3])) == []


def test_smoke_does_not_alert_by_default(alert_config):
    policy = AlertPolicy(alert_config, clock=FakeClock(), notifiers=[RecordingNotifier()])
    assert policy.evaluate(_frame_result([0], [0.99])) == []


def test_smoke_alerts_when_enabled(alert_config):
    alert_config.alert_on_smoke = True
    policy = AlertPolicy(alert_config, clock=FakeClock(), notifiers=[RecordingNotifier()])
    alerts = policy.evaluate(_frame_result([0], [0.99]))
    assert alerts[0].label is DetectionClass.SMOKE


def test_alerts_disabled_globally(alert_config):
    alert_config.enable_alerts = False
    policy = AlertPolicy(alert_config, clock=FakeClock(), notifiers=[RecordingNotifier()])
    assert policy.evaluate(_frame_result([1], [0.99])) == []


def test_consecutive_frames_suppresses_single_frame_blip(alert_config):
    alert_config.consecutive_frames = 3
    clock = FakeClock()
    notifier = RecordingNotifier()
    policy = AlertPolicy(alert_config, clock=clock, notifiers=[notifier])

    assert policy.evaluate(_frame_result([1], [0.9], 0)) == []
    assert policy.evaluate(_frame_result([1], [0.9], 1)) == []
    alerts = policy.evaluate(_frame_result([1], [0.9], 2))
    assert len(alerts) == 1
    assert alerts[0].consecutive_frames == 3


def test_streak_resets_after_clear_frame(alert_config):
    alert_config.consecutive_frames = 2
    clock = FakeClock()
    policy = AlertPolicy(alert_config, clock=clock, notifiers=[RecordingNotifier()])

    policy.evaluate(_frame_result([1], [0.9], 0))
    policy.evaluate(_frame_result([], [], 1))  # clears the streak
    assert policy.evaluate(_frame_result([1], [0.9], 2)) == []
    assert len(policy.evaluate(_frame_result([1], [0.9], 3))) == 1


def test_cooldown_suppresses_repeat_alerts(alert_config):
    alert_config.cooldown_seconds = 60.0
    clock = FakeClock()
    notifier = RecordingNotifier()
    policy = AlertPolicy(alert_config, clock=clock, notifiers=[notifier])

    assert len(policy.evaluate(_frame_result([1], [0.9], 0))) == 1
    clock.advance(30)
    assert policy.evaluate(_frame_result([1], [0.9], 1)) == []
    clock.advance(31)  # total 61s > cooldown
    assert len(policy.evaluate(_frame_result([1], [0.9], 2))) == 1
    assert len(notifier.sent) == 2


def test_cooldown_is_per_class(alert_config):
    alert_config.alert_on_smoke = True
    alert_config.cooldown_seconds = 60.0
    clock = FakeClock()
    policy = AlertPolicy(alert_config, clock=clock, notifiers=[RecordingNotifier()])

    assert len(policy.evaluate(_frame_result([1], [0.9], 0))) == 1
    # A smoke alert is not blocked by the fire cooldown.
    assert len(policy.evaluate(_frame_result([0], [0.9], 1))) == 1


def test_min_bbox_area_filters_tiny_detections(alert_config):
    alert_config.min_bbox_area_px = 1000.0
    policy = AlertPolicy(alert_config, clock=FakeClock(), notifiers=[RecordingNotifier()])
    # Box below is 20x20 = 400 px.
    assert policy.evaluate(_frame_result([1], [0.9])) == []


def test_delivery_failure_does_not_raise(alert_config):
    policy = AlertPolicy(
        alert_config, clock=FakeClock(), notifiers=[ExplodingNotifier(), RecordingNotifier()]
    )
    alerts = policy.evaluate(_frame_result([1], [0.9]))
    assert len(alerts) == 1  # pipeline survives a broken notifier


def test_alert_serialisation_shape(alert_config):
    policy = AlertPolicy(alert_config, clock=FakeClock(), notifiers=[RecordingNotifier()])
    alert = policy.evaluate(_frame_result([1], [0.87]))[0]
    payload = alert.to_dict()
    assert payload["event"] == "alert"
    assert payload["label"] == "fire"
    assert payload["confidence"] == 0.87
    assert "FIRE" in alert.format_message()


def test_logging_notifier_always_succeeds():
    alert = Alert(
        label=DetectionClass.FIRE,
        confidence=0.9,
        frame_index=1,
        timestamp=datetime.now(timezone.utc),
        consecutive_frames=1,
    )
    assert LoggingNotifier().send(alert) is True


def test_build_notifiers_includes_webhook_only_when_configured():
    assert len(build_notifiers(AlertConfig(webhook_url=None))) == 1
    assert len(build_notifiers(AlertConfig(webhook_url="http://x"))) == 2


def test_draw_detections_returns_new_image_and_preserves_original():
    import numpy as np

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = [
        Detection(DetectionClass.FIRE, "fire", 0.9, (10.0, 10.0, 60.0, 60.0)),
        Detection(DetectionClass.SMOKE, "smoke", 0.6, (0.0, 0.0, 5.0, 5.0)),
    ]
    annotated = draw_detections(frame, detections)
    assert annotated.shape == frame.shape
    assert annotated.any(), "expected boxes to be drawn"
    assert not frame.any(), "original frame must not be mutated"


def test_draw_detections_handles_empty_list():
    import numpy as np

    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    assert draw_detections(frame, []).shape == frame.shape
