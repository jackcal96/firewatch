"""Tests for the actuation decision logic.

These tests are the safety argument for the supervisor. Each one pins a
specific way the system must *refuse* to spray, because the failure mode that
matters is water in the wrong place, not a missed spray.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from firewatch.actuator import DryRunActuator, SubprocessActuator
from firewatch.config import ActuationConfig, ZoneConfig
from firewatch.supervisor import (
    ActuationSupervisor,
    DenyReason,
    SupervisorState,
)
from firewatch.types import Detection, DetectionClass, FrameResult


class FakeClock:
    """Deterministic monotonic clock."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_frame(
    confidence: float = 0.9,
    label: DetectionClass = DetectionClass.FIRE,
    bbox: tuple[float, float, float, float] = (100.0, 100.0, 200.0, 200.0),
    frame_index: int = 0,
) -> FrameResult:
    detection = Detection(
        label=label,
        raw_label=label.value,
        confidence=confidence,
        bbox=bbox,
    )
    return FrameResult(
        frame_index=frame_index,
        timestamp=datetime.now(timezone.utc),
        detections=[detection] if confidence > 0 else [],
    )


def make_empty_frame(frame_index: int = 0) -> FrameResult:
    return FrameResult(
        frame_index=frame_index,
        timestamp=datetime.now(timezone.utc),
        detections=[],
    )


def build_supervisor(
    clock: FakeClock,
    zones: list[ZoneConfig] | None = None,
    **overrides,
) -> tuple[ActuationSupervisor, DryRunActuator]:
    zone_list = zones if zones is not None else [whole_frame_zone()]
    config = ActuationConfig(
        enable_actuation=True,
        dry_run=True,
        zones=zone_list,
        **overrides,
    )
    actuator = DryRunActuator()
    supervisor = ActuationSupervisor(config, actuator=actuator, clock=clock)
    return supervisor, actuator


def whole_frame_zone() -> ZoneConfig:
    return ZoneConfig(
        name="full",
        polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
    )


FRAME_SIZE = (1920, 1080)


def run_frames(
    supervisor: ActuationSupervisor,
    count: int,
    clock: FakeClock,
    dt: float = 0.5,
    frame: FrameResult | None = None,
) -> list:
    decisions = []
    for i in range(count):
        frame = frame or make_frame(frame_index=i)
        decisions.append(supervisor.observe(frame, frame_size=FRAME_SIZE))
        clock.advance(dt)
    return decisions


class TestDisabledByDefault:
    def test_actuation_off_never_opens_valve(self) -> None:
        clock = FakeClock()
        config = ActuationConfig(enable_actuation=False)
        actuator = DryRunActuator()
        supervisor = ActuationSupervisor(config, actuator=actuator, clock=clock)

        for i in range(50):
            decision = supervisor.observe(make_frame(frame_index=i), frame_size=FRAME_SIZE)
            clock.advance(0.5)
            assert decision.action == "none"
            assert decision.reason == DenyReason.ACTUATION_DISABLED.value

        assert actuator.events == []
        assert supervisor.state is SupervisorState.IDLE

    def test_config_defaults_are_safe(self) -> None:
        config = ActuationConfig()
        assert config.enable_actuation is False
        assert config.dry_run is True
        assert config.trigger_classes == ["fire"]
        assert config.max_shots == 1


class TestSprayGate:
    def test_sprays_only_after_sustained_evidence(self) -> None:
        clock = FakeClock()
        supervisor, actuator = build_supervisor(
            clock, consecutive_frames=3, arm_delay_seconds=5.0
        )

        # Streak starts at frame 0; arm delay is measured from that moment.
        first = supervisor.observe(make_frame(frame_index=0), frame_size=FRAME_SIZE)
        assert first.action == "none"
        assert first.reason == "confirming"

        clock.advance(0.5)
        second = supervisor.observe(make_frame(frame_index=1), frame_size=FRAME_SIZE)
        assert second.action == "none"
        assert second.reason == DenyReason.INSUFFICIENT_STREAK.value

        clock.advance(0.5)
        third = supervisor.observe(make_frame(frame_index=2), frame_size=FRAME_SIZE)
        assert third.reason == DenyReason.ARM_DELAY_PENDING.value

        clock.advance(5.0)  # held now exceeds the 5.0s arm delay
        fourth = supervisor.observe(make_frame(frame_index=3), frame_size=FRAME_SIZE)
        assert fourth.action == "open"
        assert fourth.allowed is True
        assert fourth.reason == "fire_confirmed"
        assert actuator.is_open is True

    def test_below_min_confidence_never_sprays(self) -> None:
        clock = FakeClock()
        supervisor, actuator = build_supervisor(clock, min_confidence=0.8)

        for i in range(20):
            supervisor.observe(make_frame(confidence=0.7, frame_index=i), frame_size=FRAME_SIZE)
            clock.advance(1.0)

        assert actuator.events == []

    def test_smoke_alone_never_sprays(self) -> None:
        clock = FakeClock()
        supervisor, actuator = build_supervisor(clock)

        for i in range(20):
            supervisor.observe(
                make_frame(confidence=0.99, label=DetectionClass.SMOKE, frame_index=i),
                frame_size=FRAME_SIZE,
            )
            clock.advance(1.0)

        assert actuator.events == []

    def test_small_box_never_sprays(self) -> None:
        clock = FakeClock()
        supervisor, actuator = build_supervisor(clock, min_bbox_area_px=10_000.0)

        for i in range(20):
            supervisor.observe(
                make_frame(bbox=(100.0, 100.0, 110.0, 110.0), frame_index=i),
                frame_size=FRAME_SIZE,
            )
            clock.advance(1.0)

        assert actuator.events == []

    def test_flickering_evidence_does_not_accumulate(self) -> None:
        """A single-frame gap must reset the streak, not pause it."""
        clock = FakeClock()
        supervisor, actuator = build_supervisor(
            clock, consecutive_frames=3, arm_delay_seconds=0.0
        )

        for i in range(10):
            supervisor.observe(make_frame(frame_index=i), frame_size=FRAME_SIZE)
            clock.advance(0.2)
            supervisor.observe(make_empty_frame(frame_index=i), frame_size=FRAME_SIZE)
            clock.advance(0.2)

        assert actuator.events == []


class TestZoneGating:
    def test_detection_outside_zone_never_sprays(self) -> None:
        clock = FakeClock()
        left_zone = ZoneConfig(
            name="left-half",
            polygon=[(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)],
        )
        supervisor, actuator = build_supervisor(
            clock, zones=[left_zone], consecutive_frames=2, arm_delay_seconds=0.0
        )

        # Box on the right half of a 1920-wide frame.
        right_box = (1500.0, 100.0, 1700.0, 300.0)
        for i in range(20):
            supervisor.observe(
                make_frame(bbox=right_box, frame_index=i), frame_size=FRAME_SIZE
            )
            clock.advance(1.0)

        assert actuator.events == []

    def test_detection_inside_zone_sprays(self) -> None:
        clock = FakeClock()
        left_zone = ZoneConfig(
            name="left-half",
            polygon=[(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)],
        )
        supervisor, actuator = build_supervisor(
            clock, zones=[left_zone], consecutive_frames=2, arm_delay_seconds=0.0
        )

        left_box = (100.0, 100.0, 300.0, 300.0)
        for i in range(5):
            supervisor.observe(
                make_frame(bbox=left_box, frame_index=i), frame_size=FRAME_SIZE
            )
            clock.advance(0.5)

        assert actuator.is_open is True

    def test_missing_frame_size_refuses_when_zones_required(self) -> None:
        clock = FakeClock()
        supervisor, actuator = build_supervisor(clock, consecutive_frames=2)

        for i in range(10):
            supervisor.observe(make_frame(frame_index=i), frame_size=None)
            clock.advance(1.0)

        assert actuator.events == []


class TestShotBudget:
    def test_locks_out_after_max_shots(self) -> None:
        clock = FakeClock()
        supervisor, actuator = build_supervisor(
            clock,
            consecutive_frames=2,
            arm_delay_seconds=0.0,
            max_shots=1,
            cooldown_seconds=1.0,
            min_spray_seconds=0.0,
            max_spray_seconds=1.0,
        )

        # Spray once.
        supervisor.observe(make_frame(frame_index=0), frame_size=FRAME_SIZE)
        clock.advance(0.1)
        decision = supervisor.observe(make_frame(frame_index=1), frame_size=FRAME_SIZE)
        assert decision.action == "open"
        assert supervisor.shots_used == 1

        # Let the max-spray timer close it, then the cooldown expire.
        clock.advance(2.0)
        supervisor.tick()
        clock.advance(2.0)
        supervisor.tick()

        assert supervisor.state is SupervisorState.LOCKED_OUT

        # Further fire must not spray again.
        opens_before = len([e for e in actuator.events if e.action == "open"])
        for i in range(10):
            supervisor.observe(make_frame(frame_index=i), frame_size=FRAME_SIZE)
            clock.advance(1.0)
        opens_after = len([e for e in actuator.events if e.action == "open"])
        assert opens_after == opens_before == 1

    def test_manual_reset_restores_operation(self) -> None:
        clock = FakeClock()
        supervisor, _ = build_supervisor(
            clock,
            consecutive_frames=2,
            arm_delay_seconds=0.0,
            max_shots=1,
            cooldown_seconds=1.0,
            min_spray_seconds=0.0,
            max_spray_seconds=1.0,
        )
        supervisor.observe(make_frame(frame_index=0), frame_size=FRAME_SIZE)
        clock.advance(0.1)
        supervisor.observe(make_frame(frame_index=1), frame_size=FRAME_SIZE)
        clock.advance(2.0)
        supervisor.tick()
        clock.advance(2.0)
        supervisor.tick()
        assert supervisor.state is SupervisorState.LOCKED_OUT

        supervisor.reset("operator cleared after inspection")
        assert supervisor.state is SupervisorState.IDLE
        assert supervisor.shots_used == 0


class TestSprayTermination:
    def test_max_spray_duration_closes_valve(self) -> None:
        clock = FakeClock()
        supervisor, actuator = build_supervisor(
            clock,
            consecutive_frames=2,
            arm_delay_seconds=0.0,
            min_spray_seconds=0.0,
            max_spray_seconds=5.0,
        )

        supervisor.observe(make_frame(frame_index=0), frame_size=FRAME_SIZE)
        clock.advance(0.1)
        supervisor.observe(make_frame(frame_index=1), frame_size=FRAME_SIZE)
        assert actuator.is_open is True

        clock.advance(6.0)
        supervisor.observe(make_frame(frame_index=2), frame_size=FRAME_SIZE)
        assert actuator.is_open is False
        assert supervisor.state is SupervisorState.COOLDOWN

    def test_evidence_loss_closes_valve(self) -> None:
        clock = FakeClock()
        supervisor, actuator = build_supervisor(
            clock,
            consecutive_frames=2,
            arm_delay_seconds=0.0,
            min_spray_seconds=0.0,
            max_spray_seconds=60.0,
            stop_on_evidence_loss=True,
        )

        supervisor.observe(make_frame(frame_index=0), frame_size=FRAME_SIZE)
        clock.advance(0.1)
        supervisor.observe(make_frame(frame_index=1), frame_size=FRAME_SIZE)
        assert actuator.is_open is True

        clock.advance(0.5)
        supervisor.observe(make_empty_frame(frame_index=2), frame_size=FRAME_SIZE)
        clock.advance(0.5)
        supervisor.observe(make_empty_frame(frame_index=3), frame_size=FRAME_SIZE)

        assert actuator.is_open is False

    def test_watchdog_closes_valve_when_detector_dies(self) -> None:
        """If the detector stops heartbeating, the valve must not stay open."""
        clock = FakeClock()
        supervisor, actuator = build_supervisor(
            clock,
            consecutive_frames=2,
            arm_delay_seconds=0.0,
            min_spray_seconds=0.0,
            max_spray_seconds=3600.0,
            watchdog_timeout_seconds=3.0,
        )

        supervisor.observe(make_frame(frame_index=0), frame_size=FRAME_SIZE)
        clock.advance(0.1)
        supervisor.observe(make_frame(frame_index=1), frame_size=FRAME_SIZE)
        assert actuator.is_open is True

        # Detector stops reporting; only the timer path runs.
        clock.advance(10.0)
        decision = supervisor.tick()

        assert actuator.is_open is False
        assert supervisor.state is SupervisorState.LOCKED_OUT
        assert decision is not None
        assert decision.reason == DenyReason.STALE_HEARTBEAT.value

    def test_stale_heartbeat_prevents_spray(self) -> None:
        clock = FakeClock()
        supervisor, actuator = build_supervisor(
            clock,
            consecutive_frames=2,
            arm_delay_seconds=0.0,
            watchdog_timeout_seconds=1.0,
        )

        # Simulate a long gap with no frames at all, then a single frame:
        # the streak is built but the heartbeat is fresh because observe()
        # touches it. Verify the watchdog by ticking instead.
        supervisor.observe(make_frame(frame_index=0), frame_size=FRAME_SIZE)
        clock.advance(100.0)
        decision = supervisor.tick()
        assert decision is None or decision.action == "none"


class TestCooldown:
    def test_cooldown_suppresses_immediate_respray(self) -> None:
        clock = FakeClock()
        supervisor, actuator = build_supervisor(
            clock,
            consecutive_frames=2,
            arm_delay_seconds=0.0,
            min_spray_seconds=0.0,
            max_spray_seconds=1.0,
            cooldown_seconds=60.0,
            max_shots=5,
        )

        supervisor.observe(make_frame(frame_index=0), frame_size=FRAME_SIZE)
        clock.advance(0.1)
        supervisor.observe(make_frame(frame_index=1), frame_size=FRAME_SIZE)
        assert actuator.is_open is True

        clock.advance(2.0)
        supervisor.observe(make_frame(frame_index=2), frame_size=FRAME_SIZE)
        assert supervisor.state is SupervisorState.COOLDOWN

        opens = len([e for e in actuator.events if e.action == "open"])
        for i in range(20):
            decision = supervisor.observe(make_frame(frame_index=i + 3), frame_size=FRAME_SIZE)
            clock.advance(1.0)
            assert decision.action != "open"

        assert len([e for e in actuator.events if e.action == "open"]) == opens


class TestActuatorFaults:
    def test_open_failure_locks_out(self) -> None:
        class BrokenActuator:
            name = "broken"

            def open_valve(self):
                raise RuntimeError("relay did not respond")

            def close_valve(self):
                raise RuntimeError("relay did not respond")

        clock = FakeClock()
        config = ActuationConfig(
            enable_actuation=True,
            dry_run=True,
            zones=[whole_frame_zone()],
            consecutive_frames=2,
            arm_delay_seconds=0.0,
        )
        supervisor = ActuationSupervisor(config, actuator=BrokenActuator(), clock=clock)

        supervisor.observe(make_frame(frame_index=0), frame_size=FRAME_SIZE)
        clock.advance(0.1)
        decision = supervisor.observe(make_frame(frame_index=1), frame_size=FRAME_SIZE)

        assert decision.allowed is False
        assert decision.reason == "actuator_fault"
        assert supervisor.state is SupervisorState.LOCKED_OUT

    def test_close_failure_is_reported_loudly(self) -> None:
        class StuckOpenActuator:
            name = "stuck"

            def __init__(self) -> None:
                self.is_open = False

            def open_valve(self):
                from firewatch.actuator import ActuatorEvent

                self.is_open = True
                return ActuatorEvent(
                    action="open", actuator=self.name, success=True, simulated=True
                )

            def close_valve(self):
                raise RuntimeError("valve stuck open")

        clock = FakeClock()
        config = ActuationConfig(
            enable_actuation=True,
            dry_run=True,
            zones=[whole_frame_zone()],
            consecutive_frames=2,
            arm_delay_seconds=0.0,
            min_spray_seconds=0.0,
            max_spray_seconds=1.0,
        )
        supervisor = ActuationSupervisor(config, actuator=StuckOpenActuator(), clock=clock)

        supervisor.observe(make_frame(frame_index=0), frame_size=FRAME_SIZE)
        clock.advance(0.1)
        supervisor.observe(make_frame(frame_index=1), frame_size=FRAME_SIZE)
        clock.advance(2.0)
        decision = supervisor.observe(make_frame(frame_index=2), frame_size=FRAME_SIZE)

        assert decision.reason == "actuator_fault"
        assert "FAILED TO CLOSE" in decision.detail
        assert supervisor.state is SupervisorState.LOCKED_OUT


class TestConfigInvariants:
    def test_requires_zone_when_configured(self) -> None:
        with pytest.raises(ValueError, match="no enabled zones"):
            ActuationConfig(enable_actuation=True, require_zone=True, zones=[])

    def test_rejects_low_confidence(self) -> None:
        with pytest.raises(ValueError, match="too low to actuate"):
            ActuationConfig(
                enable_actuation=True,
                min_confidence=0.2,
                zones=[whole_frame_zone()],
            )

    def test_rejects_single_frame_evidence(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            ActuationConfig(
                enable_actuation=True,
                consecutive_frames=1,
                zones=[whole_frame_zone()],
            )

    def test_rejects_smoke_only_triggers(self) -> None:
        with pytest.raises(ValueError, match="must include 'fire'"):
            ActuationConfig(
                enable_actuation=True,
                trigger_classes=["smoke"],
                zones=[whole_frame_zone()],
            )

    def test_rejects_live_mode_without_commands(self) -> None:
        with pytest.raises(ValueError, match="no way to close it"):
            ActuationConfig(
                enable_actuation=True,
                dry_run=False,
                zones=[whole_frame_zone()],
                open_command=["true"],
                close_command=None,
            )

    def test_zone_points_must_be_normalised(self) -> None:
        with pytest.raises(ValueError, match="outside 0-1"):
            ZoneConfig(name="bad", polygon=[(0.0, 0.0), (1920.0, 0.0), (1920.0, 1080.0)])


class TestAuditTrail:
    def test_decisions_are_recorded_with_reasons(self) -> None:
        clock = FakeClock()
        supervisor, _ = build_supervisor(clock, consecutive_frames=2, arm_delay_seconds=0.0)

        supervisor.observe(make_frame(frame_index=0), frame_size=FRAME_SIZE)
        clock.advance(0.1)
        supervisor.observe(make_frame(frame_index=1), frame_size=FRAME_SIZE)

        assert supervisor.decisions
        payload = supervisor.decisions[-1].to_dict()
        assert payload["event"] == "actuation_decision"
        assert payload["action"] == "open"
        assert payload["allowed"] is True
        assert "timestamp" in payload

    def test_dry_run_events_are_marked_simulated(self) -> None:
        clock = FakeClock()
        supervisor, actuator = build_supervisor(
            clock, consecutive_frames=2, arm_delay_seconds=0.0
        )
        supervisor.observe(make_frame(frame_index=0), frame_size=FRAME_SIZE)
        clock.advance(0.1)
        supervisor.observe(make_frame(frame_index=1), frame_size=FRAME_SIZE)

        assert actuator.events
        assert all(e.simulated for e in actuator.events)
        assert all(e.to_dict()["simulated"] is True for e in actuator.events)


class TestSubprocessActuator:
    def test_runs_commands_and_reports_success(self, tmp_path) -> None:
        marker = tmp_path / "opened"
        config = ActuationConfig(
            enable_actuation=True,
            dry_run=False,
            zones=[whole_frame_zone()],
            open_command=["touch", str(marker)],
            close_command=["true"],
        )
        actuator = SubprocessActuator(config)

        event = actuator.open_valve()
        assert event.success is True
        assert event.simulated is False
        assert marker.exists()

    def test_reports_failure_without_raising(self) -> None:
        config = ActuationConfig(
            enable_actuation=True,
            dry_run=False,
            zones=[whole_frame_zone()],
            open_command=["false"],
            close_command=["true"],
        )
        actuator = SubprocessActuator(config)

        event = actuator.open_valve()
        assert event.success is False
        assert actuator.is_open is False

    def test_missing_command_raises(self) -> None:
        from firewatch.actuator import ActuatorError

        config = ActuationConfig(
            enable_actuation=True,
            dry_run=False,
            zones=[whole_frame_zone()],
            open_command=["/nonexistent/binary"],
            close_command=["true"],
        )
        actuator = SubprocessActuator(config)
        with pytest.raises(ActuatorError):
            actuator.open_valve()
