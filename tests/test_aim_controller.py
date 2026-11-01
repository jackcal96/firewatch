"""Tests for the aiming stack and the spray interlock it drives.

The central claim these tests defend is that a turret cannot cause water to
flow. Aiming and spraying are separate: the mount may track a fire freely,
but the valve opens only when the fire is confirmed *and* the nozzle is
verified to be on it.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from firewatch.aim_controller import AimController, build_aim_controller
from firewatch.aiming import AffineAxis, Calibration, CalibrationError, CalibrationPoint
from firewatch.actuator import DryRunActuator
from firewatch.config import AppConfig, TurretConfig
from firewatch.logging_utils import DetectionLogger
from firewatch.supervisor import ActuationSupervisor, DenyReason
from firewatch.turret import DryRunTurret, TurretError
from firewatch.types import Detection, DetectionClass, FrameResult


class FakeClock:
    """Deterministic monotonic clock so timing tests never sleep."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class StubTurret:
    """A turret whose reported position can be forced away from the command.

    This models the failure the interlock exists for: an open-loop stepper
    that is told to move, reports having moved, and is not where it claims.
    """

    def __init__(self, *, homed: bool = True, lie_by: float = 0.0) -> None:
        self._position: tuple[float, float] | None = (0.0, 0.0) if homed else None
        self._homed = homed
        self.lie_by = lie_by
        self.moves: list[tuple[float, float]] = []
        self.closed = False
        self.fail_goto = False

    @property
    def name(self) -> str:
        return "stub"

    @property
    def position(self) -> tuple[float, float] | None:
        return self._position

    @property
    def is_homed(self) -> bool:
        return self._homed

    def home(self):
        self._homed = True
        self._position = (0.0, 0.0)
        return _event("home", 0.0, 0.0)

    def goto(self, pan: float, tilt: float):
        if self.fail_goto:
            raise TurretError("simulated stall")
        self.moves.append((pan, tilt))
        # Report the requested bearing, optionally offset to simulate a mount
        # that did not actually get there.
        self._position = (pan + self.lie_by, tilt + self.lie_by)
        return _event("goto", self._position[0], self._position[1])

    def close(self) -> None:
        self.closed = True


def _event(action: str, pan: float, tilt: float):
    from firewatch.turret import TurretEvent

    return TurretEvent(
        action=action, turret="stub", success=True, simulated=False, pan=pan, tilt=tilt
    )


def _calibration() -> Calibration:
    """60 deg across the frame width, 30 deg down its height."""
    return Calibration.fit(
        [
            CalibrationPoint(u=0.0, v=0.0, pan=-30.0, tilt=15.0),
            CalibrationPoint(u=1.0, v=0.0, pan=30.0, tilt=15.0),
            CalibrationPoint(u=0.0, v=1.0, pan=-30.0, tilt=-15.0),
            CalibrationPoint(u=1.0, v=1.0, pan=30.0, tilt=-15.0),
        ]
    )


def _turret_config(**overrides) -> TurretConfig:
    base = {
        "enable_turret": True,
        "dry_run": True,
        "aim_tolerance_degrees": 5.0,
        "settle_seconds": 0.0,
        "min_pan": -60.0,
        "max_pan": 60.0,
        "min_tilt": -45.0,
        "max_tilt": 45.0,
    }
    base.update(overrides)
    return TurretConfig(**base)


def _controller(clock: FakeClock, turret=None, **overrides) -> AimController:
    return AimController(
        _turret_config(**overrides),
        turret or StubTurret(),
        _calibration(),
        clock=clock,
        sleep=lambda _s: None,
    )


class TestAimController:
    def test_aims_at_a_centred_box(self) -> None:
        clock = FakeClock()
        controller = _controller(clock)
        outcome = controller.aim_at((300.0, 220.0, 340.0, 260.0), (640, 480))
        assert outcome.ok is True
        assert outcome.error_degrees is not None

    def test_refuses_when_not_homed(self) -> None:
        clock = FakeClock()
        controller = _controller(clock, StubTurret(homed=False))
        outcome = controller.aim_at((300.0, 220.0, 340.0, 260.0), (640, 480))
        assert outcome.ok is False
        assert outcome.reason == DenyReason.AIM_UNAVAILABLE.value
        assert "homed" in outcome.detail

    def test_refuses_when_target_is_outside_envelope(self) -> None:
        clock = FakeClock()
        controller = _controller(clock, min_pan=-2.0, max_pan=2.0)
        # A box hard against the right edge needs pan near +30.
        outcome = controller.aim_at((630.0, 200.0, 640.0, 280.0), (640, 480))
        assert outcome.ok is False
        assert outcome.reason == DenyReason.AIM_OUTSIDE_ENVELOPE.value

    def test_detects_a_turret_that_did_not_arrive(self) -> None:
        # The core interlock: the mount is told to move, claims to have moved,
        # but its reported bearing is outside tolerance. Aim must fail.
        clock = FakeClock()
        controller = _controller(clock, StubTurret(lie_by=20.0))
        outcome = controller.aim_at((300.0, 220.0, 340.0, 260.0), (640, 480))
        assert outcome.ok is False
        assert outcome.reason == DenyReason.AIM_NOT_SETTLED.value
        assert outcome.error_degrees is not None and outcome.error_degrees > 5.0

    def test_reports_error_within_tolerance_as_success(self) -> None:
        clock = FakeClock()
        controller = _controller(clock, StubTurret(lie_by=1.0))
        outcome = controller.aim_at((300.0, 220.0, 340.0, 260.0), (640, 480))
        assert outcome.ok is True

    def test_turret_fault_is_reported(self) -> None:
        clock = FakeClock()
        turret = StubTurret()
        turret.fail_goto = True
        controller = _controller(clock, turret)
        outcome = controller.aim_at((300.0, 220.0, 340.0, 260.0), (640, 480))
        assert outcome.ok is False
        assert outcome.reason == DenyReason.TURRET_FAULT.value

    def test_degenerate_calibration_is_refused(self) -> None:
        clock = FakeClock()
        controller = AimController(
            _turret_config(),
            StubTurret(),
            Calibration(AffineAxis(0.0, 0.0), AffineAxis(1.0, 0.0)),
            clock=clock,
            sleep=lambda _s: None,
        )
        outcome = controller.aim_at((300.0, 220.0, 340.0, 260.0), (640, 480))
        # A zero pan scale maps every target to one bearing, so aiming is
        # meaningless. It must not be reported as success.
        assert outcome.ok is False

    def test_moves_are_rate_limited(self) -> None:
        # Commanding a mount faster than it settles produces partially
        # completed moves and a position reading that means nothing.
        clock = FakeClock()
        turret = StubTurret()
        controller = _controller(clock, turret, settle_seconds=1.0)
        controller.aim_at((300.0, 220.0, 340.0, 260.0), (640, 480))
        first_moves = len(turret.moves)
        controller.aim_at((310.0, 230.0, 350.0, 270.0), (640, 480))
        assert len(turret.moves) == first_moves  # no new move within settle

        clock.advance(2.0)
        controller.aim_at((600.0, 400.0, 640.0, 460.0), (640, 480))
        assert len(turret.moves) > first_moves

    def test_home_failure_is_reported_not_raised(self) -> None:
        clock = FakeClock()

        class BrokenTurret(StubTurret):
            def home(self):
                raise TurretError("no limit switch")

        outcome = _controller(clock, BrokenTurret()).home()
        assert outcome.ok is False
        assert outcome.reason == DenyReason.TURRET_FAULT.value


class TestAimGateInSupervisor:
    """The supervisor must refuse to spray when the aim is not good."""

    def _supervisor(self, aim_gate, **overrides):
        from firewatch.config import ActuationConfig

        config = ActuationConfig(
            enable_actuation=True,
            dry_run=True,
            trigger_classes=["fire"],
            min_confidence=0.5,
            consecutive_frames=2,
            arm_delay_seconds=0.0,
            require_zone=False,
            max_spray_seconds=5.0,
            max_shots=1,
            **overrides,
        )
        return ActuationSupervisor(
            config, actuator=DryRunActuator(), aim_gate=aim_gate
        )

    def _fire_frame(self, frame_index: int = 0) -> FrameResult:
        return FrameResult(
            frame_index=frame_index,
            timestamp=datetime.now(timezone.utc),
            detections=[
                Detection(
                    label=DetectionClass.FIRE,
                    raw_label="fire",
                    confidence=0.9,
                    bbox=(300.0, 220.0, 340.0, 260.0),
                )
            ],
        )

    def test_sprays_when_aim_is_good(self) -> None:
        clock = FakeClock()
        controller = _controller(clock)
        supervisor = self._supervisor(controller)

        supervisor.observe(self._fire_frame(0), frame_size=(640, 480))
        decision = supervisor.observe(self._fire_frame(1), frame_size=(640, 480))
        assert decision.action == "open"
        assert decision.allowed is True

    def test_refuses_to_spray_when_aim_is_off_target(self) -> None:
        # The mount never arrives, so the valve must stay shut even though the
        # fire is confirmed. This is the whole point of the aim gate.
        clock = FakeClock()
        controller = _controller(clock, StubTurret(lie_by=20.0))
        supervisor = self._supervisor(controller)

        supervisor.observe(self._fire_frame(0), frame_size=(640, 480))
        decision = supervisor.observe(self._fire_frame(1), frame_size=(640, 480))
        assert decision.action == "none"
        assert decision.reason == DenyReason.AIM_NOT_SETTLED.value
        assert supervisor.state.value != "spraying"

    def test_refuses_when_turret_is_unhomed(self) -> None:
        clock = FakeClock()
        controller = _controller(clock, StubTurret(homed=False))
        supervisor = self._supervisor(controller)

        supervisor.observe(self._fire_frame(0), frame_size=(640, 480))
        decision = supervisor.observe(self._fire_frame(1), frame_size=(640, 480))
        assert decision.action == "none"
        assert decision.reason == DenyReason.AIM_UNAVAILABLE.value

    def test_refuses_when_aim_raises(self) -> None:
        # A broken aiming stack must fail closed, not crash the loop.
        class ExplodingGate:
            def aim_at(self, bbox, frame_size):
                raise RuntimeError("serial port vanished")

        supervisor = self._supervisor(ExplodingGate())
        supervisor.observe(self._fire_frame(0), frame_size=(640, 480))
        decision = supervisor.observe(self._fire_frame(1), frame_size=(640, 480))
        assert decision.action == "none"
        assert decision.reason == DenyReason.TURRET_FAULT.value

    def test_no_aim_gate_preserves_fixed_nozzle_behaviour(self) -> None:
        # With no turret configured the gate is absent and behaviour is
        # exactly as before the aiming layer existed.
        supervisor = self._supervisor(None)
        supervisor.observe(self._fire_frame(0), frame_size=(640, 480))
        decision = supervisor.observe(self._fire_frame(1), frame_size=(640, 480))
        assert decision.action == "open"

    def test_refuses_when_frame_size_is_unknown(self) -> None:
        # Without frame dimensions the target bearing cannot be computed, so
        # there is nothing to verify against. Refuse rather than spray blind.
        clock = FakeClock()
        controller = _controller(clock)
        supervisor = self._supervisor(controller)
        supervisor.observe(self._fire_frame(0))
        decision = supervisor.observe(self._fire_frame(1))
        assert decision.action == "none"
        assert decision.reason == DenyReason.AIM_UNAVAILABLE.value


class TestAimAndSprayAreSeparate:
    """Aiming must not be able to cause water, and vice versa."""

    def test_tracking_a_fire_does_not_open_the_valve(self, tmp_path) -> None:
        # Driven through the real pipeline, because tracking lives there.
        # The turret follows the fire, but with the spray gates unmet the
        # valve must never be commanded.
        from firewatch.actuator import DryRunActuator
        from firewatch.config import (
            ActuationConfig,
            AppConfig,
            LoggingConfig,
            ModelConfig,
            SourceConfig,
        )
        from firewatch.detector import FireSmokeDetector, RawDetections
        from firewatch.pipeline import FireSmokePipeline

        import numpy as np

        clock = FakeClock()
        turret = StubTurret()
        controller = _controller(clock, turret)

        config = AppConfig(
            source=SourceConfig(source_type="file", file_path="unused.mp4"),
            model=ModelConfig(model_path="unused.pt"),
            actuation=ActuationConfig(
                enable_actuation=True,
                dry_run=True,
                min_confidence=0.5,
                consecutive_frames=10,  # deliberately never reached
                require_zone=False,
            ),
            logging=LoggingConfig(log_dir=str(tmp_path / "logs")),
        )
        actuator = DryRunActuator()
        supervisor = ActuationSupervisor(
            config.actuation, actuator=actuator, aim_gate=controller
        )

        class Backend:
            def predict(self, frame):
                return RawDetections(
                    xyxy=np.array([[10.0, 10.0, 40.0, 40.0]], dtype=np.float32),
                    conf=np.array([0.9], dtype=np.float32),
                    cls=np.array([1], dtype=np.int64),
                    names={1: "fire"},
                )

        pipeline = FireSmokePipeline(
            config,
            FireSmokeDetector(Backend(), config.model),
            detection_logger=DetectionLogger(config.logging),
            supervisor=supervisor,
            aim_controller=controller,
        )

        for i in range(3):
            pipeline.process_frame(np.zeros((64, 64, 3), dtype=np.uint8), i)
        pipeline.close()

        assert turret.moves, "the turret should track the fire"
        assert actuator.events == [], "tracking must not open the valve"

    def test_aim_gate_is_not_consulted_when_actuation_is_disabled(self) -> None:
        from firewatch.config import ActuationConfig

        class CountingGate:
            def __init__(self) -> None:
                self.calls = 0

            def aim_at(self, bbox, frame_size):
                self.calls += 1
                raise AssertionError("aim gate must not run when actuation is off")

        gate = CountingGate()
        config = ActuationConfig(enable_actuation=False)
        supervisor = ActuationSupervisor(config, actuator=DryRunActuator(), aim_gate=gate)
        supervisor.observe(
            FrameResult(
                frame_index=0,
                timestamp=datetime.now(timezone.utc),
                detections=[
                    Detection(
                        label=DetectionClass.FIRE,
                        raw_label="fire",
                        confidence=0.99,
                        bbox=(300.0, 220.0, 340.0, 260.0),
                    )
                ],
            ),
            frame_size=(640, 480),
        )
        assert gate.calls == 0


class TestBuildAimController:
    def test_returns_none_when_turret_disabled(self, tmp_path) -> None:
        config = AppConfig()
        assert build_aim_controller(config) is None

    def test_requires_a_calibration(self, tmp_path) -> None:
        config = AppConfig()
        config.turret.enable_turret = True
        config.turret.calibration_path = str(tmp_path / "absent.json")
        with pytest.raises(CalibrationError, match="calibrate"):
            build_aim_controller(config)

    def test_rejects_a_degenerate_calibration(self, tmp_path) -> None:
        path = tmp_path / "cal.json"
        Calibration(AffineAxis(0.0, 0.0), AffineAxis(1.0, 0.0)).save(path)
        config = AppConfig()
        config.turret.enable_turret = True
        config.turret.calibration_path = str(path)
        with pytest.raises(CalibrationError, match="degenerate"):
            build_aim_controller(config)

    def test_builds_from_a_valid_calibration(self, tmp_path) -> None:
        path = tmp_path / "cal.json"
        _calibration().save(path, points=[])
        config = AppConfig()
        config.turret.enable_turret = True
        config.turret.calibration_path = str(path)
        controller = build_aim_controller(config)
        assert controller is not None
        controller.close()
