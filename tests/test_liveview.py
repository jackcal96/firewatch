"""Tests for the live preview overlay.

The overlay is a diagnostic, so these tests check it draws the things a trial
operator needs to judge aiming, and - more importantly - that it degrades
safely when no display is available rather than taking down the control loop.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from firewatch.aiming import AffineAxis, Calibration, CalibrationPoint
from firewatch.config import TurretConfig
from firewatch.liveview import LiveView, _opencv_has_gui, display_available, draw_aim_overlay
from firewatch.supervisor import ActuationDecision, SupervisorState
from firewatch.types import Detection, DetectionClass, FrameResult


def _frame(width: int = 320, height: int = 240) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def _calibration() -> Calibration:
    return Calibration.fit(
        [
            CalibrationPoint(u=0.0, v=0.0, pan=-30.0, tilt=15.0),
            CalibrationPoint(u=1.0, v=0.0, pan=30.0, tilt=15.0),
            CalibrationPoint(u=0.0, v=1.0, pan=-30.0, tilt=-15.0),
            CalibrationPoint(u=1.0, v=1.0, pan=30.0, tilt=-15.0),
        ]
    )


def _result(with_fire: bool = True) -> FrameResult:
    detections = (
        [
            Detection(
                label=DetectionClass.FIRE,
                raw_label="fire",
                confidence=0.87,
                bbox=(100.0, 80.0, 180.0, 150.0),
            )
        ]
        if with_fire
        else []
    )
    return FrameResult(
        frame_index=42, timestamp=datetime.now(timezone.utc), detections=detections
    )


def _decision(allowed: bool = True) -> ActuationDecision:
    return ActuationDecision(
        state=SupervisorState.SPRAYING,
        action="open" if allowed else "none",
        allowed=allowed,
        reason="fire_confirmed" if allowed else "aim_not_settled",
        detail="",
    )


class TestDrawAimOverlay:
    def test_returns_a_copy_not_the_original(self) -> None:
        frame = _frame()
        overlay = draw_aim_overlay(frame, _result(), TurretConfig())
        assert overlay is not frame
        assert not np.array_equal(overlay, frame)

    def test_does_not_mutate_the_input(self) -> None:
        frame = _frame()
        before = frame.copy()
        draw_aim_overlay(frame, _result(), TurretConfig(), calibration=_calibration())
        assert np.array_equal(frame, before)

    def test_draws_something_for_a_fire_frame(self) -> None:
        frame = _frame()
        overlay = draw_aim_overlay(
            frame,
            _result(),
            TurretConfig(),
            calibration=_calibration(),
            aim_bearing=(0.0, 0.0),
            aim_ok=True,
            state="spraying",
            decision=_decision(),
        )
        assert overlay.sum() > frame.sum()

    def test_handles_a_frame_with_no_detections(self) -> None:
        # Must not raise: quiet frames are the common case.
        overlay = draw_aim_overlay(_frame(), _result(with_fire=False), TurretConfig())
        assert overlay.shape == _frame().shape

    def test_handles_a_missing_calibration(self) -> None:
        # Before calibration exists the overlay still has to render.
        overlay = draw_aim_overlay(
            _frame(), _result(), TurretConfig(), calibration=None, aim_bearing=(1.0, 1.0)
        )
        assert overlay.shape == _frame().shape

    def test_off_frame_aim_is_still_shown(self) -> None:
        # An aim bearing that maps outside the image is itself the finding,
        # so the marker must be clamped into view rather than dropped.
        frame = _frame()
        overlay = draw_aim_overlay(
            frame,
            _result(),
            TurretConfig(),
            calibration=_calibration(),
            aim_bearing=(5000.0, 5000.0),
            aim_ok=False,
        )
        assert overlay.sum() > frame.sum()

    def test_off_target_and_on_target_differ(self) -> None:
        frame = _frame()
        on = draw_aim_overlay(
            frame,
            _result(),
            TurretConfig(),
            calibration=_calibration(),
            aim_bearing=(0.0, 0.0),
            aim_ok=True,
        )
        off = draw_aim_overlay(
            frame,
            _result(),
            TurretConfig(),
            calibration=_calibration(),
            aim_bearing=(0.0, 0.0),
            aim_ok=False,
        )
        # The marker colour encodes the state, so the two must not be identical.
        assert not np.array_equal(on, off)

    def test_degenerate_calibration_does_not_raise(self) -> None:
        overlay = draw_aim_overlay(
            _frame(),
            _result(),
            TurretConfig(),
            calibration=Calibration(AffineAxis(0.0, 0.0), AffineAxis(1.0, 0.0)),
            aim_bearing=(0.0, 0.0),
        )
        assert overlay.shape == _frame().shape

    def test_envelope_is_drawn(self) -> None:
        frame = _frame()
        with_envelope = draw_aim_overlay(
            frame,
            _result(with_fire=False),
            TurretConfig(),
            calibration=_calibration(),
        )
        without = draw_aim_overlay(
            frame, _result(with_fire=False), TurretConfig(), calibration=None
        )
        assert with_envelope.sum() > without.sum()


class TestLiveView:
    def test_disabled_by_default(self) -> None:
        assert LiveView(enabled=False).active is False

    def test_show_is_a_no_op_when_disabled(self) -> None:
        # A headless run must not block waiting for a keypress that can never
        # arrive, so a disabled view reports "not shown".
        assert LiveView(enabled=False).show(_frame()) is False

    def test_show_returns_false_without_a_display(self) -> None:
        # Regression guard for a process-killing bug: cv2.imshow on a headless
        # box aborts the interpreter rather than raising, so reaching the end
        # of this test at all is the assertion that matters. The pre-flight
        # check must prevent imshow from ever being called here.
        view = LiveView(enabled=True, title="firewatch-test")
        assert view.show(_frame()) is False
        assert view.active is False

    def test_warns_once_at_construction_not_per_frame(self, caplog) -> None:
        view = LiveView(enabled=True, title="firewatch-test")
        view.show(_frame())
        view.show(_frame())
        warnings = [r for r in caplog.records if "no display is available" in r.message]
        assert len(warnings) == 1

    def test_no_warning_when_explicitly_disabled(self, caplog) -> None:
        LiveView(enabled=False)
        assert not [r for r in caplog.records if "no display" in r.message]

    def test_close_is_safe_when_never_shown(self) -> None:
        LiveView(enabled=False).close()

    def test_display_available_is_false_in_this_environment(self) -> None:
        # The test environment has no display, so the check must say so. This
        # is what stops the abort in the test above from happening. Note this
        # build *does* have a GUI compiled in (see the test below), so the
        # missing DISPLAY is the only thing standing between us and an abort.
        assert display_available() is False

    def test_headless_build_is_detected_from_the_build_string(self, monkeypatch) -> None:
        # Simulate opencv-python-headless, which reports "GUI: NONE".
        headless_info = "  GUI:                           NONE\n"
        monkeypatch.setattr(
            "firewatch.liveview.cv2.getBuildInformation", lambda: headless_info
        )
        assert _opencv_has_gui() is False

    def test_gui_build_is_detected_from_the_build_string(self, monkeypatch) -> None:
        gui_info = "  GUI:                           QT5\n"
        monkeypatch.setattr(
            "firewatch.liveview.cv2.getBuildInformation", lambda: gui_info
        )
        assert _opencv_has_gui() is True

    def test_unreadable_build_string_is_treated_as_headless(self, monkeypatch) -> None:
        # Conservative: an unreadable build string must not lead to calling
        # imshow, because that failure aborts rather than raises.
        def boom() -> str:
            raise RuntimeError("no build info")

        monkeypatch.setattr("firewatch.liveview.cv2.getBuildInformation", boom)
        assert _opencv_has_gui() is False

    def test_no_gui_build_disables_the_view(self, monkeypatch, caplog) -> None:
        monkeypatch.setattr("firewatch.liveview._opencv_has_gui", lambda: False)
        view = LiveView(enabled=True)
        assert view.active is False
        assert view.show(_frame()) is False

    def test_display_available_false_without_display_env(self, monkeypatch) -> None:
        monkeypatch.setattr("firewatch.liveview._opencv_has_gui", lambda: True)
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        monkeypatch.delenv("XDG_SESSION_TYPE", raising=False)
        monkeypatch.setattr("firewatch.liveview.sys.platform", "linux")
        assert display_available() is False

    def test_display_available_true_with_display_env(self, monkeypatch) -> None:
        monkeypatch.setattr("firewatch.liveview._opencv_has_gui", lambda: True)
        monkeypatch.setenv("DISPLAY", ":0")
        monkeypatch.setattr("firewatch.liveview.sys.platform", "linux")
        assert display_available() is True

    def test_display_available_true_on_non_linux(self, monkeypatch) -> None:
        # Windows and macOS have a display attached to an interactive session,
        # so only the GUI build check is meaningful there.
        monkeypatch.setattr("firewatch.liveview._opencv_has_gui", lambda: True)
        monkeypatch.setattr("firewatch.liveview.sys.platform", "win32")
        assert display_available() is True
