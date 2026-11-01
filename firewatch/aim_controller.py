"""Aim control: point the turret at a detection and report whether it worked.

:class:`AimController` is the piece that implements the supervisor's
``AimGate`` protocol. It owns the calibration, drives the turret, and answers
one question per frame: *is the nozzle on the fire?*

The answer is deliberately not "I sent the command, therefore yes". A stepper
mount is open-loop: it can stall, skip steps, or lose its datum, and it will
happily report a position it is not at. So the controller treats the turret's
*reported* position as the only evidence, checks it against the target within
a configured tolerance, and returns a failure when the two disagree. The
supervisor then refuses to spray. That is what makes a mis-aimed turret fail
closed rather than water the wrong thing.

Aim is also *rate-limited*. The controller will not issue a new move more
often than ``settle_seconds`` allows, because commanding a mount faster than
it can physically settle produces a stream of partially-completed moves and a
position reading that means nothing.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

from .aiming import Calibration, CalibrationError, angular_error, solve_bearing
from .config import AppConfig, TurretConfig
from .supervisor import AimOutcome, DenyReason
from .turret import Turret, TurretError, build_turret

logger = logging.getLogger(__name__)


class AimController:
    """Points a turret at a detection box and reports the achieved aim."""

    def __init__(
        self,
        config: TurretConfig,
        turret: Turret,
        calibration: Calibration,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._config = config
        self._turret = turret
        self._calibration = calibration
        self._clock = clock
        self._sleep = sleep

        self._last_move_at: float | None = None
        self._last_bearing: tuple[float, float] | None = None
        self._last_outcome: AimOutcome | None = None
        self.moves = 0

    @property
    def turret(self) -> Turret:
        return self._turret

    @property
    def calibration(self) -> Calibration:
        return self._calibration

    @property
    def last_outcome(self) -> AimOutcome | None:
        """The most recent aim result, for the overlay and for logging."""
        return self._last_outcome

    @property
    def last_bearing(self) -> tuple[float, float] | None:
        """The bearing most recently *achieved*, or None if never aimed."""
        return self._last_bearing

    def home(self) -> AimOutcome:
        """Establish the datum. Aiming is refused until this succeeds."""
        try:
            event = self._turret.home()
        except (TurretError, OSError) as exc:
            logger.error("Turret failed to home: %s", exc)
            outcome = AimOutcome(
                ok=False,
                reason=DenyReason.TURRET_FAULT.value,
                detail=f"homing failed: {exc}",
            )
            self._last_outcome = outcome
            return outcome

        outcome = AimOutcome(
            ok=True,
            detail=f"homed at pan={event.pan:.1f} tilt={event.tilt:.1f}",
            bearing=self._turret.position,
        )
        self._last_outcome = outcome
        return outcome

    def aim_at(
        self,
        bbox: tuple[float, float, float, float],
        frame_size: tuple[int, int],
    ) -> AimOutcome:
        """Point at the centre of ``bbox``; report whether the nozzle got there."""
        if not self._turret.is_homed:
            return self._fail(
                DenyReason.AIM_UNAVAILABLE.value,
                "turret is not homed; run 'firewatch home' first",
            )

        if self._calibration.is_degenerate:
            # A zero scale maps every target to one bearing, so a "successful"
            # aim would be meaningless. Refusing here as well as at build time
            # keeps the interlock honest however the controller was obtained.
            return self._fail(
                DenyReason.AIM_UNAVAILABLE.value,
                "calibration is degenerate; re-run 'firewatch calibrate'",
            )

        try:
            target = solve_bearing(
                self._calibration,
                bbox,
                frame_size,
                min_pan=self._config.min_pan,
                max_pan=self._config.max_pan,
                min_tilt=self._config.min_tilt,
                max_tilt=self._config.max_tilt,
            )
        except CalibrationError as exc:
            return self._fail(DenyReason.AIM_UNAVAILABLE.value, f"calibration unusable: {exc}")

        if target is None:
            return self._fail(
                DenyReason.AIM_OUTSIDE_ENVELOPE.value,
                "fire is outside the turret's travel envelope; refusing to spray",
            )

        if not self._should_move(target):
            # Already settled on this bearing from a previous frame. Re-check
            # rather than re-commanding, so the reported position is fresh.
            return self._verify(target)

        try:
            self._turret.goto(*target)
        except (TurretError, OSError) as exc:
            logger.error("Turret failed to move: %s", exc)
            return self._fail(DenyReason.TURRET_FAULT.value, f"move failed: {exc}")

        self.moves += 1
        self._last_move_at = self._clock()

        # Give the mount time to physically stop before believing its position.
        if self._config.settle_seconds:
            self._sleep(self._config.settle_seconds)

        return self._verify(target)

    def _should_move(self, target: tuple[float, float]) -> bool:
        """Rate-limit moves so the mount is not commanded mid-settle."""
        if self._last_move_at is None or self._last_bearing is None:
            return True
        if self._clock() - self._last_move_at < self._config.settle_seconds:
            return False
        return angular_error(self._last_bearing, target) > self._config.aim_tolerance_degrees

    def _verify(self, target: tuple[float, float]) -> AimOutcome:
        """Check the turret's reported position against the target bearing."""
        actual = self._turret.position
        if actual is None:
            return self._fail(
                DenyReason.AIM_UNAVAILABLE.value,
                "turret reports no position; refusing to spray",
            )

        error = angular_error(actual, target)
        if error > self._config.aim_tolerance_degrees:
            return self._fail(
                DenyReason.AIM_NOT_SETTLED.value,
                f"aim error {error:.1f}deg exceeds tolerance "
                f"{self._config.aim_tolerance_degrees}deg",
                bearing=actual,
                error=error,
            )

        self._last_bearing = actual
        outcome = AimOutcome(
            ok=True,
            detail=f"on target (error {error:.1f}deg)",
            bearing=actual,
            error_degrees=error,
        )
        self._last_outcome = outcome
        return outcome

    def _fail(
        self,
        reason: str,
        detail: str,
        bearing: tuple[float, float] | None = None,
        error: float | None = None,
    ) -> AimOutcome:
        outcome = AimOutcome(
            ok=False,
            reason=reason,
            detail=detail,
            bearing=bearing,
            error_degrees=error,
        )
        self._last_outcome = outcome
        return outcome

    def close(self) -> None:
        self._turret.close()


def build_aim_controller(
    config: AppConfig,
    calibration_path: str | None = None,
) -> AimController | None:
    """Assemble the aiming stack, or None when the turret is disabled.

    Returns None when ``enable_turret`` is false so the detection-only and
    fixed-nozzle paths carry no aiming code at all - there is no mount to
    mis-command, and the supervisor's aim gate stays inactive.
    """
    if not config.turret.enable_turret:
        return None

    turret = build_turret(config.turret)
    if turret is None:
        return None

    path = calibration_path or config.turret.calibration_path
    try:
        calibration = Calibration.load(path)
    except CalibrationError as exc:
        turret.close()
        raise CalibrationError(
            f"{exc} The turret cannot aim without a calibration."
        ) from exc

    if calibration.is_degenerate:
        turret.close()
        raise CalibrationError(
            f"calibration at {path} is degenerate: one axis has no scale, so "
            "every target maps to the same angle. Re-run 'firewatch calibrate' "
            "with the points spread across the frame."
        )

    controller = AimController(config.turret, turret, calibration)
    mode = "DRY RUN" if config.turret.dry_run else "LIVE"
    summary = (
        f"Turret ENABLED ({mode}): pan [{config.turret.min_pan}, {config.turret.max_pan}], "
        f"tilt [{config.turret.min_tilt}, {config.turret.max_tilt}], "
        f"aim tolerance {config.turret.aim_tolerance_degrees}deg"
    )
    if config.turret.dry_run:
        logger.warning("%s", summary)
    else:
        logger.warning("%s", summary)
        logger.warning("LIVE TURRET: this process can now command physical motion")
    return controller
