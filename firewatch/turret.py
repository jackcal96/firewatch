"""Turret hardware boundary: pointing the nozzle.

This mirrors :mod:`firewatch.actuator` deliberately. Both are thin seams in
front of physical hardware, both default to a dry-run implementation, and
neither knows anything about fire detection or spray policy. Keeping the shape
familiar means the two highest-risk subsystems in the project read the same way.

The unit of position is *degrees*, with pan and tilt measured from the
calibration datum established at homing time:

* ``pan``  - positive clockwise when viewed from above, 0 at the datum.
* ``tilt`` - positive upward, 0 at the datum.

Angles are absolute rather than relative because the nozzle must be aimable at
a known bearing for the spray interlock to mean anything. A relative position
cannot be checked against an envelope, so it cannot be gated on.

Implementations:

* :class:`DryRunTurret` - moves nothing, records every command. The default.
* :class:`SerialTurret` - drives an ESP32 over USB serial (see
  ``firmware/esp32_turret/`` for the matching sketch).

``SerialTurret`` imports pyserial lazily so the rest of the package remains
importable on a machine without it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from .config import TurretConfig

logger = logging.getLogger(__name__)


class TurretError(RuntimeError):
    """Raised when the turret cannot carry out a command."""


@dataclass(frozen=True)
class TurretEvent:
    """One command sent to the turret and its outcome, for the audit trail."""

    action: str
    turret: str
    success: bool
    simulated: bool
    detail: str = ""
    pan: float | None = None
    tilt: float | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, object]:
        return {
            "event": "turret",
            "action": self.action,
            "turret": self.turret,
            "success": self.success,
            "simulated": self.simulated,
            "detail": self.detail,
            "pan": None if self.pan is None else round(self.pan, 2),
            "tilt": None if self.tilt is None else round(self.tilt, 2),
            "timestamp": self.timestamp.isoformat(),
        }


@runtime_checkable
class Turret(Protocol):
    """A two-axis mount that can be pointed at an absolute bearing."""

    @property
    def name(self) -> str: ...

    @property
    def position(self) -> tuple[float, float] | None:
        """Last commanded ``(pan, tilt)`` in degrees, or None if unhomed."""
        ...

    @property
    def is_homed(self) -> bool: ...

    def home(self) -> TurretEvent: ...

    def goto(self, pan: float, tilt: float) -> TurretEvent: ...

    def close(self) -> None: ...


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class DryRunTurret:
    """Tracks an intended position without moving anything.

    This is the default so that a turret can be exercised end to end - aiming,
    envelope checks, overlay, logging - with no hardware attached. It is a
    real implementation, not a stub: the geometry it reports is what the
    aiming code would really command.
    """

    def __init__(self, config: TurretConfig, name: str = "dry-run") -> None:
        self._config = config
        self._name = name
        self._position: tuple[float, float] | None = None
        self._homed = False
        self.events: list[TurretEvent] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def position(self) -> tuple[float, float] | None:
        return self._position

    @property
    def is_homed(self) -> bool:
        return self._homed

    def home(self) -> TurretEvent:
        self._homed = True
        self._position = (self._config.home_pan, self._config.home_tilt)
        event = TurretEvent(
            action="home",
            turret=self._name,
            success=True,
            simulated=True,
            detail=f"DRY RUN: turret homed to {self._position}",
            pan=self._position[0],
            tilt=self._position[1],
        )
        self.events.append(event)
        logger.info("DRY RUN: turret homed to pan=%.1f tilt=%.1f", *self._position)
        return event

    def goto(self, pan: float, tilt: float) -> TurretEvent:
        pan = _clamp(pan, self._config.min_pan, self._config.max_pan)
        tilt = _clamp(tilt, self._config.min_tilt, self._config.max_tilt)
        self._position = (pan, tilt)
        event = TurretEvent(
            action="goto",
            turret=self._name,
            success=True,
            simulated=True,
            detail=f"DRY RUN: turret would move to pan={pan:.1f} tilt={tilt:.1f}",
            pan=pan,
            tilt=tilt,
        )
        self.events.append(event)
        return event

    def close(self) -> None:
        return None


class SerialTurret:
    """Drives a turret controller over a serial line.

    The wire protocol is deliberately line-oriented ASCII, one command per
    line, one response line back. That keeps the ESP32 side small and makes
    the link debuggable with nothing more than a serial terminal:

        ``HOME``          -> ``OK HOME pan=0.0 tilt=0.0``
        ``GOTO <p> <t>``  -> ``OK GOTO pan=<p> tilt=<t>``
        ``POS``           -> ``OK POS pan=<p> tilt=<t>``
        ``PING``          -> ``OK PONG``

    A response beginning with ``ERR`` is surfaced as :class:`TurretError`
    rather than swallowed: a turret that did not move is exactly the situation
    a human needs to know about before water is released.
    """

    def __init__(self, config: TurretConfig, name: str = "serial") -> None:
        if not config.serial_port:
            raise ValueError("SerialTurret requires turret.serial_port to be set")

        try:
            import serial  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise TurretError(
                "pyserial is required to drive a real turret. "
                "Install it with: pip install pyserial"
            ) from exc

        self._config = config
        self._name = name
        self._serial = serial.Serial(
            port=config.serial_port,
            baudrate=config.baud_rate,
            timeout=config.serial_timeout_seconds,
        )
        self._position: tuple[float, float] | None = None
        self._homed = False
        # The controller reboots when the port opens, so wait for it to settle
        # before the first command rather than failing on a spurious timeout.
        time.sleep(config.startup_delay_seconds)
        self._serial.reset_input_buffer()
        logger.info("Turret serial link open on %s @ %s", config.serial_port, config.baud_rate)

    @property
    def name(self) -> str:
        return self._name

    @property
    def position(self) -> tuple[float, float] | None:
        return self._position

    @property
    def is_homed(self) -> bool:
        return self._homed

    def _command(self, line: str) -> str:
        if self._serial is None:
            raise TurretError("serial link is closed")
        self._serial.write(f"{line}\n".encode("ascii"))
        self._serial.flush()
        raw = self._serial.readline()
        if not raw:
            raise TurretError(
                f"no response to {line!r} within {self._config.serial_timeout_seconds}s"
            )
        response = raw.decode("ascii", errors="replace").strip()
        if response.startswith("ERR"):
            raise TurretError(f"turret rejected {line!r}: {response}")
        return response

    @staticmethod
    def _parse_position(response: str) -> tuple[float, float]:
        """Pull ``pan=`` and ``tilt=`` out of a controller response."""
        values: dict[str, float] = {}
        for token in response.split():
            if "=" not in token:
                continue
            key, _, value = token.partition("=")
            if key in {"pan", "tilt"}:
                try:
                    values[key] = float(value)
                except ValueError as exc:
                    raise TurretError(f"malformed {key} in response {response!r}") from exc
        if "pan" not in values or "tilt" not in values:
            raise TurretError(f"response missing pan/tilt: {response!r}")
        return values["pan"], values["tilt"]

    def home(self) -> TurretEvent:
        response = self._command("HOME")
        self._position = self._parse_position(response)
        self._homed = True
        event = TurretEvent(
            action="home",
            turret=self._name,
            success=True,
            simulated=False,
            detail=response,
            pan=self._position[0],
            tilt=self._position[1],
        )
        logger.info("Turret homed at pan=%.1f tilt=%.1f", *self._position)
        return event

    def goto(self, pan: float, tilt: float) -> TurretEvent:
        if not self._homed:
            raise TurretError("turret is not homed; refusing to move to an unknown bearing")
        pan = _clamp(pan, self._config.min_pan, self._config.max_pan)
        tilt = _clamp(tilt, self._config.min_tilt, self._config.max_tilt)
        response = self._command(f"GOTO {pan:.3f} {tilt:.3f}")
        self._position = self._parse_position(response)
        return TurretEvent(
            action="goto",
            turret=self._name,
            success=True,
            simulated=False,
            detail=response,
            pan=self._position[0],
            tilt=self._position[1],
        )

    def close(self) -> None:
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:  # noqa: BLE001 - closing must never mask a real error
                logger.exception("Error closing turret serial link")
            finally:
                self._serial = None


def build_turret(config: TurretConfig) -> Turret | None:
    """Return a turret for this config, or None when the turret is disabled.

    Returns None rather than a dry-run instance when ``enable_turret`` is
    false, so a detection-only deployment carries no turret code at all.
    """
    if not config.enable_turret:
        return None
    if config.dry_run:
        logger.warning("Turret DRY RUN: aiming is simulated, no hardware will move")
        return DryRunTurret(config)
    logger.warning("Turret LIVE: this process can now command physical motion")
    return SerialTurret(config)
