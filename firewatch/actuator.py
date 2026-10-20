"""Hardware boundary for the suppression device.

Everything physical lives behind :class:`Actuator`. That keeps the decision
logic testable without a valve, and it means the one piece of code that can
actually spray water is small enough to read in full.

Two implementations ship:

* :class:`DryRunActuator` - records what it *would* have done. This is the
  default and the only implementation used until someone deliberately
  configures otherwise.
* :class:`SubprocessActuator` - runs operator-supplied commands. It is
  deliberately generic: this project does not know or care whether the command
  drives a solenoid over GPIO, an MQTT topic, or a relay board. Failures are
  surfaced, never swallowed, because a valve that did not open is exactly the
  situation a human needs to know about.
"""

from __future__ import annotations

import logging
import shlex
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from .config import ActuationConfig

logger = logging.getLogger(__name__)


class ActuatorError(RuntimeError):
    """Raised when a hardware command fails or times out."""


@dataclass(frozen=True)
class ActuatorEvent:
    """One physical command and its outcome, for the audit trail."""

    action: str
    actuator: str
    success: bool
    simulated: bool
    detail: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, object]:
        return {
            "event": "actuator",
            "action": self.action,
            "actuator": self.actuator,
            "success": self.success,
            "simulated": self.simulated,
            "detail": self.detail,
            "timestamp": self.timestamp.isoformat(),
        }


@runtime_checkable
class Actuator(Protocol):
    """A device that can be opened and closed."""

    @property
    def name(self) -> str: ...

    def open_valve(self) -> ActuatorEvent: ...

    def close_valve(self) -> ActuatorEvent: ...


class DryRunActuator:
    """Simulates valve commands and records them.

    Spraying water is not reversible, so the default is to do nothing at all
    and log loudly. ``events`` lets tests and operators confirm the decision
    path fired without any hardware attached.
    """

    def __init__(self, name: str = "dry-run") -> None:
        self._name = name
        self.events: list[ActuatorEvent] = []
        self.is_open = False

    @property
    def name(self) -> str:
        return self._name

    def open_valve(self) -> ActuatorEvent:
        self.is_open = True
        event = ActuatorEvent(
            action="open",
            actuator=self._name,
            success=True,
            simulated=True,
            detail="DRY RUN: valve would open now",
        )
        self.events.append(event)
        logger.warning("DRY RUN: valve OPEN commanded (no hardware attached)")
        return event

    def close_valve(self) -> ActuatorEvent:
        self.is_open = False
        event = ActuatorEvent(
            action="close",
            actuator=self._name,
            success=True,
            simulated=True,
            detail="DRY RUN: valve would close now",
        )
        self.events.append(event)
        logger.info("DRY RUN: valve CLOSE commanded (no hardware attached)")
        return event


class SubprocessActuator:
    """Runs configured commands to drive real hardware.

    ``argv`` is used directly rather than through a shell, so a command
    containing shell metacharacters cannot execute anything unintended.
    """

    def __init__(self, config: ActuationConfig, name: str = "subprocess") -> None:
        if not config.open_command or not config.close_command:
            raise ValueError(
                "SubprocessActuator requires both open_command and close_command"
            )
        self._open_command = list(config.open_command)
        self._close_command = list(config.close_command)
        self._timeout = config.command_timeout_seconds
        self._name = name
        self.is_open = False

    @property
    def name(self) -> str:
        return self._name

    def open_valve(self) -> ActuatorEvent:
        return self._run(self._open_command, "open")

    def close_valve(self) -> ActuatorEvent:
        return self._run(self._close_command, "close")

    def _run(self, command: list[str], action: str) -> ActuatorEvent:
        printable = " ".join(shlex.quote(part) for part in command)
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ActuatorError(
                f"{action} command timed out after {self._timeout}s: {printable}"
            ) from exc
        except OSError as exc:
            raise ActuatorError(f"{action} command could not run: {exc}") from exc

        success = completed.returncode == 0
        detail = (completed.stderr or completed.stdout or "").strip()
        if success:
            self.is_open = action == "open"
            logger.warning("Actuator %s: %s via %s", self._name, action, printable)
        else:
            logger.error(
                "Actuator %s: %s FAILED (exit %s): %s",
                self._name,
                action,
                completed.returncode,
                detail,
            )
        return ActuatorEvent(
            action=action,
            actuator=self._name,
            success=success,
            simulated=False,
            detail=detail or f"exit={completed.returncode}",
        )


def build_actuator(config: ActuationConfig) -> Actuator:
    """Choose an actuator implementation from config.

    Dry-run wins over everything: a live actuator cannot be built while
    ``dry_run`` is true, so a misconfigured ``open_command`` cannot fire by
    accident during evaluation.
    """
    if config.dry_run:
        return DryRunActuator()
    return SubprocessActuator(config)
